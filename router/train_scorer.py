import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from src.data import load_corpus
from router.shared_utils import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_INSTRUCTION,
    DEFAULT_RESULTS_BASE_DIR,
    DEFAULT_RUNS_DIR,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    ROUTING_INSTRUCTION,
    ROUTING_PREFIX,
    install_file_logger,
    parse_retrievers,
    plot_training_curves,
    resolve_run_dirs,
    sanitize_name,
    set_seed,
    split_specs,
)


DEFAULT_ACTIONS = [
    "rag_1_step",
    "hipporag2-0.5_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "rrf_1_step",
    "no_retrieval_0_step",
]
DEFAULT_DATASETS = ["nq", "musique", "hotpotqa", "2wikimultihopqa"]
METHOD_DIR = "scalar_scorer"
RRF_ACTION = "rrf_1_step"


def parse_csv(raw_value, default_values):
    if raw_value is None:
        return list(default_values)
    if isinstance(raw_value, (list, tuple)):
        return [str(value).strip() for value in raw_value if str(value).strip()]
    value = str(raw_value).strip()
    if not value:
        return list(default_values)
    return [part.strip() for part in value.split(",") if part.strip()]


def load_results_file(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def item_id(item, fallback_idx):
    return item.get("id", f"row_{fallback_idx}")


def f1_value(item):
    if "f1" not in item:
        raise ValueError(f"Missing f1 field for sample {item.get('id', '(unknown id)')}")
    return float(item["f1"])


def top_k_doc_indices(item, corpus_len, top_k):
    indices = [int(idx) for idx in item.get("top_k_indices", [])[:top_k]]
    invalid = [idx for idx in indices if idx < 0 or idx >= corpus_len]
    if invalid:
        raise ValueError(f"Sample {item.get('id', '(unknown id)')} has invalid doc indices: {invalid[:5]}")
    return indices


def docs_from_indices(corpus, indices):
    return [corpus[idx] for idx in indices]


def concatenate_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(f"[Document {idx + 1}]\n{doc}" for idx, doc in enumerate(docs))


def load_split_for_dataset(
    dataset_name,
    split_name,
    split_dir,
    corpus_name,
    action_names,
    results_base_dir,
    top_k,
    f1_gap_threshold,
    max_samples,
):
    corpus = load_corpus(corpus_name)
    action_rows = {}
    action_maps = {}
    ordered_ids = None

    for action_name in action_names:
        result_path = os.path.join(results_base_dir, split_dir, f"{action_name}.jsonl")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"Missing action result file: {result_path}")
        rows = load_results_file(result_path)
        action_rows[action_name] = rows
        action_maps[action_name] = {item_id(item, idx): item for idx, item in enumerate(rows)}
        if ordered_ids is None:
            ordered_ids = [item_id(item, idx) for idx, item in enumerate(rows)]

    examples = []
    skip_counts = {
        "missing_in_action": 0,
        "mismatched_question": 0,
        "invalid_doc_indices": 0,
        "below_f1_gap": 0,
        "row_order_fallback": 0,
    }
    missing_by_action = {name: 0 for name in action_names}
    invalid_by_action = {name: 0 for name in action_names}

    for row_idx, current_id in enumerate(ordered_ids):
        if max_samples is not None and len(examples) >= max_samples:
            break

        first_action = action_names[0]
        first_item = action_maps[first_action].get(current_id)
        if first_item is None:
            if row_idx >= len(action_rows[first_action]):
                skip_counts["missing_in_action"] += 1
                missing_by_action[first_action] += 1
                continue
            first_item = action_rows[first_action][row_idx]
        question = first_item["question"]

        items = [first_item]
        missing_sample = False
        used_row_fallback = False
        for action_name in action_names[1:]:
            item = action_maps[action_name].get(current_id)
            if item is None and row_idx < len(action_rows[action_name]):
                fallback_item = action_rows[action_name][row_idx]
                if fallback_item.get("question") == question:
                    item = fallback_item
                    used_row_fallback = True
            if item is None:
                skip_counts["missing_in_action"] += 1
                missing_by_action[action_name] += 1
                missing_sample = True
                break
            items.append(item)
        if missing_sample:
            continue

        if any(item.get("question") != question for item in items):
            skip_counts["mismatched_question"] += 1
            continue
        if used_row_fallback:
            skip_counts["row_order_fallback"] += 1

        doc_index_lists = []
        doc_texts = []
        f1_scores = []
        valid_sample = True
        for action_name, item in zip(action_names, items):
            try:
                doc_indices = top_k_doc_indices(item, len(corpus), top_k)
            except ValueError:
                invalid_by_action[action_name] += 1
                valid_sample = False
                break
            doc_index_lists.append(doc_indices)
            doc_texts.append(concatenate_docs(docs_from_indices(corpus, doc_indices)))
            f1_scores.append(f1_value(item))
        if not valid_sample:
            skip_counts["invalid_doc_indices"] += 1
            continue

        ranked = sorted(range(len(f1_scores)), key=lambda idx: (-f1_scores[idx], idx))
        f1_gap = f1_scores[ranked[0]] - f1_scores[ranked[1]] if len(ranked) > 1 else 0.0
        if f1_gap < f1_gap_threshold:
            skip_counts["below_f1_gap"] += 1
            continue

        examples.append(
            {
                "id": current_id,
                "dataset": dataset_name,
                "split": split_name,
                "question": question,
                "action_docs": doc_texts,
                "doc_indices": doc_index_lists,
                "label": ranked[0],
                "f1_scores": f1_scores,
                "f1_gap": f1_gap,
            }
        )

    print(
        f"{dataset_name}/{split_name}: loaded {len(examples)} examples from {split_dir}; "
        f"skip_counts={skip_counts}"
    )
    if examples:
        labels = [example["label"] for example in examples]
        print(f"{dataset_name}/{split_name}: label distribution {np.bincount(labels, minlength=len(action_names)).tolist()}")
    if any(missing_by_action.values()):
        print(f"{dataset_name}/{split_name}: missing IDs by action {missing_by_action}")
    if any(invalid_by_action.values()):
        print(f"{dataset_name}/{split_name}: invalid doc-index samples by action {invalid_by_action}")
    return examples


def load_concat_splits(
    dataset_names,
    action_names,
    results_base_dir,
    top_k,
    f1_gap_threshold,
    max_samples,
    split_mode="source",
    resplit_train_ratio=0.8,
    resplit_val_ratio=0.1,
    resplit_test_ratio=0.1,
    seed=42,
):
    if split_mode not in {"source", "resplit_filtered", "resplit_train_val"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    if split_mode == "resplit_filtered":
        source_examples = []
        for dataset_name in dataset_names:
            for split_name, split_dir, corpus_name in split_specs(dataset_name):
                source_examples.extend(
                    load_split_for_dataset(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        split_dir=split_dir,
                        corpus_name=corpus_name,
                        action_names=action_names,
                        results_base_dir=results_base_dir,
                        top_k=top_k,
                        f1_gap_threshold=f1_gap_threshold,
                        max_samples=max_samples,
                    )
                )
        print(f"Filtered source pool before resplitting: {len(source_examples)} examples")
        splits = resplit_filtered_examples(
            examples=source_examples,
            action_names=action_names,
            train_ratio=resplit_train_ratio,
            val_ratio=resplit_val_ratio,
            test_ratio=resplit_test_ratio,
            seed=seed,
        )
        print(
            "Resplit filtered examples with ratios "
            f"train={resplit_train_ratio:g}, val={resplit_val_ratio:g}, test={resplit_test_ratio:g}"
        )
        print_split_summaries(splits, action_names)
        return splits

    if split_mode == "resplit_train_val":
        source_splits = {"train": [], "val": [], "test": []}
        for dataset_name in dataset_names:
            for split_name, split_dir, corpus_name in split_specs(dataset_name):
                source_splits[split_name].extend(
                    load_split_for_dataset(
                        dataset_name=dataset_name,
                        split_name=split_name,
                        split_dir=split_dir,
                        corpus_name=corpus_name,
                        action_names=action_names,
                        results_base_dir=results_base_dir,
                        top_k=top_k,
                        f1_gap_threshold=f1_gap_threshold,
                        max_samples=max_samples,
                    )
                )
        pool_size = len(source_splits["train"]) + len(source_splits["val"])
        print(f"Train+val pool before resplitting: {pool_size} examples (test unchanged: {len(source_splits['test'])})")
        splits = resplit_train_val_examples(
            train_examples=source_splits["train"],
            val_examples=source_splits["val"],
            test_examples=source_splits["test"],
            action_names=action_names,
            train_ratio=resplit_train_ratio,
            seed=seed,
        )
        print(f"Resplit train+val: train={resplit_train_ratio:g}, val={1 - resplit_train_ratio:g} (test unchanged)")
        print_split_summaries(splits, action_names)
        return splits

    splits = {"train": [], "val": [], "test": []}
    for dataset_name in dataset_names:
        for split_name, split_dir, corpus_name in split_specs(dataset_name):
            split_examples = load_split_for_dataset(
                dataset_name=dataset_name,
                split_name=split_name,
                split_dir=split_dir,
                corpus_name=corpus_name,
                action_names=action_names,
                results_base_dir=results_base_dir,
                top_k=top_k,
                f1_gap_threshold=f1_gap_threshold,
                max_samples=max_samples,
            )
            splits[split_name].extend(split_examples)

    print_split_summaries(splits, action_names)
    return splits


def print_split_summaries(splits, action_names):
    for split_name, examples in splits.items():
        print(f"Combined {split_name}: {len(examples)} examples")
        if examples:
            labels = [example["label"] for example in examples]
            print(f"Combined {split_name}: label distribution {np.bincount(labels, minlength=len(action_names)).tolist()}")
            print_data_summary(f"Combined {split_name}", examples, action_names)


def validate_resplit_ratios(train_ratio, val_ratio, test_ratio):
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Resplit ratios must be non-negative.")
    if any(ratio == 0 for ratio in ratios):
        raise ValueError("Resplit train/val/test ratios must all be positive.")
    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Resplit ratios must sum to 1.0, got {ratio_sum}.")


def ratio_counts(n_items, ratios):
    raw_counts = np.asarray([n_items * ratio for ratio in ratios], dtype=float)
    counts = np.floor(raw_counts).astype(int)
    remaining = int(n_items - counts.sum())
    fractional_order = np.argsort(-(raw_counts - counts))
    for idx in fractional_order[:remaining]:
        counts[idx] += 1
    return counts.tolist()


def resplit_filtered_examples(examples, action_names, train_ratio, val_ratio, test_ratio, seed):
    validate_resplit_ratios(train_ratio, val_ratio, test_ratio)
    rng = np.random.default_rng(seed)
    grouped_examples = {}
    for example in examples:
        group_key = (example["dataset"], int(example["label"]))
        grouped_examples.setdefault(group_key, []).append(example)

    splits = {"train": [], "val": [], "test": []}
    split_names = ["train", "val", "test"]
    ratios = [train_ratio, val_ratio, test_ratio]

    for group_key in sorted(grouped_examples):
        group = grouped_examples[group_key]
        order = rng.permutation(len(group))
        shuffled = [group[int(idx)] for idx in order]
        split_counts = ratio_counts(len(shuffled), ratios)
        start_idx = 0
        for split_name, count in zip(split_names, split_counts):
            end_idx = start_idx + count
            for example in shuffled[start_idx:end_idx]:
                resplit_example = dict(example)
                resplit_example["source_split"] = example["split"]
                resplit_example["split"] = split_name
                splits[split_name].append(resplit_example)
            start_idx = end_idx

    for split_name in split_names:
        order = rng.permutation(len(splits[split_name]))
        splits[split_name] = [splits[split_name][int(idx)] for idx in order]

    if len(splits["train"]) == 0 or len(splits["val"]) == 0 or len(splits["test"]) == 0:
        raise ValueError(
            "Resplitting produced an empty train/val/test split. "
            "Use less aggressive filtering or different split ratios."
        )

    return splits


def resplit_train_val_examples(train_examples, val_examples, test_examples, action_names, train_ratio, seed):
    """Merge source train+val, resplit at train_ratio / (1-train_ratio). Test is kept unchanged."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    pool = train_examples + val_examples
    val_ratio = 1.0 - train_ratio
    rng = np.random.default_rng(seed)
    grouped = {}
    for example in pool:
        group_key = (example["dataset"], int(example["label"]))
        grouped.setdefault(group_key, []).append(example)

    new_train, new_val = [], []
    for group_key in sorted(grouped):
        group = grouped[group_key]
        order = rng.permutation(len(group))
        shuffled = [group[int(idx)] for idx in order]
        n_train, n_val = ratio_counts(len(shuffled), [train_ratio, val_ratio])
        for ex in shuffled[:n_train]:
            new_train.append({**ex, "source_split": ex.get("split", "?"), "split": "train"})
        for ex in shuffled[n_train:n_train + n_val]:
            new_val.append({**ex, "source_split": ex.get("split", "?"), "split": "val"})

    final_order = rng.permutation(len(new_train))
    new_train = [new_train[int(i)] for i in final_order]
    final_order = rng.permutation(len(new_val))
    new_val = [new_val[int(i)] for i in final_order]

    if not new_train or not new_val:
        raise ValueError("resplit_train_val produced an empty train or val split.")
    return {"train": new_train, "val": new_val, "test": test_examples}


def print_data_summary(name, examples, action_names):
    f1s = np.asarray([example["f1_scores"] for example in examples], dtype=float)
    labels = np.asarray([example["label"] for example in examples], dtype=int)
    oracle_labels = np.argmax(f1s, axis=1)
    if not np.array_equal(labels, oracle_labels):
        mismatch_count = int((labels != oracle_labels).sum())
        raise ValueError(f"{name}: {mismatch_count} labels do not match argmax F1.")
    baselines = {
        action_name: float(f1s[:, action_idx].mean())
        for action_idx, action_name in enumerate(action_names)
    }
    best_baseline_name = max(baselines, key=baselines.get)
    print(f"{name}: oracle F1={float(f1s.max(axis=1).mean()):.4f}")
    print(f"{name}: best fixed baseline={best_baseline_name} ({baselines[best_baseline_name]:.4f})")
    print(f"{name}: baseline F1 by action={baselines}")


class QwenConcatActionDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=2048, routing_prompt=False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Routing prompt: routing-focused system prompt + instruction, closed think block
        # so the yes-logit at the end of the suffix remains a meaningful training signal.
        if routing_prompt:
            self.prefix_tokens = tokenizer.encode(ROUTING_PREFIX, add_special_tokens=False)
            self.suffix_tokens = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
            self._instruction = ROUTING_INSTRUCTION
        else:
            self.prefix_tokens = tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
            self.suffix_tokens = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
            self._instruction = DEFAULT_INSTRUCTION
        self._encoded_cache = {}

    def __len__(self):
        return len(self.examples)

    def format_instruction(self, query, doc):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self._instruction,
            query=query,
            doc=doc,
        )

    def encode_text(self, query, doc):
        formatted_text = self.format_instruction(query, doc)
        encoded = self.tokenizer(
            formatted_text,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        content_budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if content_budget <= 0:
            raise ValueError(
                f"max_length={self.max_length} does not leave room for the reranker prompt."
            )
        original_length = len(self.prefix_tokens) + len(encoded["input_ids"]) + len(self.suffix_tokens)
        was_truncated = len(encoded["input_ids"]) > content_budget
        if was_truncated and getattr(self.tokenizer, "truncation_side", "right") == "left":
            content_ids = encoded["input_ids"][-content_budget:]
        else:
            content_ids = encoded["input_ids"][:content_budget]
        input_ids = self.prefix_tokens + content_ids + self.suffix_tokens
        attention_mask = [1] * len(input_ids)
        return input_ids, attention_mask, original_length, was_truncated

    def __getitem__(self, idx):
        cached = self._encoded_cache.get(idx)
        if cached is not None:
            return cached

        example = self.examples[idx]
        encoded = [self.encode_text(example["question"], doc) for doc in example["action_docs"]]
        item = {
            "input_ids": [torch.tensor(value[0], dtype=torch.long) for value in encoded],
            "attention_mask": [torch.tensor(value[1], dtype=torch.long) for value in encoded],
            "original_input_lengths": torch.tensor([item[2] for item in encoded], dtype=torch.long),
            "was_truncated": torch.tensor([item[3] for item in encoded], dtype=torch.bool),
            "label": torch.tensor(example["label"], dtype=torch.long),
            "action_f1s": torch.tensor(example["f1_scores"], dtype=torch.float),
        }
        self._encoded_cache[idx] = item
        return item


class QwenConcatCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        if not batch:
            raise ValueError("Cannot collate an empty batch.")
        num_actions = len(batch[0]["input_ids"])
        if any(len(item["input_ids"]) != num_actions for item in batch):
            raise ValueError("Every example must contain the same number of actions.")

        max_length = max(
            candidate.numel()
            for item in batch
            for candidate in item["input_ids"]
        )
        input_ids = torch.full(
            (len(batch), num_actions, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(input_ids)
        for batch_idx, item in enumerate(batch):
            for action_idx, candidate in enumerate(item["input_ids"]):
                length = candidate.numel()
                input_ids[batch_idx, action_idx, :length] = candidate
                attention_mask[batch_idx, action_idx, :length] = item["attention_mask"][action_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "original_input_lengths": torch.stack(
                [item["original_input_lengths"] for item in batch]
            ),
            "was_truncated": torch.stack([item["was_truncated"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "action_f1s": torch.stack([item["action_f1s"] for item in batch]),
        }


def score_action_candidates(model, input_ids, attention_mask, token_true_id, score_batch_size):
    batch_size, num_actions, seq_len = input_ids.shape
    flat_input_ids = input_ids.reshape(batch_size * num_actions, seq_len)
    flat_attention_mask = attention_mask.reshape(batch_size * num_actions, seq_len)
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError(
            "Final-token scoring requires a causal LM exposing `.model` and `.lm_head`."
        )
    score_chunks = []
    for start_idx in range(0, flat_input_ids.size(0), score_batch_size):
        end_idx = min(start_idx + score_batch_size, flat_input_ids.size(0))
        chunk_input_ids = flat_input_ids[start_idx:end_idx]
        chunk_attention_mask = flat_attention_mask[start_idx:end_idx]
        outputs = model.model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        last_token_positions = chunk_attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden_states = hidden_states[batch_indices, last_token_positions, :]
        final_token_logits = model.lm_head(last_hidden_states)
        score_chunks.append(final_token_logits[:, token_true_id].float())
        del outputs, hidden_states, last_hidden_states, final_token_logits
    return torch.cat(score_chunks, dim=0).view(batch_size, num_actions)


def score_and_embed_action_candidates(model, input_ids, attention_mask, token_true_id, score_batch_size):
    """Like score_action_candidates but also returns last hidden states [B, C, H]."""
    batch_size, num_actions, seq_len = input_ids.shape
    flat_input_ids = input_ids.reshape(batch_size * num_actions, seq_len)
    flat_attention_mask = attention_mask.reshape(batch_size * num_actions, seq_len)
    if not hasattr(model, "model") or not hasattr(model, "lm_head"):
        raise TypeError(
            "Final-token scoring requires a causal LM exposing `.model` and `.lm_head`."
        )
    score_chunks = []
    hidden_chunks = []
    for start_idx in range(0, flat_input_ids.size(0), score_batch_size):
        end_idx = min(start_idx + score_batch_size, flat_input_ids.size(0))
        chunk_input_ids = flat_input_ids[start_idx:end_idx]
        chunk_attention_mask = flat_attention_mask[start_idx:end_idx]
        outputs = model.model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        last_token_positions = chunk_attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, last_token_positions, :]  # [chunk, H]
        final_token_logits = model.lm_head(last_hidden)
        score_chunks.append(final_token_logits[:, token_true_id].float())
        hidden_chunks.append(last_hidden.float())
        del outputs, hidden_states, final_token_logits
    scores = torch.cat(score_chunks, dim=0).view(batch_size, num_actions)
    hidden = torch.cat(hidden_chunks, dim=0).view(batch_size, num_actions, -1)
    return scores, hidden


def metric_learning_loss(hidden, f1s):
    """
    Pairwise cosine distance regression: train embeddings so that
    cosine_distance(h_i, h_j) ≈ |F1_i - F1_j| for all retriever pairs.

    hidden: [B, C, H]   f1s: [B, C]  (float, range [0, 1])
    """
    B, C, _ = hidden.shape
    h_norm = F.normalize(hidden, dim=-1)
    sim_matrix = torch.bmm(h_norm, h_norm.transpose(1, 2))   # [B, C, C]
    target_dist = (f1s.unsqueeze(2) - f1s.unsqueeze(1)).abs()  # [B, C, C]
    actual_dist = 1.0 - sim_matrix                             # [B, C, C]
    triu_mask = torch.triu(torch.ones(C, C, device=hidden.device, dtype=torch.bool), diagonal=1)
    return F.mse_loss(actual_dist[:, triu_mask], target_dist[:, triu_mask])


def gradient_l2_norm(model):
    grad_norms = [
        parameter.grad.detach().norm(2).float()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not grad_norms:
        return 0.0
    return float(torch.stack(grad_norms).norm(2).item())


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    token_true_id,
    score_batch_size,
    epoch,
    global_step,
    step_metrics_path,
    log_every_steps,
    wandb_run=None,
    max_grad_norm=0.0,
    label_smoothing=0.0,
    loss_mode="contrastive",
    metric_lambda=1.0,
):
    """
    loss_mode:
      "contrastive" — cross-entropy on yes-logit scores (default)
      "metric"      — pairwise cosine distance regression on hidden states only
      "combined"    — contrastive + metric_lambda * metric
    """
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    loss_ema = None
    os.makedirs(os.path.dirname(step_metrics_path), exist_ok=True)
    use_metric = loss_mode in ("metric", "combined")

    with open(step_metrics_path, "a", encoding="utf-8", buffering=1) as step_handle:
        for epoch_step, batch in enumerate(tqdm(dataloader, desc="Training"), start=1):
            step_started = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_metric:
                scores, hidden = score_and_embed_action_candidates(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_true_id=token_true_id,
                    score_batch_size=score_batch_size,
                )
                f1s = batch["action_f1s"].to(device)
                loss_m = metric_learning_loss(hidden, f1s)
                if loss_mode == "combined":
                    loss_ce = loss_fn(scores, labels)
                    loss = loss_ce + metric_lambda * loss_m
                    loss_ce_value = float(loss_ce.item())
                else:
                    loss = loss_m
                    loss_ce_value = 0.0
                loss_m_value = float(loss_m.item())
            else:
                scores = score_action_candidates(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_true_id=token_true_id,
                    score_batch_size=score_batch_size,
                )
                loss = loss_fn(scores, labels)
                loss_ce_value = float(loss.item())
                loss_m_value = 0.0

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm = gradient_l2_norm(model)
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

            global_step += 1
            loss_value = float(loss.item())
            loss_ema = loss_value if loss_ema is None else 0.05 * loss_value + 0.95 * loss_ema
            total_loss += loss_value

            elapsed = max(time.perf_counter() - step_started, 1e-9)
            batch_examples = int(labels.size(0))
            non_padding_tokens = int(attention_mask.sum().item())
            truncation_fraction = float(batch["was_truncated"].float().mean().item())
            original_lengths = batch["original_input_lengths"].reshape(-1)
            record = {
                "global_step": global_step,
                "epoch": epoch,
                "epoch_step": epoch_step,
                "loss": loss_value,
                "loss_ema": float(loss_ema),
                "loss_ce": loss_ce_value,
                "loss_metric": loss_m_value,
                "learning_rate": learning_rate,
                "gradient_norm": grad_norm,
                "batch_examples": batch_examples,
                "non_padding_tokens": non_padding_tokens,
                "examples_per_second": batch_examples / elapsed,
                "tokens_per_second": non_padding_tokens / elapsed,
                "step_seconds": elapsed,
                "truncation_fraction": truncation_fraction,
                "mean_original_input_tokens": float(original_lengths.float().mean().item()),
                "max_original_input_tokens": int(original_lengths.max().item()),
                "gpu_peak_memory_gib": (
                    float(torch.cuda.max_memory_allocated() / (1024 ** 3))
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
            step_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            step_handle.flush()

            if wandb_run is not None:
                wandb_log = {
                    "train/loss": loss_value,
                    "train/loss_ema": float(loss_ema),
                    "train/learning_rate": learning_rate,
                    "train/gradient_norm": grad_norm,
                    "train/gpu_peak_memory_gib": record["gpu_peak_memory_gib"],
                    "train/examples_per_second": record["examples_per_second"],
                    "train/truncation_fraction": truncation_fraction,
                    "epoch": epoch,
                }
                if use_metric:
                    wandb_log["train/loss_metric"] = loss_m_value
                if loss_mode in ("contrastive", "combined"):
                    wandb_log["train/loss_ce"] = loss_ce_value
                wandb_run.log(wandb_log, step=global_step)

            if global_step == 1 or global_step % log_every_steps == 0:
                extra = ""
                if use_metric:
                    extra = f", ce={loss_ce_value:.4f}, metric={loss_m_value:.4f}"
                print(
                    "Training step "
                    f"{global_step}: loss={loss_value:.4f}, ema={loss_ema:.4f}"
                    f"{extra}, "
                    f"lr={learning_rate:.3e}, grad_norm={grad_norm:.3f}, "
                    f"truncated={truncation_fraction:.1%}, "
                    f"peak_memory={record['gpu_peak_memory_gib']:.2f} GiB",
                    flush=True,
                )

            del scores, loss, input_ids, attention_mask, labels
    return total_loss / max(len(dataloader), 1), global_step


def fixed_action_baselines(action_f1s, action_names):
    return {
        action_name: float(action_f1s[:, action_idx].mean())
        for action_idx, action_name in enumerate(action_names)
    }


def evaluate(model, dataloader, device, action_names, token_true_id, score_batch_size):
    model.eval()
    all_predictions = []
    all_scores = []
    all_labels = []
    all_action_f1s = []
    all_original_input_lengths = []
    truncated_candidates = 0
    total_candidates = 0
    total_loss = 0.0
    total_examples = 0
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_device = batch["label"].to(device)
            scores = score_action_candidates(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_true_id=token_true_id,
                score_batch_size=score_batch_size,
            )
            total_loss += float(loss_fn(scores, labels_device).item())
            total_examples += int(labels_device.size(0))
            predictions = torch.argmax(scores, dim=-1)
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            all_action_f1s.extend(batch["action_f1s"].cpu().numpy().tolist())
            lengths = batch["original_input_lengths"].reshape(-1)
            all_original_input_lengths.extend(lengths.cpu().numpy().tolist())
            truncated_candidates += int(batch["was_truncated"].sum().item())
            total_candidates += int(batch["was_truncated"].numel())
            del input_ids, attention_mask, labels_device, scores, predictions

    predictions = np.array(all_predictions, dtype=int)
    scores = np.array(all_scores, dtype=float)
    labels = np.array(all_labels, dtype=int)
    action_f1s = np.array(all_action_f1s, dtype=float)
    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    scores_for_softmax = scores - np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores_for_softmax)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    sorted_probs = np.sort(probs, axis=1)
    confidences = sorted_probs[:, -1]
    score_margins = sorted_probs[:, -1] - sorted_probs[:, -2] if probs.shape[1] > 1 else sorted_probs[:, -1]
    entropy = -(probs * np.log(np.clip(probs, 1e-12, None))).sum(axis=1)
    correctness = (predictions == labels).astype(float)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (confidences >= lower) & (confidences < upper if upper < 1.0 else confidences <= upper)
        if in_bin.any():
            ece += float(in_bin.mean() * abs(correctness[in_bin].mean() - confidences[in_bin].mean()))

    cls_metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(action_names)))).tolist(),
        "prediction_counts": {
            action_name: int((predictions == action_idx).sum())
            for action_idx, action_name in enumerate(action_names)
        },
        "label_counts": {
            action_name: int((labels == action_idx).sum())
            for action_idx, action_name in enumerate(action_names)
        },
        "confidence": {
            "mean_top1_probability": float(confidences.mean()),
            "mean_top1_top2_margin": float(score_margins.mean()),
            "mean_entropy": float(entropy.mean()),
            "expected_calibration_error_10_bin": ece,
        },
    }
    try:
        if len(np.unique(labels)) > 1:
            cls_metrics["auc_ovr_macro"] = float(roc_auc_score(labels, probs, multi_class="ovr", average="macro"))
        else:
            cls_metrics["auc_ovr_macro"] = None
    except ValueError:
        cls_metrics["auc_ovr_macro"] = None

    row_indices = np.arange(len(predictions))
    selected_f1 = action_f1s[row_indices, predictions]
    oracle_f1 = np.max(action_f1s, axis=1)
    baselines = fixed_action_baselines(action_f1s, action_names)
    best_baseline_name = max(baselines, key=baselines.get)
    rrf_f1 = baselines.get(RRF_ACTION)
    downstream = {
        "selected_f1": float(selected_f1.mean()),
        "oracle_f1": float(oracle_f1.mean()),
        "oracle_gap": float(oracle_f1.mean() - selected_f1.mean()),
        "baseline_f1_by_action": baselines,
        "best_baseline_name": best_baseline_name,
        "best_baseline_f1": float(baselines[best_baseline_name]),
        "improvement_vs_best_baseline": float(selected_f1.mean() - baselines[best_baseline_name]),
        "rrf_f1": None if rrf_f1 is None else float(rrf_f1),
        "improvement_vs_rrf": None if rrf_f1 is None else float(selected_f1.mean() - rrf_f1),
    }
    input_lengths = np.asarray(all_original_input_lengths, dtype=int)
    input_stats = {
        "candidate_count": total_candidates,
        "truncated_candidate_count": truncated_candidates,
        "truncation_fraction": (
            float(truncated_candidates / total_candidates) if total_candidates else 0.0
        ),
        "original_token_length_p50": float(np.percentile(input_lengths, 50)),
        "original_token_length_p90": float(np.percentile(input_lengths, 90)),
        "original_token_length_p95": float(np.percentile(input_lengths, 95)),
        "original_token_length_p99": float(np.percentile(input_lengths, 99)),
        "original_token_length_max": int(input_lengths.max()),
    }
    return {
        "loss": total_loss / max(total_examples, 1),
        "classification": cls_metrics,
        "downstream": downstream,
        "input_stats": input_stats,
    }


def print_eval_summary(prefix, metrics):
    cls = metrics["classification"]
    downstream = metrics["downstream"]
    input_stats = metrics["input_stats"]
    print(
        f"{prefix} - Loss: {metrics['loss']:.4f}, Accuracy: "
        f"{cls['accuracy']:.4f}, Macro F1: {cls['f1_macro']:.4f}, "
        f"Selected F1: {downstream['selected_f1']:.4f}, "
        f"Oracle F1: {downstream['oracle_f1']:.4f}, "
        f"RRF F1: {downstream['rrf_f1']}, "
        f"Truncated candidates: {input_stats['truncation_fraction']:.2%}"
    )
    print(f"{prefix} prediction counts: {cls['prediction_counts']}")
    print(f"{prefix} label counts: {cls['label_counts']}")


def metric_mean_std(values):
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    arr = np.asarray(clean_values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "values": clean_values,
    }


def _wandb_log_eval(wandb_run, prefix, metrics, epoch, global_step):
    if wandb_run is None:
        return
    cls = metrics["classification"]
    downstream = metrics["downstream"]
    input_stats = metrics["input_stats"]
    payload = {
        f"{prefix}/loss": metrics["loss"],
        f"{prefix}/accuracy": cls["accuracy"],
        f"{prefix}/f1_macro": cls["f1_macro"],
        f"{prefix}/selected_f1": downstream["selected_f1"],
        f"{prefix}/oracle_f1": downstream["oracle_f1"],
        f"{prefix}/oracle_gap": downstream["oracle_gap"],
        f"{prefix}/improvement_vs_best_baseline": downstream["improvement_vs_best_baseline"],
        f"{prefix}/mean_confidence": cls["confidence"]["mean_top1_probability"],
        f"{prefix}/score_margin": cls["confidence"]["mean_top1_top2_margin"],
        f"{prefix}/calibration_error": cls["confidence"]["expected_calibration_error_10_bin"],
        f"{prefix}/truncation_fraction": input_stats["truncation_fraction"],
        "epoch": epoch,
    }
    if downstream.get("rrf_f1") is not None:
        payload[f"{prefix}/improvement_vs_rrf"] = downstream["improvement_vs_rrf"]
    for action_name, count in cls["prediction_counts"].items():
        payload[f"{prefix}/pred_count/{sanitize_name(action_name)}"] = count
    wandb_run.log(payload, step=global_step)


def examples_for_splits(splits, split_names):
    examples = []
    for split_name in split_names:
        if split_name not in splits:
            raise ValueError(f"Unknown CV pool split {split_name!r}; expected one of {sorted(splits)}.")
        examples.extend(splits[split_name])
    return examples


def make_cross_validation_folds(examples, num_folds, seed, action_names):
    if num_folds < 2:
        raise ValueError("--cv_folds must be at least 2 when cross-validation is enabled.")
    if len(examples) < num_folds:
        raise ValueError(f"Cannot create {num_folds} folds from only {len(examples)} examples.")

    labels = np.asarray([int(example["label"]) for example in examples], dtype=int)
    label_counts = np.bincount(labels, minlength=len(action_names))
    nonzero_label_counts = label_counts[label_counts > 0]
    can_stratify = len(nonzero_label_counts) > 1 and int(nonzero_label_counts.min()) >= num_folds
    if can_stratify:
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        print(f"Using StratifiedKFold with {num_folds} folds.")
        return list(splitter.split(np.arange(len(examples)), labels))

    splitter = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    print(
        f"Using shuffled KFold with {num_folds} folds because at least one non-empty label "
        f"has fewer than {num_folds} examples. Label counts: {label_counts.tolist()}"
    )
    return list(splitter.split(np.arange(len(examples))))


def load_training_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        print("Gradient checkpointing enabled")

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    if token_true_id is None or token_true_id < 0:
        raise ValueError("Could not resolve token id for 'yes'.")
    return tokenizer, model, torch_dtype, token_true_id


def train_one_run(
    args,
    splits,
    action_names,
    device,
    tokenizer,
    model,
    torch_dtype,
    token_true_id,
    model_dir,
    metrics_dir,
    plots_dir,
    run_label,
    wandb_run=None,
):
    train_dataset = QwenConcatActionDataset(splits["train"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)
    val_dataset = QwenConcatActionDataset(splits["val"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)
    test_dataset = QwenConcatActionDataset(splits["test"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    collator = QwenConcatCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print("=" * 80)
    print(f"TRAINING {run_label}")
    print("=" * 80)
    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval batch size: {args.eval_batch_size}")
    print(f"Score batch size: {args.score_batch_size}")
    print("Candidate scoring: final non-padding token logits only")
    print("Input batching: cached tokenization with dynamic right padding")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max length: {args.max_length}")
    print("=" * 80)

    num_training_steps = max(len(train_loader) * args.epochs, 1)
    num_warmup_steps = int(0.1 * num_training_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    if args.lr_schedule == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_cycles=0.5,
            num_training_steps=num_training_steps,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    best_val_selected_f1 = -float("inf")
    best_state_dict = None
    global_step = 0
    step_metrics_path = os.path.join(metrics_dir, "training_steps.jsonl")
    if os.path.exists(step_metrics_path):
        os.remove(step_metrics_path)
    training_history = {
        "method": METHOD_DIR,
        "run_label": run_label,
        "seed": args.seed,
        "actions": action_names,
        "top_k": args.top_k,
        "f1_gap_threshold": args.f1_gap_threshold,
        "split_mode": args.split_mode,
        "resplit_train_ratio": args.resplit_train_ratio,
        "resplit_val_ratio": args.resplit_val_ratio,
        "resplit_test_ratio": args.resplit_test_ratio,
        "final_eval_checkpoint": args.final_eval_checkpoint,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "score_batch_size": args.score_batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "scheduler": "linear_with_warmup",
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "gradient_clipping": args.max_grad_norm if args.max_grad_norm > 0 else None,
        "dynamic_padding": True,
        "tokenization_cache": True,
        "cuda_empty_cache_per_batch": False,
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_selected_f1": [],
        "val_oracle_f1": [],
        "val_oracle_gap": [],
        "val_best_baseline_f1": [],
        "val_improvement_vs_best_baseline": [],
        "val_truncation_fraction": [],
        "val_mean_confidence": [],
        "val_score_margin": [],
        "val_calibration_error": [],
        "step_metrics_file": step_metrics_path,
        "best_val_selected_f1": 0.0,
        "best_epoch": 0,
    }

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            token_true_id=token_true_id,
            score_batch_size=args.score_batch_size,
            epoch=epoch + 1,
            global_step=global_step,
            step_metrics_path=step_metrics_path,
            log_every_steps=args.log_every_steps,
            wandb_run=wandb_run,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing,
            loss_mode=args.loss_mode,
            metric_lambda=args.metric_lambda,
        )
        print(f"Train loss: {train_loss:.4f}")

        train_metrics = None
        if args.eval_train_each_epoch:
            train_metrics = evaluate(
                model=model,
                dataloader=train_loader,
                device=device,
                action_names=action_names,
                token_true_id=token_true_id,
                score_batch_size=args.score_batch_size,
            )
            print_eval_summary("Train", train_metrics)
            _wandb_log_eval(wandb_run, "train_eval", train_metrics, epoch + 1, global_step)

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            action_names=action_names,
            token_true_id=token_true_id,
            score_batch_size=args.score_batch_size,
        )
        cls = val_metrics["classification"]
        downstream = val_metrics["downstream"]
        print_eval_summary("Validation", val_metrics)
        _wandb_log_eval(wandb_run, "val", val_metrics, epoch + 1, global_step)

        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(float(train_loss))
        training_history["val_loss"].append(float(val_metrics["loss"]))
        training_history["val_accuracy"].append(float(cls["accuracy"]))
        training_history["val_precision"].append(float(cls["precision_macro"]))
        training_history["val_recall"].append(float(cls["recall_macro"]))
        training_history["val_f1"].append(float(cls["f1_macro"]))
        training_history["val_selected_f1"].append(float(downstream["selected_f1"]))
        training_history["val_oracle_f1"].append(float(downstream["oracle_f1"]))
        training_history["val_oracle_gap"].append(float(downstream["oracle_gap"]))
        training_history["val_best_baseline_f1"].append(float(downstream["best_baseline_f1"]))
        training_history["val_improvement_vs_best_baseline"].append(
            float(downstream["improvement_vs_best_baseline"])
        )
        training_history["val_truncation_fraction"].append(
            float(val_metrics["input_stats"]["truncation_fraction"])
        )
        training_history["val_mean_confidence"].append(
            float(cls["confidence"]["mean_top1_probability"])
        )
        training_history["val_score_margin"].append(
            float(cls["confidence"]["mean_top1_top2_margin"])
        )
        training_history["val_calibration_error"].append(
            float(cls["confidence"]["expected_calibration_error_10_bin"])
        )

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "train_classification": None if train_metrics is None else train_metrics["classification"],
            "train_downstream": None if train_metrics is None else train_metrics["downstream"],
            "val_classification": cls,
            "val_downstream": downstream,
            "val_input_stats": val_metrics["input_stats"],
        }
        with open(os.path.join(metrics_dir, f"epoch_{epoch + 1}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(epoch_metrics, f, indent=2)

        if downstream["selected_f1"] > best_val_selected_f1:
            best_val_selected_f1 = downstream["selected_f1"]
            training_history["best_val_selected_f1"] = float(best_val_selected_f1)
            training_history["best_epoch"] = epoch + 1
            print(f"New best validation downstream F1: {best_val_selected_f1:.4f}")
            if args.no_save_model:
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            save_best_checkpoint(model, tokenizer, model_dir, args)
        plot_training_curves(training_history, plots_dir)

    if args.final_eval_checkpoint == "best" and training_history["best_epoch"] > 0:
        if args.no_save_model:
            print("Loading best in-memory checkpoint for final evaluation")
            if best_state_dict is None:
                raise ValueError("Best checkpoint state was not captured in memory.")
            model.load_state_dict(best_state_dict)
            del best_state_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"Loading best checkpoint for final evaluation: {model_dir}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(device)
    else:
        print("Using final epoch model for final evaluation")

    final_train_metrics = None
    if args.eval_train_final:
        print("\n" + "=" * 80)
        print("FINAL TRAIN EVALUATION")
        print("=" * 80)
        final_train_metrics = evaluate(
            model=model,
            dataloader=train_loader,
            device=device,
            action_names=action_names,
            token_true_id=token_true_id,
            score_batch_size=args.score_batch_size,
        )
        print_eval_summary("Train", final_train_metrics)

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        action_names=action_names,
        token_true_id=token_true_id,
        score_batch_size=args.score_batch_size,
    )
    cls = test_metrics["classification"]
    downstream = test_metrics["downstream"]
    print_eval_summary("Test", test_metrics)
    print(f"Best fixed action baseline: {downstream['best_baseline_name']} ({downstream['best_baseline_f1']:.4f})")
    print(f"RRF F1: {downstream['rrf_f1']}")
    print(f"Improvement vs RRF: {downstream['improvement_vs_rrf']}")
    _wandb_log_eval(wandb_run, "test", test_metrics, training_history["best_epoch"], global_step)

    test_metrics_dict = {
        "train_classification": None if final_train_metrics is None else final_train_metrics["classification"],
        "train_downstream": None if final_train_metrics is None else final_train_metrics["downstream"],
        "test_classification": cls,
        "test_downstream": downstream,
        "test_loss": float(test_metrics["loss"]),
        "test_input_stats": test_metrics["input_stats"],
        "test_accuracy": float(cls["accuracy"]),
        "test_f1_macro": float(cls["f1_macro"]),
        "test_downstream_selected_f1": float(downstream["selected_f1"]),
        "test_downstream_oracle_f1": float(downstream["oracle_f1"]),
        "test_downstream_best_baseline_name": downstream["best_baseline_name"],
        "test_downstream_best_baseline_f1": float(downstream["best_baseline_f1"]),
        "test_downstream_improvement_vs_best_baseline": float(downstream["improvement_vs_best_baseline"]),
        "test_downstream_rrf_f1": downstream["rrf_f1"],
        "test_downstream_improvement_vs_rrf": downstream["improvement_vs_rrf"],
        "final_eval_checkpoint": args.final_eval_checkpoint,
    }
    with open(os.path.join(metrics_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_metrics_dict, f, indent=2)

    training_history.update(test_metrics_dict)
    with open(os.path.join(metrics_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)
    plot_training_curves(training_history, plots_dir)

    return model, training_history, test_metrics_dict


def run_cross_validation(args, splits, action_names, device, model_dir, metrics_dir, plots_dir):
    pool_split_names = parse_csv(args.cv_pool_splits, ["train", "val"])
    cv_examples = examples_for_splits(splits, pool_split_names)
    fold_indices = make_cross_validation_folds(
        examples=cv_examples,
        num_folds=args.cv_folds,
        seed=args.seed,
        action_names=action_names,
    )
    print(f"Cross-validation pool splits: {pool_split_names}")
    print(f"Cross-validation pool examples: {len(cv_examples)}")
    print_data_summary("Cross-validation pool", cv_examples, action_names)

    fold_summaries = []
    for fold_idx, (train_indices, val_indices) in enumerate(fold_indices, start=1):
        print("\n" + "=" * 80)
        print(f"CROSS-VALIDATION FOLD {fold_idx}/{args.cv_folds}")
        print("=" * 80)
        fold_splits = {
            "train": [cv_examples[int(idx)] for idx in train_indices],
            "val": [cv_examples[int(idx)] for idx in val_indices],
            "test": splits["test"],
        }
        print_split_summaries({"train": fold_splits["train"], "val": fold_splits["val"], "test": fold_splits["test"]}, action_names)

        fold_model_dir = os.path.join(model_dir, f"fold_{fold_idx}")
        fold_metrics_dir = os.path.join(metrics_dir, "folds", f"fold_{fold_idx}")
        fold_plots_dir = os.path.join(plots_dir, "folds", f"fold_{fold_idx}")

        fold_wandb_run = None
        if wandb is not None and getattr(args, "wandb_project", None):
            os.makedirs(fold_metrics_dir, exist_ok=True)
            fold_dataset_tag = "__".join(sanitize_name(n) for n in parse_csv(args.datasets, DEFAULT_DATASETS))
            fold_wandb_run = wandb.init(
                project=args.wandb_project,
                entity=getattr(args, "wandb_entity", None),
                name=f"{fold_dataset_tag}-{args.lr_schedule}-clip{args.max_grad_norm}-seed{args.seed}-fold{fold_idx}of{args.cv_folds}",
                config={**vars(args), "cv_fold": fold_idx},
                dir=fold_metrics_dir,
                resume="allow",
            )

        tokenizer, model, torch_dtype, token_true_id = load_training_model(args, device)
        print(f"Token ID - 'yes': {token_true_id}")
        model, training_history, test_metrics = train_one_run(
            args=args,
            splits=fold_splits,
            action_names=action_names,
            device=device,
            tokenizer=tokenizer,
            model=model,
            torch_dtype=torch_dtype,
            token_true_id=token_true_id,
            model_dir=fold_model_dir,
            metrics_dir=fold_metrics_dir,
            plots_dir=fold_plots_dir,
            run_label=f"(CROSS-VALIDATION FOLD {fold_idx}/{args.cv_folds})",
            wandb_run=fold_wandb_run,
        )
        fold_summary = {
            "fold": fold_idx,
            "train_examples": int(len(fold_splits["train"])),
            "val_examples": int(len(fold_splits["val"])),
            "test_examples": int(len(fold_splits["test"])),
            "best_epoch": training_history["best_epoch"],
            "best_val_selected_f1": training_history["best_val_selected_f1"],
            "test_downstream_selected_f1": test_metrics["test_downstream_selected_f1"],
            "test_downstream_oracle_f1": test_metrics["test_downstream_oracle_f1"],
            "test_downstream_best_baseline_f1": test_metrics["test_downstream_best_baseline_f1"],
            "test_downstream_improvement_vs_best_baseline": test_metrics["test_downstream_improvement_vs_best_baseline"],
            "test_downstream_rrf_f1": test_metrics["test_downstream_rrf_f1"],
            "test_downstream_improvement_vs_rrf": test_metrics["test_downstream_improvement_vs_rrf"],
            "metrics_dir": fold_metrics_dir,
            "model_dir": None if args.no_save_model else fold_model_dir,
        }
        fold_summaries.append(fold_summary)

        del model, tokenizer
        if fold_wandb_run is not None:
            fold_wandb_run.finish()
            fold_wandb_run = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aggregate = {
        "method": METHOD_DIR,
        "mode": "cross_validation",
        "seed": args.seed,
        "actions": action_names,
        "cv_folds": args.cv_folds,
        "cv_pool_splits": pool_split_names,
        "f1_gap_threshold": args.f1_gap_threshold,
        "split_mode": args.split_mode,
        "folds": fold_summaries,
        "metrics": {
            "best_val_selected_f1": metric_mean_std(
                [fold["best_val_selected_f1"] for fold in fold_summaries]
            ),
            "test_downstream_selected_f1": metric_mean_std(
                [fold["test_downstream_selected_f1"] for fold in fold_summaries]
            ),
            "test_downstream_oracle_f1": metric_mean_std(
                [fold["test_downstream_oracle_f1"] for fold in fold_summaries]
            ),
            "test_downstream_best_baseline_f1": metric_mean_std(
                [fold["test_downstream_best_baseline_f1"] for fold in fold_summaries]
            ),
            "test_downstream_improvement_vs_best_baseline": metric_mean_std(
                [fold["test_downstream_improvement_vs_best_baseline"] for fold in fold_summaries]
            ),
            "test_downstream_rrf_f1": metric_mean_std(
                [fold["test_downstream_rrf_f1"] for fold in fold_summaries]
            ),
            "test_downstream_improvement_vs_rrf": metric_mean_std(
                [fold["test_downstream_improvement_vs_rrf"] for fold in fold_summaries]
            ),
        },
    }
    os.makedirs(metrics_dir, exist_ok=True)
    output_path = os.path.join(metrics_dir, "cv_summary.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 80)
    for metric_name, stats in aggregate["metrics"].items():
        if stats is not None:
            print(f"{metric_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, values={stats['values']}")
    print(f"Saved CV summary to: {output_path}")
    return aggregate


def save_best_checkpoint(model, tokenizer, model_dir, args):
    if args.no_save_model:
        print("Skipping checkpoint save because --no_save_model is set.")
        return
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen reranker with concatenated-doc listwise contrastive loss")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--f1_gap_threshold", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--split_mode",
        choices=["source", "resplit_filtered", "resplit_train_val"],
        default="source",
        help=(
            "source: filter existing train/val/test splits independently. "
            "resplit_filtered: pool all filtered source examples and create new train/val/test splits. "
            "resplit_train_val: merge source train+val and resplit at resplit_train_ratio/(1-resplit_train_ratio); test is kept unchanged."
        ),
    )
    parser.add_argument("--resplit_train_ratio", type=float, default=0.8)
    parser.add_argument("--resplit_val_ratio", type=float, default=0.1)
    parser.add_argument("--resplit_test_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=16,
        help=(
            "Maximum candidate sequences scored in one model call. This does not "
            "change the optimizer batch size."
        ),
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=20,
        help="Print a compact optimization status line every N optimizer steps.",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Gradient clipping max norm. Set to 0 to disable (default).",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor for CrossEntropyLoss (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="contrastive",
        choices=["contrastive", "metric", "combined"],
        help=(
            "Training objective. "
            "'contrastive': cross-entropy on yes-logit scores (original). "
            "'metric': pairwise cosine-distance regression on hidden states, "
            "targeting cosine_dist(h_i,h_j)=|F1_i-F1_j|. "
            "'combined': contrastive + metric_lambda * metric."
        ),
    )
    parser.add_argument(
        "--metric_lambda",
        type=float,
        default=1.0,
        help="Weight for the metric learning loss when loss_mode=combined (default: 1.0).",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eval_train_each_epoch",
        action="store_true",
        help="Evaluate the train split after each epoch for pipeline sanity checks.",
    )
    parser.add_argument(
        "--eval_train_final",
        action="store_true",
        help="Evaluate the train split with the best checkpoint before final test evaluation.",
    )
    parser.add_argument(
        "--count_only",
        action="store_true",
        help="Load data, apply filtering, print counts, and exit before loading the reranker.",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help=(
            "If greater than 1, run k-fold cross-validation instead of a single train/val/test run. "
            "The held-out test split is still evaluated once per fold."
        ),
    )
    parser.add_argument(
        "--cv_pool_splits",
        type=str,
        default="train,val",
        help="Comma-separated filtered splits to pool before creating CV folds.",
    )
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for logs, metrics and plots.")
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help="Base directory for model checkpoints (used when --output_dir is not given).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Exact directory for this run's model checkpoint.")
    parser.add_argument(
        "--balance_train",
        action="store_true",
        help="Undersample majority-label examples in the train split so every label has equal count.",
    )
    parser.add_argument(
        "--no_save_model",
        action="store_true",
        help="Do not write model checkpoints to disk; keep the best checkpoint in CPU memory for final evaluation.",
    )
    parser.add_argument(
        "--final_eval_checkpoint",
        choices=["best", "last"],
        default="best",
        help="Use the best-validation checkpoint or the final epoch model for final train/test evaluation.",
    )
    parser.add_argument(
        "--routing_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use routing-focused system prompt and instruction instead of the default relevance prompt. "
             "The prompt suffix is unchanged, so the yes-logit training signal remains valid.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT"),
        help="W&B project name. W&B logging is disabled when this is absent and WANDB_PROJECT env var is not set.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.getenv("WANDB_ENTITY"),
        help="W&B entity (user or team). Falls back to WANDB_ENTITY env var.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.cv_folds == 1:
        raise ValueError("--cv_folds must be 0 or at least 2.")
    if args.log_every_steps <= 0:
        raise ValueError("--log_every_steps must be positive.")
    set_seed(args.seed)

    dataset_names = parse_csv(args.datasets, DEFAULT_DATASETS)
    action_names = parse_retrievers(args.retrievers) or list(DEFAULT_ACTIONS)
    dataset_tag = "__".join(sanitize_name(name) for name in dataset_names)
    run_suffix = f"f1gap_{args.f1_gap_threshold:g}"
    if args.split_mode == "resplit_filtered":
        run_suffix += (
            f"-resplit_{args.resplit_train_ratio:g}_"
            f"{args.resplit_val_ratio:g}_{args.resplit_test_ratio:g}"
        )
    if args.cv_folds > 1:
        cv_pool_tag = sanitize_name(args.cv_pool_splits.replace(",", "_"))
        run_suffix += f"-cv{args.cv_folds}_{cv_pool_tag}"
    run_suffix = run_suffix.replace(".", "_")
    if args.balance_train:
        run_suffix += "-balanced"

    splits = load_concat_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        results_base_dir=args.results_base_dir,
        top_k=args.top_k,
        f1_gap_threshold=args.f1_gap_threshold,
        max_samples=args.max_samples,
        split_mode=args.split_mode,
        resplit_train_ratio=args.resplit_train_ratio,
        resplit_val_ratio=args.resplit_val_ratio,
        resplit_test_ratio=args.resplit_test_ratio,
        seed=args.seed,
    )
    combined = splits["train"] + splits["val"]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(combined))
    n_train, n_val = ratio_counts(len(combined), [0.8, 0.2])
    splits["train"] = [combined[int(i)] for i in order[:n_train]]
    splits["val"] = [combined[int(i)] for i in order[n_train:n_train + n_val]]
    print(f"80/20 redistribution of train+val — train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")
    if args.balance_train:
        buckets: dict = defaultdict(list)
        for ex in splits["train"]:
            buckets[ex["label"]].append(ex)
        original_dist = [len(buckets[lbl]) for lbl in sorted(buckets)]
        min_count = min(len(v) for v in buckets.values())
        balanced = []
        for lbl in sorted(buckets):
            chosen = rng.choice(len(buckets[lbl]), size=min_count, replace=False)
            balanced.extend(buckets[lbl][int(i)] for i in chosen)
        rng.shuffle(balanced)
        splits["train"] = balanced
        print(
            f"balance_train: {min_count} per label × {len(buckets)} labels = {len(splits['train'])} examples "
            f"(dropped {sum(original_dist) - len(splits['train'])}); original dist={original_dist}"
        )
    if len(splits["train"]) == 0:
        raise ValueError("Training split is empty after filtering.")

    if args.count_only:
        print("Count-only mode complete; exiting before model/tokenizer loading.")
        return

    run_dirs = resolve_run_dirs(
        method_dir_name=METHOD_DIR,
        dataset_name=dataset_tag,
        seed=args.seed,
        retriever_names=action_names,
        run_name_suffix=run_suffix,
        runs_dir=args.runs_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
    )
    model_dir = run_dirs["model_dir"]
    log_dir = run_dirs["log_dir"]
    metrics_dir = run_dirs["metrics_dir"]
    plots_dir = run_dirs["plots_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    restore_logger = install_file_logger(log_file)
    wandb_run = None

    try:
        print(f"Method: {METHOD_DIR}")
        print(f"Datasets: {dataset_names}")
        print(f"Actions: {action_names}")
        print(f"Seed: {args.seed}")
        print(f"Top-k docs: {args.top_k}")
        print(f"F1 gap threshold: {args.f1_gap_threshold}")
        print(f"Split mode: {args.split_mode}")
        if args.split_mode == "resplit_filtered":
            print(
                "Resplit ratios: "
                f"train={args.resplit_train_ratio:g}, "
                f"val={args.resplit_val_ratio:g}, "
                f"test={args.resplit_test_ratio:g}"
            )
        print(f"Logging to: {log_file}")
        print(f"Model directory: {model_dir}")
        print(f"Save model checkpoints: {not args.no_save_model}")
        print(f"Final evaluation checkpoint: {args.final_eval_checkpoint}")
        print(f"Metrics directory: {metrics_dir}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if args.cv_folds > 1:
            print(f"Cross-validation folds: {args.cv_folds}")
            print(f"Cross-validation pool splits: {args.cv_pool_splits}")
            run_cross_validation(
                args=args,
                splits=splits,
                action_names=action_names,
                device=device,
                model_dir=model_dir,
                metrics_dir=metrics_dir,
                plots_dir=plots_dir,
            )
            return

        if wandb is not None and args.wandb_project:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{dataset_tag}-{args.lr_schedule}-clip{args.max_grad_norm}-seed{args.seed}",
                config=vars(args),
                dir=log_dir,
                resume="allow",
            )

        tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
            print("Gradient checkpointing enabled")

        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        if token_true_id is None or token_true_id < 0:
            raise ValueError("Could not resolve token id for 'yes'.")
        print(f"Token ID - 'yes': {token_true_id}")

        train_dataset = QwenConcatActionDataset(splits["train"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)
        val_dataset = QwenConcatActionDataset(splits["val"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)
        test_dataset = QwenConcatActionDataset(splits["test"], tokenizer, max_length=args.max_length, routing_prompt=args.routing_prompt)

        generator = torch.Generator()
        generator.manual_seed(args.seed)
        collator = QwenConcatCollator(tokenizer.pad_token_id)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        print("=" * 80)
        print("TRAINING (CONCAT-DOC LISTWISE CONTRASTIVE RERANKER FT)")
        print("=" * 80)
        print(f"Train examples: {len(train_dataset)}")
        print(f"Val examples: {len(val_dataset)}")
        print(f"Test examples: {len(test_dataset)}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Eval batch size: {args.eval_batch_size}")
        print(f"Score batch size: {args.score_batch_size}")
        print("Candidate scoring: final non-padding token logits only")
        print("Input batching: cached tokenization with dynamic right padding")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Max length: {args.max_length}")
        print("=" * 80)

        num_training_steps = max(len(train_loader) * args.epochs, 1)
        num_warmup_steps = int(0.1 * num_training_steps)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        if args.lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_cycles=0.5,
                num_training_steps=num_training_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )

        best_val_selected_f1 = -float("inf")
        best_state_dict = None
        global_step = 0
        step_metrics_path = os.path.join(metrics_dir, "training_steps.jsonl")
        if os.path.exists(step_metrics_path):
            os.remove(step_metrics_path)
        training_history = {
            "method": METHOD_DIR,
            "datasets": dataset_names,
            "seed": args.seed,
            "actions": action_names,
            "top_k": args.top_k,
            "f1_gap_threshold": args.f1_gap_threshold,
            "split_mode": args.split_mode,
            "resplit_train_ratio": args.resplit_train_ratio,
            "resplit_val_ratio": args.resplit_val_ratio,
            "resplit_test_ratio": args.resplit_test_ratio,
            "final_eval_checkpoint": args.final_eval_checkpoint,
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "score_batch_size": args.score_batch_size,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "scheduler": "linear_with_warmup",
            "num_training_steps": num_training_steps,
            "num_warmup_steps": num_warmup_steps,
            "gradient_clipping": args.max_grad_norm if args.max_grad_norm > 0 else None,
            "dynamic_padding": True,
            "tokenization_cache": True,
            "cuda_empty_cache_per_batch": False,
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_selected_f1": [],
            "val_oracle_f1": [],
            "val_oracle_gap": [],
            "val_best_baseline_f1": [],
            "val_improvement_vs_best_baseline": [],
            "val_truncation_fraction": [],
            "val_mean_confidence": [],
            "val_score_margin": [],
            "val_calibration_error": [],
            "step_metrics_file": step_metrics_path,
            "best_val_selected_f1": 0.0,
            "best_epoch": 0,
        }

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            train_loss, global_step = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                token_true_id=token_true_id,
                score_batch_size=args.score_batch_size,
                epoch=epoch + 1,
                global_step=global_step,
                step_metrics_path=step_metrics_path,
                log_every_steps=args.log_every_steps,
                wandb_run=wandb_run,
                max_grad_norm=args.max_grad_norm,
                label_smoothing=args.label_smoothing,
                loss_mode=args.loss_mode,
                metric_lambda=args.metric_lambda,
            )
            print(f"Train loss: {train_loss:.4f}")

            train_metrics = None
            if args.eval_train_each_epoch:
                train_metrics = evaluate(
                    model=model,
                    dataloader=train_loader,
                    device=device,
                    action_names=action_names,
                    token_true_id=token_true_id,
                    score_batch_size=args.score_batch_size,
                )
                print_eval_summary("Train", train_metrics)
                _wandb_log_eval(wandb_run, "train_eval", train_metrics, epoch + 1, global_step)

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                action_names=action_names,
                token_true_id=token_true_id,
                score_batch_size=args.score_batch_size,
            )
            cls = val_metrics["classification"]
            downstream = val_metrics["downstream"]
            print_eval_summary("Validation", val_metrics)
            _wandb_log_eval(wandb_run, "val", val_metrics, epoch + 1, global_step)

            training_history["epochs"].append(epoch + 1)
            training_history["train_loss"].append(float(train_loss))
            training_history["val_loss"].append(float(val_metrics["loss"]))
            training_history["val_accuracy"].append(float(cls["accuracy"]))
            training_history["val_precision"].append(float(cls["precision_macro"]))
            training_history["val_recall"].append(float(cls["recall_macro"]))
            training_history["val_f1"].append(float(cls["f1_macro"]))
            training_history["val_selected_f1"].append(float(downstream["selected_f1"]))
            training_history["val_oracle_f1"].append(float(downstream["oracle_f1"]))
            training_history["val_oracle_gap"].append(float(downstream["oracle_gap"]))
            training_history["val_best_baseline_f1"].append(float(downstream["best_baseline_f1"]))
            training_history["val_improvement_vs_best_baseline"].append(
                float(downstream["improvement_vs_best_baseline"])
            )
            training_history["val_truncation_fraction"].append(
                float(val_metrics["input_stats"]["truncation_fraction"])
            )
            training_history["val_mean_confidence"].append(
                float(cls["confidence"]["mean_top1_probability"])
            )
            training_history["val_score_margin"].append(
                float(cls["confidence"]["mean_top1_top2_margin"])
            )
            training_history["val_calibration_error"].append(
                float(cls["confidence"]["expected_calibration_error_10_bin"])
            )

            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]),
                "train_classification": None if train_metrics is None else train_metrics["classification"],
                "train_downstream": None if train_metrics is None else train_metrics["downstream"],
                "val_classification": cls,
                "val_downstream": downstream,
                "val_input_stats": val_metrics["input_stats"],
            }
            with open(os.path.join(metrics_dir, f"epoch_{epoch + 1}_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(epoch_metrics, f, indent=2)

            if downstream["selected_f1"] > best_val_selected_f1:
                best_val_selected_f1 = downstream["selected_f1"]
                training_history["best_val_selected_f1"] = float(best_val_selected_f1)
                training_history["best_epoch"] = epoch + 1
                print(f"New best validation downstream F1: {best_val_selected_f1:.4f}")
                if args.no_save_model:
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                save_best_checkpoint(model, tokenizer, model_dir, args)
            plot_training_curves(training_history, plots_dir)

        if args.final_eval_checkpoint == "best" and training_history["best_epoch"] > 0:
            if args.no_save_model:
                print("Loading best in-memory checkpoint for final evaluation")
                if best_state_dict is None:
                    raise ValueError("Best checkpoint state was not captured in memory.")
                model.load_state_dict(best_state_dict)
                del best_state_dict
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print(f"Loading best checkpoint for final evaluation: {model_dir}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                ).to(device)
        else:
            print("Using final epoch model for final evaluation")

        final_train_metrics = None
        if args.eval_train_final:
            print("\n" + "=" * 80)
            print("FINAL TRAIN EVALUATION")
            print("=" * 80)
            final_train_metrics = evaluate(
                model=model,
                dataloader=train_loader,
                device=device,
                action_names=action_names,
                token_true_id=token_true_id,
                score_batch_size=args.score_batch_size,
            )
            print_eval_summary("Train", final_train_metrics)

        print("\n" + "=" * 80)
        print("FINAL TEST EVALUATION")
        print("=" * 80)
        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            action_names=action_names,
            token_true_id=token_true_id,
            score_batch_size=args.score_batch_size,
        )
        cls = test_metrics["classification"]
        downstream = test_metrics["downstream"]
        print_eval_summary("Test", test_metrics)
        print(f"Best fixed action baseline: {downstream['best_baseline_name']} ({downstream['best_baseline_f1']:.4f})")
        print(f"RRF F1: {downstream['rrf_f1']}")
        print(f"Improvement vs RRF: {downstream['improvement_vs_rrf']}")
        _wandb_log_eval(wandb_run, "test", test_metrics, training_history["best_epoch"], global_step)

        test_metrics_dict = {
            "train_classification": None if final_train_metrics is None else final_train_metrics["classification"],
            "train_downstream": None if final_train_metrics is None else final_train_metrics["downstream"],
            "test_classification": cls,
            "test_downstream": downstream,
            "test_loss": float(test_metrics["loss"]),
            "test_input_stats": test_metrics["input_stats"],
            "test_accuracy": float(cls["accuracy"]),
            "test_f1_macro": float(cls["f1_macro"]),
            "test_downstream_selected_f1": float(downstream["selected_f1"]),
            "test_downstream_oracle_f1": float(downstream["oracle_f1"]),
            "test_downstream_best_baseline_name": downstream["best_baseline_name"],
            "test_downstream_best_baseline_f1": float(downstream["best_baseline_f1"]),
            "test_downstream_improvement_vs_best_baseline": float(downstream["improvement_vs_best_baseline"]),
            "test_downstream_rrf_f1": downstream["rrf_f1"],
            "test_downstream_improvement_vs_rrf": downstream["improvement_vs_rrf"],
            "final_eval_checkpoint": args.final_eval_checkpoint,
        }
        with open(os.path.join(metrics_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics_dict, f, indent=2)

        training_history.update(test_metrics_dict)
        with open(os.path.join(metrics_dir, "training_history.json"), "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)
        plot_training_curves(training_history, plots_dir)

        summary = {
            "method": METHOD_DIR,
            "datasets": dataset_names,
            "seed": args.seed,
            "actions": action_names,
            "split_mode": args.split_mode,
            "resplit_train_ratio": args.resplit_train_ratio,
            "resplit_val_ratio": args.resplit_val_ratio,
            "resplit_test_ratio": args.resplit_test_ratio,
            "final_eval_checkpoint": args.final_eval_checkpoint,
            "best_val_selected_f1": float(best_val_selected_f1),
            "best_epoch": training_history["best_epoch"],
            "model_path": None if args.no_save_model else model_dir,
            **test_metrics_dict,
        }
        with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation downstream F1: {best_val_selected_f1:.4f} (epoch {training_history['best_epoch']})")
        print(f"Model saved to: {'(disabled)' if args.no_save_model else model_dir}")
        print(f"Metrics saved to: {metrics_dir}")
        print(f"Logs saved to: {log_dir}")
        print(f"Plots saved to: {plots_dir}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        restore_logger()


if __name__ == "__main__":
    main()
