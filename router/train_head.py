import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from src.data import load_corpus
from src.metric import get_ndcg
from router.shared_utils import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_BASE_DIR,
    DEFAULT_RUNS_DIR,
    ROUTING_INSTRUCTION,
    ROUTING_PREFIX,
    ROUTING_SUFFIX,
    default_run_suffix,
    parse_retrievers,
    plot_training_curves,
    rank_candidates,
    resolve_run_dirs,
    set_seed,
    split_specs,
)


DEFAULT_DATASETS = ["nq", "musique", "hotpotqa", "2wikimultihopqa"]
DEFAULT_EMBEDDING_CACHE_BASE_DIR = "late_inter_embedding_cache"
PREFERRED_ACTION_ORDER = [
    "rag_1_step",
    "hipporag2-0.5_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "rrf_1_step",
    "no_retrieval_0_step",
]
INACTIVE_ACTIONS = {
    "rag_2_step",
    "bge_m3_1_step",
    "reason_moderncolbert_1_step",
    "adaptive_rag",
}


def load_results_file(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def torch_load(path, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def item_id(item, fallback_idx):
    return item.get("id", f"row_{fallback_idx}")


def parse_dataset_names(dataset_arg, fallback_dataset):
    raw = dataset_arg if dataset_arg is not None else fallback_dataset
    if raw is None:
        return DEFAULT_DATASETS
    if isinstance(raw, (list, tuple)):
        names = [str(name).strip() for name in raw if str(name).strip()]
    else:
        names = [name.strip() for name in str(raw).replace(",", " ").split() if name.strip()]
    if not names:
        return DEFAULT_DATASETS
    return names


def dataset_run_name(dataset_names):
    return "__".join(dataset_names)


def default_embedding_cache_dir(base_dir, dataset_names, action_names, label_metric, top_k_docs, max_length, model_name):
    model_tag = model_name.replace("/", "_").replace(",", "__").replace("-", "_").replace(".", "_")
    action_tag = "__".join(name.replace("/", "_").replace(",", "__").replace("-", "_").replace(".", "_") for name in action_names)
    cache_name = f"{dataset_run_name(dataset_names)}__{action_tag}__{label_metric}__top{top_k_docs}__len{max_length}__{model_tag}"
    return os.path.join(base_dir, cache_name)


def discover_actions(base_dir, dataset_names, actions=None):
    explicit_actions = parse_retrievers(actions)
    if explicit_actions is not None:
        return explicit_actions

    ordered_first_names = None
    action_sets = []

    for dataset_name in dataset_names:
        for _, split_dir, _ in split_specs(dataset_name):
            split_path = os.path.join(base_dir, split_dir)
            if not os.path.isdir(split_path):
                raise FileNotFoundError(f"Missing results directory: {split_path}")
            split_names = [
                os.path.splitext(name)[0]
                for name in sorted(os.listdir(split_path))
                if name.endswith(".jsonl")
            ]
            if ordered_first_names is None:
                ordered_first_names = split_names
            action_sets.append(set(split_names))

    if not action_sets or not action_sets[0]:
        raise FileNotFoundError(f"No action .jsonl files found under {base_dir}.")

    common_actions = set.intersection(*action_sets)
    common_actions = {name for name in common_actions if name not in INACTIVE_ACTIONS}
    ordered = [name for name in PREFERRED_ACTION_ORDER if name in common_actions]
    ordered.extend(
        name for name in ordered_first_names
        if name in common_actions and name not in ordered and name not in INACTIVE_ACTIONS
    )
    if not ordered:
        raise FileNotFoundError(
            f"No active action files are present in every selected split for datasets {dataset_names}."
        )
    return ordered


def empty_split_data():
    return {
        "ids": [],
        "dataset_names": [],
        "queries": [],
        "docs_per_action": [],
        "labels": [],
        "f1_scores": [],
        "metric_scores": [],
        "latency_scores": [],
    }


def extend_split_data(target, source):
    for key in target:
        target[key].extend(source[key])


def metric_value(item, metric_name, gold_doc_list, doc_list):
    if metric_name in item:
        return float(item[metric_name])
    if metric_name == "ndcg":
        return float(get_ndcg(gold_doc_list, doc_list, k=5))
    if metric_name == "f1":
        raise ValueError(f"Missing f1 field for sample {item.get('id', '(unknown id)')}")
    raise ValueError(f"Unsupported label metric: {metric_name}")


def docs_for_action_item(item, corpus, top_k_docs):
    top_k_indices = item.get("top_k_indices", [])[:top_k_docs]
    invalid_indices = [idx for idx in top_k_indices if idx < 0 or idx >= len(corpus)]
    if invalid_indices:
        return None, invalid_indices
    docs = [corpus[idx] for idx in top_k_indices]
    return docs if docs else [""], []


def load_split_action_data(
    base_dir,
    split_dir,
    corpus_name,
    dataset_name,
    action_names,
    threshold=0.05,
    max_samples=None,
    label_metric="ndcg",
    top_k_docs=5,
):
    corpus = load_corpus(corpus_name)
    action_results = {}
    action_rows = {}
    ordered_item_ids = None

    print(f"Corpus for {split_dir}: {len(corpus)} docs")
    for action_name in action_names:
        result_path = os.path.join(base_dir, split_dir, f"{action_name}.jsonl")
        print(f"Loading action results from {result_path}...")
        if not os.path.exists(result_path):
            raise FileNotFoundError(result_path)
        results = load_results_file(result_path)
        if not results:
            raise ValueError(f"{result_path} is empty; finish generation before training late_inter.")
        result_map = {item_id(item, idx): item for idx, item in enumerate(results)}
        action_results[action_name] = result_map
        action_rows[action_name] = results

        if ordered_item_ids is None:
            ordered_item_ids = [item_id(item, idx) for idx, item in enumerate(results)]

    queries = []
    docs_per_action = []
    labels = []
    f1_scores = []
    metric_scores = []
    latency_scores = []
    sample_ids = []
    sample_dataset_names = []
    skip_counts = {
        "missing_in_action": 0,
        "mismatched_question": 0,
        "invalid_doc_indices": 0,
        "below_threshold": 0,
        "row_order_fallback": 0,
    }
    missing_by_action = {name: 0 for name in action_names}
    invalid_by_action = {name: 0 for name in action_names}

    for row_idx, current_item_id in enumerate(ordered_item_ids):
        if max_samples is not None and len(queries) >= max_samples:
            break

        first_action = action_names[0]
        first_item = action_results[first_action].get(current_item_id)
        if first_item is None:
            if row_idx >= len(action_rows[first_action]):
                skip_counts["missing_in_action"] += 1
                missing_by_action[first_action] += 1
                continue
            first_item = action_rows[first_action][row_idx]
        question = first_item["question"]

        items = [first_item]
        used_row_fallback = False
        missing_sample = False
        for action_name in action_names[1:]:
            item = action_results[action_name].get(current_item_id)
            if item is None and row_idx < len(action_rows[action_name]):
                row_item = action_rows[action_name][row_idx]
                if row_item.get("question") == question:
                    item = row_item
                    used_row_fallback = True
            if item is None:
                skip_counts["missing_in_action"] += 1
                missing_by_action[action_name] += 1
                missing_sample = True
                break
            items.append(item)
        if missing_sample:
            continue

        if any(item["question"] != question for item in items[1:]):
            skip_counts["mismatched_question"] += 1
            continue
        if used_row_fallback:
            skip_counts["row_order_fallback"] += 1

        metric_values = []
        latency_values = []
        sample_docs = []
        sample_f1_scores = []
        sample_metric_scores = []
        sample_latency_scores = []
        valid_sample = True

        for action_name, item in zip(action_names, items):
            doc_list, invalid_indices = docs_for_action_item(item, corpus, top_k_docs)
            if invalid_indices:
                invalid_by_action[action_name] += 1
                valid_sample = False
                break

            gold_doc_list = item.get("gold_doc_list")
            if gold_doc_list is None and (label_metric not in item or "f1" not in item):
                raise ValueError(f"Missing gold_doc_list needed to compute metrics for {current_item_id}")

            metric = metric_value(item, label_metric, gold_doc_list, doc_list)
            f1 = metric_value(item, "f1", gold_doc_list, doc_list)
            latency = float(item.get("latency", 0.0))
            metric_values.append(metric)
            latency_values.append(latency)
            sample_docs.append(doc_list)
            sample_f1_scores.append(f1)
            sample_metric_scores.append(metric)
            sample_latency_scores.append(latency)

        if not valid_sample:
            skip_counts["invalid_doc_indices"] += 1
            continue

        ranking = rank_candidates(metric_values, latency_values)
        if len(ranking) > 1 and threshold is not None and threshold > 0:
            metric_gap = metric_values[ranking[0]] - metric_values[ranking[1]]
            if metric_gap < threshold:
                skip_counts["below_threshold"] += 1
                continue

        queries.append(question)
        docs_per_action.append(sample_docs)
        labels.append(ranking[0])
        f1_scores.append(sample_f1_scores)
        metric_scores.append(sample_metric_scores)
        latency_scores.append(sample_latency_scores)
        sample_ids.append(current_item_id)
        sample_dataset_names.append(dataset_name)

    print(f"Loaded {len(queries)} action-selection examples from {split_dir}")
    if labels:
        print(f"Label distribution: {np.bincount(labels, minlength=len(action_names)).tolist()}")
    else:
        print(f"No action-selection examples loaded from {split_dir}.")
    print(f"Skip counts for {split_dir}: {skip_counts}")
    if any(missing_by_action.values()):
        print(f"Missing IDs by action for {split_dir}: {missing_by_action}")
    if any(invalid_by_action.values()):
        print(f"Invalid doc-index samples by action for {split_dir}: {invalid_by_action}")

    return {
        "ids": sample_ids,
        "dataset_names": sample_dataset_names,
        "queries": queries,
        "docs_per_action": docs_per_action,
        "labels": labels,
        "f1_scores": f1_scores,
        "metric_scores": metric_scores,
        "latency_scores": latency_scores,
    }


def load_action_splits(dataset_names, action_names, threshold, max_samples, base_dir, label_metric, top_k_docs):
    splits = {split_name: empty_split_data() for split_name in ["train", "val", "test"]}

    for dataset_name in dataset_names:
        print(f"\n{'=' * 80}")
        print(f"Loading {dataset_name} dataset with actions: {action_names}")
        print(f"{'=' * 80}")
        for split_name, split_dir, corpus_name in split_specs(dataset_name):
            print(f"\nLoading {dataset_name}/{split_name} split...")
            split_data = load_split_action_data(
                base_dir=base_dir,
                split_dir=split_dir,
                corpus_name=corpus_name,
                dataset_name=dataset_name,
                action_names=action_names,
                threshold=threshold,
                max_samples=max_samples,
                label_metric=label_metric,
                top_k_docs=top_k_docs,
            )
            extend_split_data(splits[split_name], split_data)
            print(f"{dataset_name}/{split_name.upper()}: {len(split_data['queries'])} examples")

    for split_name, split_data in splits.items():
        print(f"\nCombined {split_name.upper()}: {len(split_data['queries'])} examples")
        if split_data["labels"]:
            print(f"Combined {split_name} label distribution: {np.bincount(split_data['labels'], minlength=len(action_names)).tolist()}")

    return splits


class QwenActionDataset(Dataset):
    def __init__(self, split_data, tokenizer, max_length=2048, top_k_docs=5, routing_prompt=False,
                 track_doc_boundaries=False):
        self.queries = split_data["queries"]
        self.docs_per_action = split_data["docs_per_action"]
        self.labels = split_data["labels"]
        self.action_f1_scores = split_data["f1_scores"]
        self.action_metric_scores = split_data["metric_scores"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k_docs = top_k_docs
        # Track each retained document's own end-token position (one per
        # document, via encode_text_per_document) instead of only the single
        # content_end.
        self.track_doc_boundaries = track_doc_boundaries

        if routing_prompt:
            prefix = ROUTING_PREFIX
            suffix = ROUTING_SUFFIX
            self.default_instruction = ROUTING_INSTRUCTION
        else:
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            self.default_instruction = "Given a web search query, retrieve relevant passages that answer the query"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def __len__(self):
        return len(self.queries)

    def format_instruction(self, query, doc):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=self.default_instruction,
            query=query,
            doc=doc,
        )

    def encode_text(self, query, doc):
        formatted_text = self.format_instruction(query, doc)
        encoded = self.tokenizer(
            formatted_text,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
        )

        # Offset of the last document-content token, before the prompt
        # suffix's fixed control tokens.
        content_end = len(self.prefix_tokens) + len(encoded["input_ids"])

        input_ids = self.prefix_tokens + encoded["input_ids"] + self.suffix_tokens
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return input_ids, attention_mask, content_end

    def encode_text_per_document(self, query, docs):
        """
        Same prompt text as concatenate_docs + encode_text (best-ranked
        document first, "\\n\\n"-separated, "[Document i+1]\\n" headers), but
        tokenized piece-by-piece (like build_joint_prompt_tokens's
        retriever_end_positions in cache_embeddings.py) so each
        document's own last-token position can be tracked exactly, instead
        of only the position of the very last document in the blob.

        Returns (input_ids, attention_mask, content_end, doc_end_offsets).
        doc_end_offsets has exactly self.top_k_docs entries -- the 0-indexed
        position of each retained document's LAST token within input_ids.
        Missing document slots (fewer than top_k_docs docs available, or
        truncation cutting a later document off entirely) repeat the last
        valid offset, so extraction always gets a well-defined, in-bounds
        position and downstream code never needs to special-case a ragged K.
        """
        instruct_query_text = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: ".format(
            instruction=self.default_instruction, query=query,
        )
        instruct_query_ids = self.tokenizer.encode(instruct_query_text, add_special_tokens=False)
        budget = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens) - len(instruct_query_ids)

        content_ids = []
        doc_end_offsets = []
        for i, doc in enumerate(docs):
            piece = f"[Document {i + 1}]\n{doc}" if i == 0 else f"\n\n[Document {i + 1}]\n{doc}"
            piece_ids = self.tokenizer.encode(piece, add_special_tokens=False)
            remaining = budget - len(content_ids)
            if remaining <= 0:
                break
            if len(piece_ids) > remaining:
                piece_ids = piece_ids[:remaining]
            content_ids.extend(piece_ids)
            doc_end_offsets.append(len(self.prefix_tokens) + len(instruct_query_ids) + len(content_ids) - 1)

        if not doc_end_offsets:
            # Degenerate: even the first document didn't fit (budget <= 0).
            # Point at the last instruct/query token so extraction still
            # gets a valid, in-bounds position rather than crashing.
            doc_end_offsets.append(len(self.prefix_tokens) + len(instruct_query_ids) - 1)
        while len(doc_end_offsets) < self.top_k_docs:
            doc_end_offsets.append(doc_end_offsets[-1])

        content_end = len(self.prefix_tokens) + len(instruct_query_ids) + len(content_ids)
        input_ids = self.prefix_tokens + instruct_query_ids + content_ids + self.suffix_tokens
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]

        return input_ids, attention_mask, content_end, doc_end_offsets

    @staticmethod
    def concatenate_docs(docs):
        return "\n\n".join(f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(docs))

    def __getitem__(self, idx):
        query = self.queries[idx]
        candidate_docs = self.docs_per_action[idx]

        action_input_ids = []
        action_attention_masks = []
        action_content_ends = []
        action_doc_end_offsets = [] if self.track_doc_boundaries else None
        for docs in candidate_docs:
            docs = list(docs[: self.top_k_docs]) or [""]
            if self.track_doc_boundaries:
                input_ids, attention_mask, content_end, doc_end_offsets = self.encode_text_per_document(query, docs)
                action_doc_end_offsets.append(doc_end_offsets)
            else:
                concat = self.concatenate_docs(docs)
                input_ids, attention_mask, content_end = self.encode_text(query, concat)
            action_input_ids.append(input_ids)
            action_attention_masks.append(attention_mask)
            action_content_ends.append(content_end)

        item = {
            "input_ids": torch.tensor(action_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(action_attention_masks, dtype=torch.long),
            "content_ends": torch.tensor(action_content_ends, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "action_f1s": torch.tensor(self.action_f1_scores[idx], dtype=torch.float),
            "action_metrics": torch.tensor(self.action_metric_scores[idx], dtype=torch.float),
        }
        if self.track_doc_boundaries:
            item["doc_end_offsets"] = torch.tensor(action_doc_end_offsets, dtype=torch.long)  # [C, top_k_docs]
        return item


class CachedActionEmbeddingDataset(Dataset):
    def __init__(self, cache_path, threshold=0.0):
        self.cache_path = cache_path
        self.data = torch_load(cache_path, map_location="cpu")
        self.original_size = int(self.data["labels"].shape[0])
        self.threshold = float(threshold or 0.0)
        if self.threshold > 0:
            self._filter_by_metric_gap(self.threshold)
        self.filtered_size = int(self.data["labels"].shape[0])

    def __len__(self):
        return int(self.data["labels"].shape[0])

    def _keep_rows(self, keep_indices):
        """Row-subset every per-example entry in self.data, in place."""
        current_size = int(self.data["labels"].shape[0])
        for key, value in list(self.data.items()):
            if torch.is_tensor(value) and value.shape[0] == current_size:
                self.data[key] = value.index_select(0, keep_indices)
            elif isinstance(value, list) and len(value) == current_size:
                self.data[key] = [value[int(idx)] for idx in keep_indices]

    def _filter_by_metric_gap(self, threshold):
        metrics = self.data["action_metrics"].float()
        if metrics.shape[1] < 2:
            return
        top2 = torch.topk(metrics, k=2, dim=1).values
        keep_mask = (top2[:, 0] - top2[:, 1]) >= threshold
        self._keep_rows(torch.nonzero(keep_mask, as_tuple=False).squeeze(1))

    def filter_by_dataset(self, names, keep):
        """
        Restrict (keep=True) or exclude (keep=False) rows by source dataset.

        Used for leave-one-dataset-out: train/val exclude the held-out dataset,
        test keeps only it, so the held-out corpus and query distribution are
        never seen during training. Relies on "dataset_names" being stored per
        row by cache_embeddings.py.
        """
        if "dataset_names" not in self.data:
            raise ValueError(
                f"Cache {self.cache_path} has no per-row 'dataset_names'; "
                "rebuild it with a current cache_embeddings.py."
            )
        names = set(names)
        row_names = self.data["dataset_names"]
        mask = [(name in names) == bool(keep) for name in row_names]
        keep_indices = torch.tensor([i for i, m in enumerate(mask) if m], dtype=torch.long)
        if keep_indices.numel() == 0:
            raise ValueError(
                f"Dataset filter (names={sorted(names)}, keep={keep}) left 0 rows in "
                f"{self.cache_path}. Present datasets: {sorted(set(row_names))}."
            )
        self._keep_rows(keep_indices)
        self.filtered_size = int(self.data["labels"].shape[0])

    def restrict_actions(self, keep_action_indices, num_actions_total):
        """
        Drop retrievers from the candidate axis (leave-one-retriever-out).

        Slices every per-retriever tensor to the retained columns and recomputes
        labels as the argmax over what remains -- which is the correct label for
        a pool that no longer contains the dropped retriever. Applied AFTER the
        F1-gap filter so the example set stays identical to the control run;
        only the number of candidates per example changes.

        Records the retained retrievers' ORIGINAL indices as "action_ids" so a
        head with retriever-identity embeddings still looks up the right ones
        after columns move (without this, column k would silently take identity
        k rather than its own).
        """
        keep = torch.tensor(list(keep_action_indices), dtype=torch.long)
        n_rows = int(self.data["labels"].shape[0])
        for key in ("embeddings", "action_f1s", "action_metrics", "action_ndcg", "yes_logits"):
            value = self.data.get(key)
            if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == num_actions_total:
                self.data[key] = value.index_select(1, keep)
        self.data["labels"] = self.data["action_metrics"].float().argmax(dim=1).long()
        self.data["action_ids"] = keep.view(1, -1).expand(n_rows, keep.numel()).contiguous()

    def __getitem__(self, idx):
        item = {
            "embeddings": self.data["embeddings"][idx],
            "label": self.data["labels"][idx],
            "action_f1s": self.data["action_f1s"][idx],
            "action_metrics": self.data["action_metrics"][idx],
        }
        if "yes_logits" in self.data:
            item["yes_logits"] = self.data["yes_logits"][idx]
        if "action_ndcg" in self.data:
            item["action_ndcg"] = self.data["action_ndcg"][idx]
        if "action_ids" in self.data:
            item["action_ids"] = self.data["action_ids"][idx]
        return item


def load_embedding_cache_metadata(cache_dir):
    metadata_path = os.path.join(cache_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing embedding cache metadata: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_embedding_cache(metadata, dataset_names, action_names, label_metric, threshold, top_k_docs, max_length):
    expected = {
        "datasets": dataset_names,
        "action_names": action_names,
        "label_metric": label_metric,
        "top_k_docs": top_k_docs,
        "max_length": max_length,
    }
    mismatches = []
    for key, expected_value in expected.items():
        cached_value = metadata.get(key)
        if cached_value != expected_value:
            mismatches.append(f"{key}: cache={cached_value!r}, requested={expected_value!r}")
    cached_threshold = float(metadata.get("threshold", 0.0) or 0.0)
    requested_threshold = float(threshold or 0.0)
    if cached_threshold > requested_threshold:
        mismatches.append(
            f"threshold: cache={cached_threshold!r} is stricter than requested={requested_threshold!r}; "
            "rebuild the cache with a lower/equal threshold"
        )
    if mismatches:
        raise ValueError("Embedding cache does not match this run:\n" + "\n".join(mismatches))


def late_inter_logits(encoded, type_emb, cross_encoder, classifier, yes_logits=None, logit_proj=None):
    # encoded: [B, num_classes, hidden_size] — one vector per retriever
    encoded = encoded.float()
    batch_size, num_classes, _ = encoded.shape
    if yes_logits is not None and logit_proj is not None:
        # yes_logits: [B, num_classes] — backbone's yes-token score per retriever
        encoded = encoded + logit_proj(yes_logits.float().unsqueeze(-1))
    if type_emb is not None:
        type_ids = torch.arange(num_classes, device=encoded.device).unsqueeze(0).expand(batch_size, num_classes)
        encoded = encoded + type_emb(type_ids)
    fused = cross_encoder(encoded)
    return classifier(fused).squeeze(-1)


class LateEarlyHybridModel(nn.Module):
    def __init__(
        self,
        base_model,
        hidden_size,
        num_classes,
        action_names,
        n_layers=2,
        n_heads=8,
        dropout=0.1,
        eos_token_id=None,
        encode_batch_size=8,
    ):
        super().__init__()
        self.base = base_model
        self.eos_token_id = eos_token_id
        self.num_classes = num_classes
        self.action_names = list(action_names)
        self.encode_batch_size = encode_batch_size

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.type_emb = nn.Embedding(num_classes, hidden_size)
        self.logit_proj = nn.Linear(1, hidden_size, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def last_token_pool(self, last_hidden, attention_mask):
        lengths = attention_mask.sum(dim=1)
        idx = (lengths - 1).clamp(min=0)
        batch_size, _, hidden_size = last_hidden.size()
        idx_expanded = idx.view(batch_size, 1, 1).expand(batch_size, 1, hidden_size)
        return last_hidden.gather(1, idx_expanded).squeeze(1)

    def encode(self, input_ids, attention_mask):
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = out.hidden_states[-1]
        return self.last_token_pool(last_hidden, attention_mask)

    def encode_flat(self, input_ids, attention_mask):
        if self.encode_batch_size is None or self.encode_batch_size <= 0 or input_ids.size(0) <= self.encode_batch_size:
            return self.encode(input_ids, attention_mask)

        encoded_chunks = []
        for start_idx in range(0, input_ids.size(0), self.encode_batch_size):
            end_idx = start_idx + self.encode_batch_size
            encoded_chunks.append(self.encode(input_ids[start_idx:end_idx], attention_mask[start_idx:end_idx]))
        return torch.cat(encoded_chunks, dim=0)

    def forward(self, input_ids, attention_mask):
        batch_size, num_classes, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * num_classes, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * num_classes, seq_len)

        encoded = self.encode_flat(flat_input_ids, flat_attention_mask)
        encoded = encoded.view(batch_size, num_classes, -1)
        return self.forward_from_embeddings(encoded)

    def forward_from_embeddings(self, encoded, yes_logits=None):
        return late_inter_logits(
            encoded=encoded,
            type_emb=self.type_emb,
            cross_encoder=self.cross_encoder,
            classifier=self.classifier,
            yes_logits=yes_logits,
            logit_proj=self.logit_proj,
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "hybrid_model.bin"))
        config = {
            "hidden_size": self.base.config.hidden_size,
            "num_classes": self.num_classes,
            "action_names": self.action_names,
            "encode_batch_size": self.encode_batch_size,
        }
        with open(os.path.join(save_directory, "hybrid_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)


def _resolve_type_ids(action_ids, batch_size, num_classes, device):
    """
    Which retriever-identity embedding each candidate column should look up.

    Normally column k IS retriever k, so a plain arange is right. After a
    leave-one-retriever-out column drop the columns no longer line up with the
    retriever ids, and the dataset supplies the original ids explicitly.
    """
    if action_ids is None:
        return torch.arange(num_classes, device=device).unsqueeze(0).expand(batch_size, num_classes)
    return action_ids.to(device=device, dtype=torch.long)


class LateInterHeadModel(nn.Module):
    def __init__(self, hidden_size, num_classes, action_names, n_layers=2, n_heads=8, dropout=0.1, use_type_emb=True, ffn_dim=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.action_names = list(action_names)
        self.use_type_emb = use_type_emb
        self.ffn_dim = ffn_dim if ffn_dim is not None else 2 * hidden_size

        self.input_norm = nn.LayerNorm(hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.type_emb = nn.Embedding(num_classes, hidden_size) if use_type_emb else None
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, embeddings, yes_logits=None, action_ids=None):
        # yes_logits ignored: redundant given full hidden state is available
        encoded = self.input_norm(embeddings.float())
        batch_size, num_classes, _ = encoded.shape
        if self.type_emb is not None:
            type_ids = _resolve_type_ids(action_ids, batch_size, num_classes, encoded.device)
            encoded = encoded + self.type_emb(type_ids)
        fused = self.cross_encoder(encoded)
        return self.classifier(fused).squeeze(-1)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "hybrid_model.bin"))
        config = {
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "action_names": self.action_names,
            "cached_embeddings": True,
            "use_type_emb": self.use_type_emb,
            "ffn_dim": self.ffn_dim,
        }
        with open(os.path.join(save_directory, "hybrid_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load_pretrained(cls, load_directory, action_names, device="cpu", **kwargs):
        config_path = os.path.join(load_directory, "hybrid_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        saved_actions = config.get("action_names")
        if saved_actions is not None and list(saved_actions) != list(action_names):
            raise ValueError(
                f"Action name mismatch: checkpoint was trained on {saved_actions}, "
                f"but inference was requested with {list(action_names)}."
            )
        kwargs.setdefault("use_type_emb", config.get("use_type_emb", True))
        kwargs.setdefault("ffn_dim", config.get("ffn_dim", None))
        model = cls(
            hidden_size=config["hidden_size"],
            num_classes=config["num_classes"],
            action_names=action_names,
            **kwargs,
        )
        weights_path = os.path.join(load_directory, "hybrid_model.bin")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        return model.to(device)


class LateInteractionHeadModel(nn.Module):
    """
    Comparison head over multi-vector per-retriever representations.

    Input:  [B, C, K, H] — K token vectors per retriever (one per retrieved
            passage), instead of the single pooled vector the
            LateInterHeadModel consumes.
    Output: [B, C] — one routing logit per retriever.

    Unlike LateInterHeadModel (self-attention over C pooled slots), tokens from
    *different* retrievers attend to each other at the token level, so the head
    can contrast documents across retrievers rather than only their
    collapsed summaries. A retriever-type embedding tags each token with its
    retriever identity and an intra-retriever positional embedding tags its slot
    within the K kept tokens; after cross-retriever attention the K tokens of
    each retriever are pooled back to one vector and scored.
    """

    def __init__(self, hidden_size, num_classes, action_names, num_keep_tokens,
                 n_layers=2, n_heads=8, dropout=0.1, use_type_emb=True, ffn_dim=None,
                 pool="attn", use_yes_logit=False, yes_logit_mode="add"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.action_names = list(action_names)
        self.num_keep_tokens = int(num_keep_tokens)
        self.use_type_emb = use_type_emb
        self.ffn_dim = ffn_dim if ffn_dim is not None else 2 * hidden_size
        self.pool = pool
        self.use_yes_logit = use_yes_logit
        # How the yes-logit enters the head:
        #   "add"      -- project into hidden_size and add as a token feature,
        #                 letting the cross-encoder use it as it sees fit.
        #   "residual" -- add the (aggregated) scalar directly to the output
        #                 logit through a learned gain. At init the head's own
        #                 contribution is near zero, so the model starts as the
        #                 scalar router and learns a correction on top of it.
        #                 Motivated by the representation diagnostic
        #                 (analysis/representation_diagnostics.py): the scalars carry MORE
        #                 routing signal than the representations the head reads,
        #                 so refining them beats replacing them.
        #   "both"     -- both of the above.
        self.yes_logit_mode = yes_logit_mode

        self.input_norm = nn.LayerNorm(hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        # Per-retriever identity (broadcast over that retriever's K tokens) and
        # intra-retriever position (which of the K tokens this is).
        self.type_emb = nn.Embedding(num_classes, hidden_size) if use_type_emb else None
        self.pos_emb = nn.Embedding(self.num_keep_tokens, hidden_size) if self.num_keep_tokens > 1 else None
        # Project the scalar yes-logit
        # (per document, or broadcast per retriever) into hidden_size and add
        # it in as an extra token feature, same additive pattern as
        # type_emb/pos_emb, letting the cross-encoder use it however it wants
        # rather than hand-designing a late-fusion rule. Previously accepted
        # but silently unused by this class; opt-in via use_yes_logit.
        _add = use_yes_logit and yes_logit_mode in ("add", "both")
        _res = use_yes_logit and yes_logit_mode in ("residual", "both")
        self.yes_logit_proj = nn.Linear(1, hidden_size) if _add else None
        if _res:
            # Gain init 1.0 => at initialization the head reproduces the scalar
            # router (up to the near-zero classifier output), so training can
            # only add to it.
            self.yes_logit_gain = nn.Parameter(torch.ones(1))
            self.yes_logit_bias = nn.Parameter(torch.zeros(1))
        else:
            self.yes_logit_gain = None
        if pool == "attn":
            self.pool_query = nn.Parameter(torch.zeros(hidden_size))
        self.classifier = nn.Linear(hidden_size, 1)
        if _res:
            # Start the learned correction at ~0 so the residual dominates
            # initially; otherwise random classifier output swamps the scalar.
            nn.init.zeros_(self.classifier.bias)
            nn.init.normal_(self.classifier.weight, std=1e-3)

    def forward(self, embeddings, yes_logits=None, action_ids=None):
        # embeddings: [B, C, K, H] (or [B, C, H] -> treated as K=1)
        x = embeddings.float()
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [B, C, 1, H]
        B, C, K, H = x.shape
        x = self.input_norm(x)

        if self.type_emb is not None:
            type_ids = _resolve_type_ids(action_ids, B, C, x.device).unsqueeze(-1).expand(B, C, K)
            x = x + self.type_emb(type_ids)
        if self.pos_emb is not None and K > 1:
            pos_ids = torch.arange(K, device=x.device).view(1, 1, K).expand(B, C, K)
            x = x + self.pos_emb(pos_ids)
        if self.yes_logit_proj is not None and yes_logits is not None:
            yl = yes_logits.float()
            if yl.dim() == 2:  # [B, C] scalar per retriever -> broadcast across K
                yl = yl.unsqueeze(-1).expand(B, C, K)
            x = x + self.yes_logit_proj(yl.unsqueeze(-1))  # yl: [B, C, K] -> [B, C, K, 1] -> [B, C, K, H]

        # Cross-retriever attention over the full C*K token set.
        tokens = x.reshape(B, C * K, H)
        fused = self.cross_encoder(tokens).view(B, C, K, H)

        # Pool the K tokens of each retriever back to one vector.
        if self.pool == "mean" or K == 1:
            pooled = fused.mean(dim=2)  # [B, C, H]
        else:  # attention pool with a learned query
            scores = torch.einsum("bckh,h->bck", fused, self.pool_query) / (H ** 0.5)
            weights = torch.softmax(scores, dim=2).unsqueeze(-1)  # [B, C, K, 1]
            pooled = (fused * weights).sum(dim=2)  # [B, C, H]

        out = self.classifier(pooled).squeeze(-1)  # [B, C]

        if self.yes_logit_gain is not None and yes_logits is not None:
            yl = yes_logits.float()
            if yl.dim() == 3:  # [B, C, K] per-document -> one scalar per retriever
                yl = yl.mean(dim=2)
            out = out + self.yes_logit_gain * yl + self.yes_logit_bias

        return out  # [B, C]

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "hybrid_model.bin"))
        config = {
            "model_type": "late_interaction",
            "hidden_size": self.hidden_size,
            "num_classes": self.num_classes,
            "action_names": self.action_names,
            "num_keep_tokens": self.num_keep_tokens,
            "cached_embeddings": True,
            "use_type_emb": self.use_type_emb,
            "ffn_dim": self.ffn_dim,
            "pool": self.pool,
            "use_yes_logit": self.use_yes_logit,
            "yes_logit_mode": self.yes_logit_mode,
        }
        with open(os.path.join(save_directory, "hybrid_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)

    @classmethod
    def load_pretrained(cls, load_directory, action_names, device="cpu", **kwargs):
        config_path = os.path.join(load_directory, "hybrid_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        saved_actions = config.get("action_names")
        if saved_actions is not None and list(saved_actions) != list(action_names):
            raise ValueError(
                f"Action name mismatch: checkpoint was trained on {saved_actions}, "
                f"but inference was requested with {list(action_names)}."
            )
        kwargs.setdefault("use_type_emb", config.get("use_type_emb", True))
        kwargs.setdefault("ffn_dim", config.get("ffn_dim", None))
        kwargs.setdefault("pool", config.get("pool", "attn"))
        kwargs.setdefault("use_yes_logit", config.get("use_yes_logit", False))
        kwargs.setdefault("yes_logit_mode", config.get("yes_logit_mode", "add"))
        kwargs.pop("num_keep_tokens", None)
        model = cls(
            hidden_size=config["hidden_size"],
            num_classes=config["num_classes"],
            action_names=action_names,
            num_keep_tokens=config["num_keep_tokens"],
            **kwargs,
        )
        weights_path = os.path.join(load_directory, "hybrid_model.bin")
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
        return model.to(device)


class YesLogitMLPModel(nn.Module):
    """Routes between retrievers using only their yes-token logit scores as input."""

    def __init__(self, num_classes, hidden_size=64, dropout=0.1, action_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.action_names = list(action_names) if action_names else []
        self.mlp = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, yes_logits):
        return self.mlp(yes_logits.float())

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "hybrid_model.bin"))
        config = {
            "model_type": "logit_mlp",
            "num_classes": self.num_classes,
            "hidden_size": self.mlp[0].out_features,
            "action_names": self.action_names,
        }
        with open(os.path.join(save_directory, "hybrid_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)


def resplit_train_val_cached(train_dataset, val_dataset, train_ratio=0.8, seed=42):
    """Merge train+val cached datasets and resplit at train_ratio/1-train_ratio."""
    combined = {}
    for key in train_dataset.data:
        a = train_dataset.data[key]
        b = val_dataset.data[key]
        if torch.is_tensor(a):
            combined[key] = torch.cat([a, b], dim=0)
        elif isinstance(a, list):
            combined[key] = a + b
    N = combined["labels"].shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    n_train = int(N * train_ratio)
    train_idx = torch.tensor(order[:n_train], dtype=torch.long)
    val_idx = torch.tensor(order[n_train:], dtype=torch.long)

    def subset(data, indices):
        result = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                result[key] = value.index_select(0, indices)
            elif isinstance(value, list):
                result[key] = [value[int(i)] for i in indices]
        return result

    train_ds = CachedActionEmbeddingDataset.__new__(CachedActionEmbeddingDataset)
    train_ds.data = subset(combined, train_idx)
    train_ds.original_size = N
    train_ds.filtered_size = n_train
    train_ds.threshold = train_dataset.threshold
    train_ds.cache_path = train_dataset.cache_path

    val_ds = CachedActionEmbeddingDataset.__new__(CachedActionEmbeddingDataset)
    val_ds.data = subset(combined, val_idx)
    val_ds.original_size = N
    val_ds.filtered_size = N - n_train
    val_ds.threshold = val_dataset.threshold
    val_ds.cache_path = val_dataset.cache_path

    return train_ds, val_ds


def forward_batch(model, batch, device):
    if isinstance(model, YesLogitMLPModel):
        if "yes_logits" not in batch:
            raise ValueError("logit_mlp model requires yes_logits in the cache. Rebuild cache with yes-logit extraction.")
        return model(batch["yes_logits"].to(device))
    yes_logits = batch["yes_logits"].to(device) if "yes_logits" in batch else None
    # Present only when a leave-one-retriever-out run has dropped columns, so
    # the head can map each retained column back to its original retriever id.
    action_ids = batch["action_ids"].to(device) if "action_ids" in batch else None
    if "embeddings" in batch:
        return model(batch["embeddings"].to(device), yes_logits=yes_logits, action_ids=action_ids)
    return model(batch["input_ids"].to(device), batch["attention_mask"].to(device))


def dataset_labels(dataset):
    if isinstance(dataset, CachedActionEmbeddingDataset):
        return dataset.data["labels"].cpu().numpy().astype(int).tolist()
    return [int(label) for label in dataset.labels]


def compute_class_weights(labels, num_classes, mode="balanced"):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    if mode == "none":
        return None, counts.tolist()
    if mode != "balanced":
        raise ValueError(f"Unsupported class weight mode: {mode}")

    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if np.any(nonzero):
        weights[nonzero] = counts.sum() / (float(num_classes) * counts[nonzero])
        weights = weights / weights[nonzero].mean()
    return torch.tensor(weights, dtype=torch.float), counts.tolist()


def _loss_targets(batch, device, target_key):
    """
    Per-retriever supervision targets, [B, C].

    target_key selects WHICH quantity supervises the model, holding everything
    else (example set, filtering, eval) fixed:
      "action_metrics" -- downstream answer utility (default; what the
                          formulation calls for).
      "action_ndcg"    -- retrieval relevance against gold passages, injected
                          offline by inject_ndcg_targets.py. Used
                          for the utility-vs-relevance ablation; note the
                          F1-gap example filter still runs off action_metrics,
                          so both arms see an identical example set and only
                          the training target differs.
    """
    if target_key not in batch:
        raise ValueError(
            f"loss target '{target_key}' not in batch (keys: {sorted(batch.keys())}). "
            f"For action_ndcg, run router/inject_ndcg_targets.py on the cache."
        )
    return batch[target_key].to(device).float()


def compute_training_loss(logits, batch, labels, device, loss_mode, class_weights=None,
                          soft_target_temperature=0.1, target_key="action_metrics"):
    if loss_mode == "ce":
        return torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
    if loss_mode == "expected_metric":
        action_metrics = _loss_targets(batch, device, target_key)
        probs = torch.softmax(logits.float(), dim=-1)
        expected_metric = (probs * action_metrics).sum(dim=-1)
        return -expected_metric.mean()
    if loss_mode == "soft_metric_ce":
        action_metrics = _loss_targets(batch, device, target_key)
        target_probs = torch.softmax(action_metrics / soft_target_temperature, dim=-1)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        return torch.nn.functional.kl_div(log_probs, target_probs, reduction="batchmean")
    # --- Pointwise ablation (no shared normalization) ---------------------
    # Every mode above normalizes ACROSS retrievers (softmax over dim=-1), so
    # the gradient on retriever i depends on every other retriever's score --
    # that shared normalization is the formal content of "the routing decision
    # is comparative". The two modes below remove exactly that and nothing
    # else: each retriever is fit to its own utility independently, with no
    # cross-retriever coupling. Same inputs, same targets, same example set,
    # same eval -- only the coupling is gone. Argmax at inference is unchanged,
    # so the comparison is apples-to-apples.
    if loss_mode == "pointwise_bce":
        targets = _loss_targets(batch, device, target_key).clamp(0.0, 1.0)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), targets)
    if loss_mode == "pointwise_mse":
        targets = _loss_targets(batch, device, target_key).clamp(0.0, 1.0)
        return torch.nn.functional.mse_loss(torch.sigmoid(logits.float()), targets)
    raise ValueError(f"Unsupported loss mode: {loss_mode}")


def _wandb_log_eval(wandb_run, prefix, metrics, epoch, global_step):
    if wandb_run is None:
        return
    cls = metrics["classification"]
    downstream = metrics["downstream"]
    payload = {
        f"{prefix}/accuracy": cls["accuracy"],
        f"{prefix}/f1_macro": cls["f1_macro"],
        f"{prefix}/selected_f1": downstream["selected_f1"],
        f"{prefix}/oracle_f1": downstream["oracle_f1"],
        f"{prefix}/oracle_gap": downstream["oracle_gap"],
        f"{prefix}/improvement_vs_best_baseline": downstream["improvement_vs_best_baseline"],
        "epoch": epoch,
    }
    wandb_run.log(payload, step=global_step)


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    loss_mode,
    class_weights=None,
    soft_target_temperature=0.1,
    wandb_run=None,
    global_step=0,
    target_key="action_metrics",
):
    model.train()
    if hasattr(model, "base") and all(not param.requires_grad for param in model.base.parameters()):
        model.base.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        labels = batch["label"].to(device)

        logits = forward_batch(model, batch, device)
        loss = compute_training_loss(
            logits=logits,
            batch=batch,
            labels=labels,
            device=device,
            loss_mode=loss_mode,
            class_weights=class_weights,
            soft_target_temperature=soft_target_temperature,
            target_key=target_key,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        scheduler.step()

        step_loss = loss.item()
        total_loss += step_loss
        global_step += 1

        if wandb_run is not None:
            wandb_run.log({"train/loss": step_loss}, step=global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(len(dataloader), 1), global_step


def evaluate_multiclass(model, dataloader, device, action_names):
    model.eval()

    all_predictions = []
    all_probs = []
    all_labels = []
    all_action_f1s = []
    all_action_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            logits = forward_batch(model, batch, device)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_predictions.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            all_action_f1s.extend(batch["action_f1s"].cpu().numpy().tolist())
            all_action_metrics.extend(batch["action_metrics"].cpu().numpy().tolist())

    predictions = np.array(all_predictions)
    probs = np.array(all_probs)
    labels = np.array(all_labels)
    action_f1s = np.array(all_action_f1s)
    action_metrics = np.array(all_action_metrics)

    classification_metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(action_names)))).tolist(),
    }

    try:
        if len(np.unique(labels)) > 1:
            classification_metrics["auc_ovr_macro"] = roc_auc_score(
                labels,
                probs,
                multi_class="ovr",
                average="macro",
            )
        else:
            classification_metrics["auc_ovr_macro"] = None
    except ValueError:
            classification_metrics["auc_ovr_macro"] = None

    row_indices = np.arange(len(predictions))
    selected_f1 = action_f1s[row_indices, predictions]
    oracle_f1 = np.max(action_f1s, axis=1)
    selected_metric = action_metrics[row_indices, predictions]
    oracle_metric = np.max(action_metrics, axis=1)

    baseline_f1_by_action = {
        action_name: float(action_f1s[:, idx].mean())
        for idx, action_name in enumerate(action_names)
    }
    baseline_metric_by_action = {
        action_name: float(action_metrics[:, idx].mean())
        for idx, action_name in enumerate(action_names)
    }

    best_baseline_name = max(baseline_f1_by_action, key=baseline_f1_by_action.get)

    # How the router actually distributes its choices, and -- per action -- how
    # often it picks that action when that action is the oracle choice. Under a
    # leave-one-retriever-out run this is the number that says whether a
    # never-trained-on retriever is reachable at all, which mean F1 alone hides.
    oracle_choice = np.argmax(action_f1s, axis=1)
    pick_rate_by_action = {}
    oracle_recall_by_action = {}
    for idx, action_name in enumerate(action_names):
        pick_rate_by_action[action_name] = float((predictions == idx).mean())
        is_oracle = oracle_choice == idx
        oracle_recall_by_action[action_name] = (
            float((predictions[is_oracle] == idx).mean()) if is_oracle.any() else None
        )

    downstream_metrics = {
        "selected_f1": float(selected_f1.mean()),
        "oracle_f1": float(oracle_f1.mean()),
        "oracle_gap": float(oracle_f1.mean() - selected_f1.mean()),
        "selected_label_metric": float(selected_metric.mean()),
        "oracle_label_metric": float(oracle_metric.mean()),
        "oracle_label_metric_gap": float(oracle_metric.mean() - selected_metric.mean()),
        "baseline_f1_by_action": baseline_f1_by_action,
        "baseline_f1_by_retriever": baseline_f1_by_action,
        "baseline_label_metric_by_action": baseline_metric_by_action,
        "best_baseline_name": best_baseline_name,
        "best_baseline_f1": float(baseline_f1_by_action[best_baseline_name]),
        "improvement_vs_best_baseline": float(selected_f1.mean() - baseline_f1_by_action[best_baseline_name]),
        "pick_rate_by_action": pick_rate_by_action,
        "oracle_recall_by_action": oracle_recall_by_action,
    }

    return {
        "classification": classification_metrics,
        "downstream": downstream_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3: train the comparison head over the retrievers' cached representations")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--test_threshold", type=float, default=None,
        help="Utility-gap filter for the TEST split only; defaults to --threshold. Set 0.0 to "
             "report the deployment-realistic number: trained on routable queries, evaluated on "
             "every query including the near-ties no router can win. Cached path only.",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--encode_batch_size", type=int, default=8)
    parser.add_argument(
        "--embedding_cache_dir",
        type=str,
        default=os.getenv("LATE_INTER_EMBEDDING_CACHE_DIR"),
        help="Directory containing cached train.pt/val.pt/test.pt Qwen embeddings. If set, train only the late-inter head.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma- or space-separated base datasets to combine, e.g. nq,musique,hotpotqa,2wikimultihopqa.",
    )
    parser.add_argument("--retrievers", type=str, default="auto")
    parser.add_argument("--actions", type=str, default=None, help="Optional explicit comma-separated action list.")
    parser.add_argument(
        "--holdout_dataset", type=str, default=None,
        help="Leave-one-dataset-out: exclude this dataset from train/val and evaluate on it alone. "
             "The held-out dataset has its own corpus, so this is an unseen-corpus test, not just "
             "an unseen-query-distribution one. Cached path only.",
    )
    parser.add_argument(
        "--holdout_dataset_test_on", type=str, default="holdout", choices=["holdout", "rest"],
        help="Which test rows to evaluate under --holdout_dataset. 'holdout' (default) is the "
             "transfer measurement; 'rest' trains on the same 3 datasets but evaluates in "
             "distribution, giving the reference the transfer number must be read against.",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default=None,
        help="Restrict the TEST split to one dataset without changing training. This is the "
             "correct in-distribution control for --holdout_dataset: it answers 'how much does "
             "the router help on dataset X when X WAS in training', which is what the transfer "
             "number must be compared against. (--holdout_dataset_test_on rest instead evaluates "
             "on the other datasets, a different test set entirely.) Cached path only.",
    )
    parser.add_argument(
        "--holdout_action", type=str, default=None,
        help="Leave-one-retriever-out: drop this retriever from the train/val candidate pool while "
             "keeping it in the test pool, so the router must rank a retrieval mechanism it never "
             "trained on. Cached path only.",
    )
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default=os.getenv("CLASS_WEIGHT_MODE", "balanced"),
        choices=["balanced", "none"],
        help="Use inverse-frequency class weights for cross entropy, or disable with 'none'.",
    )
    parser.add_argument(
        "--loss_mode",
        type=str,
        default=os.getenv("LOSS_MODE", "ce"),
        choices=["ce", "expected_metric", "soft_metric_ce", "pointwise_bce", "pointwise_mse"],
        help="Training objective. ce / expected_metric / soft_metric_ce all normalize ACROSS "
             "retrievers (softmax over the candidate set), i.e. they train the model to compare. "
             "pointwise_bce / pointwise_mse remove exactly that coupling -- each retriever is fit "
             "to its own utility independently -- and exist to ablate the claim "
             "that the routing decision is comparative rather than independently pointwise.",
    )
    parser.add_argument(
        "--loss_target_key",
        type=str,
        default="action_metrics",
        choices=["action_metrics", "action_ndcg"],
        help="Which per-retriever quantity supervises training. action_metrics: downstream answer "
             "utility (default). action_ndcg: retrieval relevance vs gold passages, injected by "
             "inject_ndcg_targets.py (the utility-vs-relevance ablation). The example "
             "filter always runs off action_metrics, so both arms see an identical example set and "
             "only the training target changes.",
    )
    parser.add_argument(
        "--soft_target_temperature",
        type=float,
        default=float(os.getenv("SOFT_TARGET_TEMPERATURE", "0.1")),
        help="Temperature for soft_metric_ce targets built from action metrics; lower is sharper.",
    )
    parser.add_argument("--cross_layers", type=int, default=2)
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--cross_dropout", type=float, default=0.1)
    parser.add_argument("--cross_ffn_dim", type=int, default=2048, help="FFN hidden dim in each TransformerEncoderLayer (default 2048 = 2×hidden_size for Qwen3-0.6B).")
    parser.add_argument(
        "--model_type",
        type=str,
        default="emb_transformer",
        choices=["emb_transformer", "logit_mlp", "late_interaction"],
        help="emb_transformer: full cross-encoder on 1024-dim pooled embeddings (default, one vector/retriever). "
             "logit_mlp: tiny MLP on the 5 yes-token logit scalars only (requires yes_logits in cache). "
             "late_interaction: token-level cross-retriever head over a multi-vector cache "
             "(requires a cache built with --extraction_source document_boundaries).",
    )
    parser.add_argument("--mlp_hidden_size", type=int, default=64, help="Hidden size for logit_mlp model.")
    parser.add_argument(
        "--late_interaction_pool",
        type=str,
        default="attn",
        choices=["attn", "mean"],
        help="How the late_interaction head pools the K per-retriever tokens after cross-retriever attention.",
    )
    parser.add_argument(
        "--use_type_emb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a learnable per-retriever type embedding before the cross-encoder. "
             "Disable (--no_use_type_emb) when using joint-encoded caches to prevent the model "
             "from learning a query-independent retriever prior.",
    )
    parser.add_argument(
        "--use_yes_logit",
        action="store_true",
        default=False,
        help="model_type=late_interaction only. Project the cached per-document yes-logit into "
             "hidden_size and add it as an extra token feature before the cross-encoder (same "
             "additive pattern as type_emb/pos_emb). Requires a cache built with "
             "--cache_per_document_yes_logits (cache_embeddings.py); errors otherwise.",
    )
    parser.add_argument(
        "--yes_logit_mode",
        type=str,
        default="add",
        choices=["add", "residual", "both"],
        help="How --use_yes_logit enters the head. add: project to hidden_size and add as a token "
             "feature. residual: add the per-retriever scalar straight to the output logit via a "
             "learned gain (init 1.0, classifier init ~0), so the head STARTS as the scalar router "
             "and learns a correction -- motivated by the collapse diagnostic showing the scalars "
             "carry more routing signal than the representations. both: both.",
    )
    parser.add_argument(
        "--resplit_train_val",
        action="store_true",
        default=False,
        help="Merge train+val cached splits and resplit 80/20, matching the contrastive reranker setup.",
    )
    parser.add_argument("--freeze_base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--routing_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use routing-focused system prompt and instruction (ROUTING_PREFIX / ROUTING_INSTRUCTION / ROUTING_SUFFIX). "
             "Only applies when NOT using a pre-cached embedding cache.",
    )
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for logs, metrics and plots.")
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help="Base directory for model checkpoints (used when --output_dir is not given).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Exact directory for this run's model checkpoint.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name appended to the run directory, to keep runs apart. "
                             "Defaults to '<label_metric>-<threshold tag>'.")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.getenv("WANDB_PROJECT"),
        help="W&B project name. W&B logging is disabled when absent and WANDB_PROJECT env var is not set.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.getenv("WANDB_ENTITY"),
        help="W&B entity (user or team). Falls back to WANDB_ENTITY env var.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    run_dataset_name = dataset_run_name(dataset_names)
    action_arg = args.actions if args.actions is not None else args.retrievers
    cache_metadata = None
    if args.embedding_cache_dir:
        cache_metadata = load_embedding_cache_metadata(args.embedding_cache_dir)
        action_names = cache_metadata["action_names"]
        if action_arg and action_arg != "auto":
            explicit_action_names = parse_retrievers(action_arg)
            if explicit_action_names != action_names:
                raise ValueError(
                    f"Explicit actions {explicit_action_names} do not match cached actions {action_names}."
                )
    else:
        action_names = discover_actions(args.results_base_dir, dataset_names, action_arg)
    if len(action_names) < 2:
        raise ValueError("Please provide at least two actions for multiclass selection.")
    # The pool the model is TRAINED on; differs from action_names (the pool it is
    # TESTED on) only under --holdout_action.
    train_action_names = list(action_names)

    run_name_suffix = args.run_name if args.run_name is not None else default_run_suffix(args.threshold, args.label_metric)
    run_dirs = resolve_run_dirs(
        method_dir_name="comparison_head",
        dataset_name=run_dataset_name,
        seed=args.seed,
        retriever_names=action_names,
        run_name_suffix=run_name_suffix,
        runs_dir=args.runs_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
    )
    args.output_dir = run_dirs["model_dir"]
    log_dir = run_dirs["log_dir"]
    metrics_dir = run_dirs["metrics_dir"]
    plots_dir = run_dirs["plots_dir"]

    pretrained_model_path = args.pretrained_model_path

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_f = open(log_file, "w", encoding="utf-8")

    class Tee:
        def __init__(self, file, stream):
            self.file = file
            self.stream = stream

        def write(self, text):
            self.file.write(text)
            self.stream.write(text)
            self.file.flush()
            self.stream.flush()

        def flush(self):
            self.file.flush()
            self.stream.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    TeeStdout = Tee
    TeeStderr = Tee

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStdout(log_f, sys.stdout)
    sys.stderr = TeeStderr(log_f, sys.stderr)

    print(f"Datasets: {dataset_names}")
    print(f"Dataset run name: {run_dataset_name}")
    print(f"Seed: {args.seed}")
    print(f"Actions: {action_names}")
    print(f"Label metric: {args.label_metric}")
    print(f"Top-k docs per action: {args.top_k_docs}")
    print(f"Encode batch size: {args.encode_batch_size}")
    print(f"Embedding cache dir: {args.embedding_cache_dir if args.embedding_cache_dir else '(disabled)'}")
    print(f"Logging to: {log_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metrics directory: {metrics_dir}")
    print(f"Run name: {run_dirs['run_name']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = None
    if args.embedding_cache_dir:
        print("Loading cached embedding datasets...")
        validate_embedding_cache(
            metadata=cache_metadata,
            dataset_names=dataset_names,
            action_names=action_names,
            label_metric=args.label_metric,
            threshold=args.threshold,
            top_k_docs=args.top_k_docs,
            max_length=args.max_length,
        )
        train_dataset = CachedActionEmbeddingDataset(os.path.join(args.embedding_cache_dir, "train.pt"), threshold=args.threshold)
        val_dataset = CachedActionEmbeddingDataset(os.path.join(args.embedding_cache_dir, "val.pt"), threshold=args.threshold)
        test_threshold = args.threshold if args.test_threshold is None else args.test_threshold
        test_dataset = CachedActionEmbeddingDataset(os.path.join(args.embedding_cache_dir, "test.pt"), threshold=test_threshold)
        if test_threshold != args.threshold:
            print(f"Test split uses a different utility-gap filter: {test_threshold} (train/val: {args.threshold})")
        print(
            "Cached split sizes after threshold filtering: "
            f"train={train_dataset.filtered_size}/{train_dataset.original_size}, "
            f"val={val_dataset.filtered_size}/{val_dataset.original_size}, "
            f"test={test_dataset.filtered_size}/{test_dataset.original_size}"
        )

        # --- Generalization holdouts -------------------------------------
        # Applied AFTER the F1-gap filter so the surviving example set matches
        # the no-holdout control, and BEFORE the train/val resplit so held-out
        # rows cannot leak back in through the resplit.
        if args.holdout_dataset:
            if args.holdout_dataset not in dataset_names:
                raise ValueError(
                    f"--holdout_dataset {args.holdout_dataset!r} is not among the trained datasets {dataset_names}."
                )
            test_on_holdout = args.holdout_dataset_test_on == "holdout"
            train_dataset.filter_by_dataset([args.holdout_dataset], keep=False)
            val_dataset.filter_by_dataset([args.holdout_dataset], keep=False)
            test_dataset.filter_by_dataset([args.holdout_dataset], keep=test_on_holdout)
            print(
                f"Leave-one-dataset-out ({args.holdout_dataset}, test_on={args.holdout_dataset_test_on}): "
                f"train={len(train_dataset)}, val={len(val_dataset)} (that dataset removed), "
                f"test={len(test_dataset)} "
                f"({'that dataset only -- transfer' if test_on_holdout else 'the other datasets -- in-distribution reference'})"
            )

        if args.eval_dataset:
            if args.holdout_dataset:
                raise ValueError(
                    "--eval_dataset and --holdout_dataset are alternatives, not a pair: the first "
                    "is the in-distribution control for the second. Pass exactly one."
                )
            if args.eval_dataset not in dataset_names:
                raise ValueError(
                    f"--eval_dataset {args.eval_dataset!r} is not among the trained datasets {dataset_names}."
                )
            test_dataset.filter_by_dataset([args.eval_dataset], keep=True)
            print(
                f"In-distribution per-dataset eval ({args.eval_dataset}): trained on all "
                f"{len(dataset_names)} datasets, test={len(test_dataset)} rows from that dataset only"
            )

        if args.holdout_action:
            if args.holdout_action not in action_names:
                raise ValueError(
                    f"--holdout_action {args.holdout_action!r} is not among the cached retrievers {action_names}."
                )
            holdout_idx = action_names.index(args.holdout_action)
            keep_idx = [i for i in range(len(action_names)) if i != holdout_idx]
            train_action_names = [action_names[i] for i in keep_idx]
            # Test keeps ALL retrievers: the point is that the router must rank
            # a mechanism it never saw, not that it never encounters one.
            train_dataset.restrict_actions(keep_idx, len(action_names))
            val_dataset.restrict_actions(keep_idx, len(action_names))
            print(
                f"Leave-one-retriever-out ({args.holdout_action}): "
                f"train/val pool = {train_action_names}; test pool = {action_names}"
            )

        if args.resplit_train_val:
            train_dataset, val_dataset = resplit_train_val_cached(
                train_dataset, val_dataset, train_ratio=0.8, seed=args.seed
            )
            print(f"80/20 resplit of train+val — train: {len(train_dataset)}, val: {len(val_dataset)}")
        model_path = cache_metadata.get("model_path", cache_metadata.get("model_name", args.model_name))
        if args.model_type == "logit_mlp":
            model = YesLogitMLPModel(
                num_classes=len(action_names),
                hidden_size=args.mlp_hidden_size,
                dropout=args.cross_dropout,
                action_names=action_names,
            ).to(device)
        elif args.model_type == "late_interaction":
            num_keep_tokens = int(cache_metadata.get("num_keep_tokens", 1))
            if num_keep_tokens < 2:
                raise ValueError(
                    "model_type=late_interaction expects a multi-vector cache "
                    f"(num_keep_tokens>=2); this cache has num_keep_tokens={num_keep_tokens}. "
                    "Rebuild the cache with --extraction_source document_boundaries."
                )
            if args.use_yes_logit and not cache_metadata.get("cache_per_document_yes_logits", False):
                raise ValueError(
                    "--use_yes_logit requires a cache built with --cache_per_document_yes_logits "
                    "(this cache's metadata does not have it set); rebuild the cache with that flag."
                )
            model = LateInteractionHeadModel(
                hidden_size=int(cache_metadata["hidden_size"]),
                num_classes=len(action_names),
                action_names=action_names,
                num_keep_tokens=num_keep_tokens,
                n_layers=args.cross_layers,
                n_heads=args.cross_heads,
                dropout=args.cross_dropout,
                use_type_emb=args.use_type_emb,
                ffn_dim=args.cross_ffn_dim,
                pool=args.late_interaction_pool,
                use_yes_logit=args.use_yes_logit,
                yes_logit_mode=args.yes_logit_mode,
            ).to(device)
        else:
            model = LateInterHeadModel(
                hidden_size=int(cache_metadata["hidden_size"]),
                num_classes=len(action_names),
                action_names=action_names,
                n_layers=args.cross_layers,
                n_heads=args.cross_heads,
                dropout=args.cross_dropout,
                use_type_emb=args.use_type_emb,
                ffn_dim=args.cross_ffn_dim,
            ).to(device)
    else:
        print("Loading datasets...")
        splits = load_action_splits(
            dataset_names=dataset_names,
            action_names=action_names,
            threshold=args.threshold,
            max_samples=args.max_samples,
            base_dir=args.results_base_dir,
            label_metric=args.label_metric,
            top_k_docs=args.top_k_docs,
        )

        model_path = pretrained_model_path if pretrained_model_path else args.model_name
        model_source = "fine-tuned scorer" if pretrained_model_path else "base model"

        print(f"Loading {model_source} from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(device)

        if not args.freeze_base and hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

        model = LateEarlyHybridModel(
            base_model=base_model,
            hidden_size=base_model.config.hidden_size,
            num_classes=len(action_names),
            action_names=action_names,
            n_layers=args.cross_layers,
            n_heads=args.cross_heads,
            dropout=args.cross_dropout,
            eos_token_id=base_model.config.eos_token_id,
            encode_batch_size=args.encode_batch_size,
        ).to(device)

        if args.freeze_base:
            for param in model.base.parameters():
                param.requires_grad = False
            print("Backbone frozen; training only interaction head.")

            if hasattr(model.base, "gradient_checkpointing_disable"):
                model.base.gradient_checkpointing_disable()
                print("Disabled gradient checkpointing (base frozen).")

            if hasattr(model.base.config, "use_cache"):
                model.base.config.use_cache = False

        train_dataset = QwenActionDataset(
            splits["train"],
            tokenizer,
            max_length=args.max_length,
            top_k_docs=args.top_k_docs,
            routing_prompt=args.routing_prompt,
        )
        val_dataset = QwenActionDataset(
            splits["val"],
            tokenizer,
            max_length=args.max_length,
            top_k_docs=args.top_k_docs,
            routing_prompt=args.routing_prompt,
        )
        test_dataset = QwenActionDataset(
            splits["test"],
            tokenizer,
            max_length=args.max_length,
            top_k_docs=args.top_k_docs,
            routing_prompt=args.routing_prompt,
        )

    if not train_dataset or not val_dataset or not test_dataset:
        raise ValueError(
            "One or more combined splits are empty. Check that every selected dataset has complete action files."
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_labels = dataset_labels(train_dataset)
    class_weights, train_label_counts = compute_class_weights(
        labels=train_labels,
        num_classes=len(train_action_names),
        mode=args.class_weight_mode,
    )
    if class_weights is not None:
        class_weights = class_weights.to(device)
    if args.loss_mode != "ce" and class_weights is not None:
        print(f"Class weights are ignored for loss_mode={args.loss_mode}.")
        class_weights = None

    num_training_steps = len(train_dataloader) * args.epochs
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    print(f"\n{'=' * 80}")
    print("TRAINING (LATE-INTER ACTION MULTICLASS)")
    print(f"{'=' * 80}")
    print(f"Base model: {model_path}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Datasets: {dataset_names}")
    print(f"Actions: {action_names}")
    print(f"Train-time action pool: {train_action_names}")
    print(f"Holdout dataset: {args.holdout_dataset or '(none)'}")
    print(f"Holdout action: {args.holdout_action or '(none)'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Cross layers: {args.cross_layers}, heads: {args.cross_heads}, ffn_dim: {args.cross_ffn_dim}, dropout: {args.cross_dropout}")
    print(f"Max length: {args.max_length}")
    print(f"Top-k docs per action: {args.top_k_docs}")
    print(f"Encode batch size: {args.encode_batch_size}")
    print(f"Cached embeddings: {bool(args.embedding_cache_dir)}")
    print(f"Freeze base: {args.freeze_base}")
    print(f"Loss mode: {args.loss_mode}")
    print(f"Soft target temperature: {args.soft_target_temperature}")
    print(f"Class weight mode: {args.class_weight_mode}")
    print(f"Train label counts: {train_label_counts}")
    print(f"Class weights: {None if class_weights is None else class_weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"{'=' * 80}\n")

    wandb_run = None
    if wandb is not None and getattr(args, "wandb_project", None):
        action_tag = "_".join(a.replace("_1_step", "") for a in action_names)
        run_name = f"{run_dataset_name}-{action_tag}-seed{args.seed}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            dir=log_dir,
            resume="allow",
            settings=wandb.Settings(init_timeout=300),
        )

    best_val_selected_f1 = -float("inf")
    global_step = 0
    training_history = {
        "dataset": run_dataset_name,
        "datasets": dataset_names,
        "seed": args.seed,
        "actions": action_names,
        "retrievers": action_names,
        "base_model_path": model_path,
        "embedding_cache_dir": args.embedding_cache_dir,
        "loss_mode": args.loss_mode,
        "soft_target_temperature": args.soft_target_temperature,
        "class_weight_mode": args.class_weight_mode,
        "train_label_counts": train_label_counts,
        "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        "epochs": [],
        "train_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "best_val_acc": 0,
        "best_epoch": 0,
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, global_step = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            device,
            loss_mode=args.loss_mode,
            class_weights=class_weights,
            soft_target_temperature=args.soft_target_temperature,
            wandb_run=wandb_run,
            global_step=global_step,
            target_key=args.loss_target_key,
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validation shares the training pool (it may be smaller than the test
        # pool under --holdout_action); test always uses the full pool.
        val_metrics = evaluate_multiclass(model, val_dataloader, device, train_action_names)
        cls_metrics = val_metrics["classification"]
        downstream_metrics = val_metrics["downstream"]
        _wandb_log_eval(wandb_run, "val", val_metrics, epoch + 1, global_step)

        print(
            "Validation - Accuracy: "
            f"{cls_metrics['accuracy']:.4f}, "
            f"Macro Precision: {cls_metrics['precision_macro']:.4f}, "
            f"Macro Recall: {cls_metrics['recall_macro']:.4f}, "
            f"Macro F1: {cls_metrics['f1_macro']:.4f}"
        )
        print(f"Downstream F1 Val | Avg F1: {downstream_metrics['selected_f1']:.4f}")

        training_history["epochs"].append(epoch + 1)
        training_history["train_loss"].append(float(train_loss))
        training_history["val_accuracy"].append(float(cls_metrics["accuracy"]))
        training_history["val_precision"].append(float(cls_metrics["precision_macro"]))
        training_history["val_recall"].append(float(cls_metrics["recall_macro"]))
        training_history["val_f1"].append(float(cls_metrics["f1_macro"]))

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "val_accuracy": float(cls_metrics["accuracy"]),
            "val_precision_macro": float(cls_metrics["precision_macro"]),
            "val_recall_macro": float(cls_metrics["recall_macro"]),
            "val_f1_macro": float(cls_metrics["f1_macro"]),
            "val_auc_ovr_macro": None if cls_metrics["auc_ovr_macro"] is None else float(cls_metrics["auc_ovr_macro"]),
            "val_confusion_matrix": cls_metrics["confusion_matrix"],
            "val_downstream_selected_f1": float(downstream_metrics["selected_f1"]),
            "val_downstream_oracle_f1": float(downstream_metrics["oracle_f1"]),
            "val_downstream_oracle_gap": float(downstream_metrics["oracle_gap"]),
            "val_downstream_selected_label_metric": float(downstream_metrics["selected_label_metric"]),
            "val_downstream_oracle_label_metric": float(downstream_metrics["oracle_label_metric"]),
            "val_downstream_oracle_label_metric_gap": float(downstream_metrics["oracle_label_metric_gap"]),
            "val_downstream_best_baseline_name": downstream_metrics["best_baseline_name"],
            "val_downstream_best_baseline_f1": float(downstream_metrics["best_baseline_f1"]),
            "val_downstream_improvement_vs_best_baseline": float(downstream_metrics["improvement_vs_best_baseline"]),
            "val_downstream_baseline_f1_by_action": downstream_metrics["baseline_f1_by_action"],
            "val_downstream_baseline_f1_by_retriever": downstream_metrics["baseline_f1_by_retriever"],
            "val_downstream_baseline_label_metric_by_action": downstream_metrics["baseline_label_metric_by_action"],
        }

        epoch_metrics_file = os.path.join(metrics_dir, f"epoch_{epoch + 1}_metrics.json")
        with open(epoch_metrics_file, "w", encoding="utf-8") as f:
            json.dump(epoch_metrics, f, indent=2)

        if downstream_metrics["selected_f1"] > best_val_selected_f1:
            best_val_selected_f1 = downstream_metrics["selected_f1"]
            training_history["best_val_acc"] = float(best_val_selected_f1)
            training_history["best_epoch"] = epoch + 1
            print(f"New best validation downstream F1: {best_val_selected_f1:.4f}")
            model.save_pretrained(args.output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(args.output_dir)

    print(f"\n{'=' * 80}")
    print("FINAL TEST EVALUATION")
    print(f"{'=' * 80}")

    best_checkpoint = os.path.join(args.output_dir, "hybrid_model.bin")
    if os.path.exists(best_checkpoint):
        print(f"Loading best validation checkpoint from {best_checkpoint}")
        model.load_state_dict(torch_load(best_checkpoint, map_location=device))

    test_metrics = evaluate_multiclass(model, test_dataloader, device, action_names)
    cls_metrics = test_metrics["classification"]
    downstream_metrics = test_metrics["downstream"]

    print(
        "Test - Accuracy: "
        f"{cls_metrics['accuracy']:.4f}, "
        f"Macro Precision: {cls_metrics['precision_macro']:.4f}, "
        f"Macro Recall: {cls_metrics['recall_macro']:.4f}, "
        f"Macro F1: {cls_metrics['f1_macro']:.4f}"
    )
    print(f"\nDownstream F1 Test | Avg F1: {downstream_metrics['selected_f1']:.4f}")
    print(f"Upper-bound Downstream F1 Test | Avg F1: {downstream_metrics['oracle_f1']:.4f}")
    print(f"Best fixed action baseline: {downstream_metrics['best_baseline_name']} ({downstream_metrics['best_baseline_f1']:.4f})")

    test_metrics_dict = {
        "test_accuracy": float(cls_metrics["accuracy"]),
        "test_precision_macro": float(cls_metrics["precision_macro"]),
        "test_recall_macro": float(cls_metrics["recall_macro"]),
        "test_f1_macro": float(cls_metrics["f1_macro"]),
        "test_auc_ovr_macro": None if cls_metrics["auc_ovr_macro"] is None else float(cls_metrics["auc_ovr_macro"]),
        "test_confusion_matrix": cls_metrics["confusion_matrix"],
        "test_downstream_selected_f1": float(downstream_metrics["selected_f1"]),
        "test_downstream_oracle_f1": float(downstream_metrics["oracle_f1"]),
        "test_downstream_oracle_gap": float(downstream_metrics["oracle_gap"]),
        "test_downstream_selected_label_metric": float(downstream_metrics["selected_label_metric"]),
        "test_downstream_oracle_label_metric": float(downstream_metrics["oracle_label_metric"]),
        "test_downstream_oracle_label_metric_gap": float(downstream_metrics["oracle_label_metric_gap"]),
        "test_downstream_best_baseline_name": downstream_metrics["best_baseline_name"],
        "test_downstream_best_baseline_f1": float(downstream_metrics["best_baseline_f1"]),
        "test_downstream_improvement_vs_best_baseline": float(downstream_metrics["improvement_vs_best_baseline"]),
        "test_downstream_baseline_f1_by_action": downstream_metrics["baseline_f1_by_action"],
        "test_downstream_baseline_f1_by_retriever": downstream_metrics["baseline_f1_by_retriever"],
        "test_downstream_baseline_label_metric_by_action": downstream_metrics["baseline_label_metric_by_action"],
        # Under leave-one-retriever-out these two are the point of the run: mean
        # F1 cannot show whether a never-trained-on retriever is reachable at all.
        "test_downstream_pick_rate_by_action": downstream_metrics["pick_rate_by_action"],
        "test_downstream_oracle_recall_by_action": downstream_metrics["oracle_recall_by_action"],
    }

    _wandb_log_eval(wandb_run, "test", test_metrics, training_history["best_epoch"], global_step)

    test_metrics_file = os.path.join(metrics_dir, "test_metrics.json")
    with open(test_metrics_file, "w", encoding="utf-8") as f:
        json.dump(test_metrics_dict, f, indent=2)

    training_history.update(test_metrics_dict)

    history_file = os.path.join(metrics_dir, "training_history.json")
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2)

    plot_training_curves(training_history, plots_dir)

    summary = {
        "dataset": run_dataset_name,
        "datasets": dataset_names,
        "seed": args.seed,
        "actions": action_names,
        "retrievers": action_names,
        "best_val_accuracy": float(best_val_selected_f1),
        "best_epoch": training_history["best_epoch"],
        "test_accuracy": float(cls_metrics["accuracy"]),
        "test_precision_macro": float(cls_metrics["precision_macro"]),
        "test_recall_macro": float(cls_metrics["recall_macro"]),
        "test_f1_macro": float(cls_metrics["f1_macro"]),
        "test_auc_ovr_macro": None if cls_metrics["auc_ovr_macro"] is None else float(cls_metrics["auc_ovr_macro"]),
        "test_downstream_selected_f1": float(downstream_metrics["selected_f1"]),
        "test_downstream_oracle_f1": float(downstream_metrics["oracle_f1"]),
        "test_downstream_best_baseline_name": downstream_metrics["best_baseline_name"],
        "test_downstream_best_baseline_f1": float(downstream_metrics["best_baseline_f1"]),
        "test_downstream_improvement_vs_best_baseline": float(downstream_metrics["improvement_vs_best_baseline"]),
        "base_model_path": model_path,
        "embedding_cache_dir": args.embedding_cache_dir,
        "loss_mode": args.loss_mode,
        "soft_target_temperature": args.soft_target_temperature,
        "class_weight_mode": args.class_weight_mode,
        "train_label_counts": train_label_counts,
        "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
        "model_path": args.output_dir,
    }

    summary_file = os.path.join(metrics_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Datasets: {dataset_names}")
    print(f"Seed: {args.seed}")
    print(f"Actions: {action_names}")
    print(f"Best validation downstream F1: {best_val_selected_f1:.4f} (epoch {training_history['best_epoch']})")
    print(f"Test accuracy: {cls_metrics['accuracy']:.4f}")
    print(f"Model saved to: {args.output_dir}")
    print(f"Metrics saved to: {metrics_dir}")
    print(f"Logs saved to: {log_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"{'=' * 80}")

    if wandb_run is not None:
        wandb_run.finish()

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_f.close()


if __name__ == "__main__":
    main()
