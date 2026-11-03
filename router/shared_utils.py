import glob
import json
import os
import random
import sys
from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Dataset

from src.data import load_corpus
from src.metric import get_f1, get_ndcg


DEFAULT_RESULTS_BASE_DIR = "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
DEFAULT_RUNS_DIR = "outputs/runs"
DEFAULT_CHECKPOINTS_DIR = "outputs/checkpoints"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
RERANKER_PREFIX = (
    "<|im_start|>system\n"
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

# Routing-focused prompt variants.
# ROUTING_PREFIX / ROUTING_INSTRUCTION frame the task as retriever selection rather than
# document relevance, encouraging the backbone to encode routing-discriminative signal.
# ROUTING_SUFFIX ends the prompt at the start of the assistant turn, where the
# routing representation is read.
ROUTING_INSTRUCTION = (
    "Given a web search query and documents retrieved by a specific retrieval method, "
    "assess whether this retrieval strategy is well-suited to the query type "
    "and whether the retrieved documents contain enough information to answer the query"
)
ROUTING_PREFIX = (
    "<|im_start|>system\n"
    "You are an expert at evaluating retrieval strategies for question answering. "
    "Given a query and documents retrieved by a specific retrieval method, "
    "reason about how well this retrieval approach suits the query type "
    'and whether the documents are sufficient to answer it. '
    'Answer "yes" if this retrieval strategy is effective for this query, "no" otherwise.<|im_end|>\n'
    "<|im_start|>user\n"
)
ROUTING_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n"


def sanitize_name(name):
    return name.replace("/", "_").replace(",", "__").replace("-", "_").replace(".", "_")


def retriever_tag(retriever_names):
    return "__".join(sanitize_name(name) for name in retriever_names)


def default_run_suffix(threshold, label_metric="ndcg"):
    """Run-directory suffix describing the label metric and the utility-gap filter."""
    threshold_value = float(threshold)
    if threshold_value <= 0:
        threshold_suffix = "no_filtering"
    else:
        threshold_tag = f"{threshold_value:g}".replace("-", "neg_").replace(".", "_")
        threshold_suffix = f"threshold_{threshold_tag}"

    metric_suffix = sanitize_name(label_metric)
    return f"{metric_suffix}-{threshold_suffix}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_specs(dataset_name):
    return [
        ("train", f"{dataset_name}_train", f"{dataset_name}_train"),
        ("val", f"{dataset_name}_val", f"{dataset_name}_val"),
        ("test", dataset_name, dataset_name),
    ]


def parse_retrievers(retrievers):
    if retrievers is None:
        return None
    if isinstance(retrievers, (list, tuple)):
        parsed = [str(name).strip() for name in retrievers if str(name).strip()]
    else:
        raw = str(retrievers).strip()
        if not raw or raw.lower() == "auto":
            return None
        parsed = [name.strip() for name in raw.split(",") if name.strip()]
    if not parsed:
        return None
    return parsed


def discover_retrievers(base_dir, dataset_name, retrievers=None):
    explicit_retrievers = parse_retrievers(retrievers)
    if explicit_retrievers is not None:
        return explicit_retrievers

    ordered_train_names = None
    retriever_sets = []

    for _, split_dir, _ in split_specs(dataset_name):
        pattern = os.path.join(base_dir, split_dir, "*.jsonl")
        split_names = [os.path.splitext(os.path.basename(path))[0] for path in sorted(glob.glob(pattern))]
        if ordered_train_names is None:
            ordered_train_names = split_names
        retriever_sets.append(set(split_names))

    if not retriever_sets or not retriever_sets[0]:
        raise FileNotFoundError(
            f"No retrieval result files found under {base_dir}/{dataset_name}_train."
        )

    common_retrievers = set.intersection(*retriever_sets)
    if not common_retrievers:
        raise FileNotFoundError(
            f"No retriever .jsonl files are present in every split for dataset {dataset_name}."
        )

    inactive_default_retrievers = {
        "rag_2_step",
        "bge_m3_1_step",
        "reason_moderncolbert_1_step",
    }
    filtered_retrievers = [
        name for name in ordered_train_names
        if name in common_retrievers and name not in inactive_default_retrievers
    ]
    if not filtered_retrievers:
        raise FileNotFoundError(
            f"Only inactive default retrievers were found for dataset {dataset_name}: "
            f"{sorted(common_retrievers)}. Generate active retriever outputs first or "
            "pass an explicit --retrievers list."
        )
    common_retrievers = set(filtered_retrievers)

    preferred_order = [
        "rag_1_step",
        "hipporag2-0.5_1_step",
        "bm25_1_step",
        "splade_v3_1_step",
        "colbertv2_1_step",
    ]
    ordered = [name for name in preferred_order if name in common_retrievers]
    ordered.extend(name for name in ordered_train_names if name in common_retrievers and name not in ordered)
    return ordered


def load_results_file(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def item_id(item, fallback_idx):
    return item.get("id", f"row_{fallback_idx}")


def rank_candidates(metric_values, latency_values):
    return sorted(
        range(len(metric_values)),
        key=lambda idx: (-metric_values[idx], latency_values[idx], idx),
    )


def _metric_value(item, metric_name, gold_doc_list, doc_list):
    if metric_name in item:
        return float(item[metric_name])
    if metric_name == "ndcg":
        return float(get_ndcg(gold_doc_list, doc_list, k=5))
    if metric_name == "f1":
        return float(get_f1(gold_doc_list, doc_list))
    raise ValueError(f"Unsupported label metric: {metric_name}")


def _docs_for_item(item, corpus, top_k):
    top_k_indices = item.get("top_k_indices", [])[:top_k]
    if any(idx < 0 or idx >= len(corpus) for idx in top_k_indices):
        return None
    return [corpus[idx] for idx in top_k_indices]


def load_split_selection_data(
    base_dir,
    split_dir,
    corpus_name,
    retrievers,
    threshold=0.05,
    max_samples=None,
    label_metric="ndcg",
    top_k=5,
):
    corpus = load_corpus(corpus_name)
    retriever_results = {}
    retriever_rows = {}
    ordered_item_ids = None

    for retriever_name in retrievers:
        result_path = os.path.join(base_dir, split_dir, f"{retriever_name}.jsonl")
        print(f"Loading retrieval results from {result_path}...")
        results = load_results_file(result_path)
        result_map = {item_id(item, idx): item for idx, item in enumerate(results)}
        retriever_results[retriever_name] = result_map
        retriever_rows[retriever_name] = results
        if ordered_item_ids is None:
            ordered_item_ids = [item_id(item, idx) for idx, item in enumerate(results)]

    queries = []
    docs_per_retriever = []
    labels = []
    f1_scores = []
    metric_scores = []
    latency_scores = []
    sample_ids = []
    skip_counts = {
        "missing_in_retriever": 0,
        "mismatched_question": 0,
        "invalid_doc_indices": 0,
        "below_threshold": 0,
        "row_order_fallback": 0,
    }
    missing_by_retriever = {name: 0 for name in retrievers}
    invalid_by_retriever = {name: 0 for name in retrievers}

    for row_idx, current_item_id in enumerate(ordered_item_ids):
        if max_samples is not None and len(queries) >= max_samples:
            break

        first_retriever = retrievers[0]
        first_item = retriever_results[first_retriever].get(current_item_id)
        if first_item is None:
            if row_idx >= len(retriever_rows[first_retriever]):
                skip_counts["missing_in_retriever"] += 1
                missing_by_retriever[first_retriever] += 1
                continue
            first_item = retriever_rows[first_retriever][row_idx]
        question = first_item["question"]

        items = [first_item]
        used_row_fallback = False
        missing_sample = False

        for retriever_name in retrievers[1:]:
            item = retriever_results[retriever_name].get(current_item_id)
            if item is None:
                if row_idx < len(retriever_rows[retriever_name]):
                    row_item = retriever_rows[retriever_name][row_idx]
                    if row_item.get("question") == question:
                        item = row_item
                        used_row_fallback = True
                if item is None:
                    skip_counts["missing_in_retriever"] += 1
                    missing_by_retriever[retriever_name] += 1
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

        sample_docs = []
        sample_f1_scores = []
        sample_metric_scores = []
        sample_latency_scores = []
        valid_sample = True

        for retriever_name, item in zip(retrievers, items):
            doc_list = _docs_for_item(item, corpus, top_k)
            if doc_list is None:
                invalid_by_retriever[retriever_name] += 1
                valid_sample = False
                break

            gold_doc_list = item.get("gold_doc_list")
            if gold_doc_list is None and (label_metric not in item or "f1" not in item):
                raise ValueError(f"Missing gold_doc_list needed to compute metrics for {current_item_id}")

            sample_docs.append("\n\n".join(doc_list))
            sample_metric_scores.append(_metric_value(item, label_metric, gold_doc_list, doc_list))
            sample_f1_scores.append(_metric_value(item, "f1", gold_doc_list, doc_list))
            sample_latency_scores.append(float(item.get("latency", 0.0)))

        if not valid_sample:
            skip_counts["invalid_doc_indices"] += 1
            continue

        ranking = rank_candidates(sample_metric_scores, sample_latency_scores)
        if len(ranking) > 1 and threshold is not None and threshold > 0:
            metric_gap = sample_metric_scores[ranking[0]] - sample_metric_scores[ranking[1]]
            if metric_gap < threshold:
                skip_counts["below_threshold"] += 1
                continue

        queries.append(question)
        docs_per_retriever.append(sample_docs)
        labels.append(ranking[0])
        f1_scores.append(sample_f1_scores)
        metric_scores.append(sample_metric_scores)
        latency_scores.append(sample_latency_scores)
        sample_ids.append(current_item_id)

    print(f"Loaded {len(queries)} selection examples from {split_dir}")
    if labels:
        print(f"Label distribution: {np.bincount(labels, minlength=len(retrievers)).tolist()}")
    else:
        print(f"No selection examples loaded from {split_dir}.")
    print(f"Skip counts for {split_dir}: {skip_counts}")
    if any(missing_by_retriever.values()):
        print(f"Missing IDs by retriever for {split_dir}: {missing_by_retriever}")
    if any(invalid_by_retriever.values()):
        print(f"Invalid doc-index samples by retriever for {split_dir}: {invalid_by_retriever}")

    return {
        "ids": sample_ids,
        "queries": queries,
        "docs_per_retriever": docs_per_retriever,
        "labels": labels,
        "f1_scores": f1_scores,
        "metric_scores": metric_scores,
        "latency_scores": latency_scores,
    }


def load_selection_splits(
    dataset_name,
    retrievers=None,
    threshold=0.05,
    max_samples=None,
    base_dir=DEFAULT_RESULTS_BASE_DIR,
    label_metric="ndcg",
    top_k=5,
):
    retriever_names = discover_retrievers(base_dir, dataset_name, retrievers)
    splits = {}

    print(f"\n{'=' * 80}")
    print(f"Loading {dataset_name} dataset with retrievers: {retriever_names}")
    print(f"{'=' * 80}")

    for split_name, split_dir, corpus_name in split_specs(dataset_name):
        print(f"\nLoading {split_name} split...")
        splits[split_name] = load_split_selection_data(
            base_dir=base_dir,
            split_dir=split_dir,
            corpus_name=corpus_name,
            retrievers=retriever_names,
            threshold=threshold,
            max_samples=max_samples,
            label_metric=label_metric,
            top_k=top_k,
        )
        print(f"{split_name.upper()}: {len(splits[split_name]['queries'])} examples")

    return splits, retriever_names


class QwenEncodingMixin:
    def _init_qwen_encoding(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix_tokens = self.tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)

    def format_instruction(self, query, doc):
        return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
            instruction=DEFAULT_INSTRUCTION,
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

        input_ids = self.prefix_tokens + encoded["input_ids"] + self.suffix_tokens
        attention_mask = [1] * len(input_ids)

        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        else:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]

        return input_ids, attention_mask


class QwenRetrieverSelectionDataset(QwenEncodingMixin, Dataset):
    def __init__(self, split_data, tokenizer, max_length=2048):
        self.queries = split_data["queries"]
        self.docs_per_retriever = split_data["docs_per_retriever"]
        self.labels = split_data["labels"]
        self.retriever_f1_scores = split_data["f1_scores"]
        self.metric_scores = split_data["metric_scores"]
        self._init_qwen_encoding(tokenizer, max_length)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        encoded_inputs = [self.encode_text(query, doc) for doc in self.docs_per_retriever[idx]]
        input_ids = [item[0] for item in encoded_inputs]
        attention_masks = [item[1] for item in encoded_inputs]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "retriever_f1s": torch.tensor(self.retriever_f1_scores[idx], dtype=torch.float),
            "retriever_metrics": torch.tensor(self.metric_scores[idx], dtype=torch.float),
        }


class QwenPointwiseSelectionDataset(QwenEncodingMixin, Dataset):
    def __init__(self, split_data, tokenizer, max_length=2048):
        self.examples = []
        self._init_qwen_encoding(tokenizer, max_length)

        for query, docs, label in zip(
            split_data["queries"],
            split_data["docs_per_retriever"],
            split_data["labels"],
        ):
            for retriever_idx, doc in enumerate(docs):
                self.examples.append((query, doc, 1 if retriever_idx == label else 0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, doc, label = self.examples[idx]
        input_ids, attention_mask = self.encode_text(query, doc)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class QwenPairwiseSelectionDataset(QwenEncodingMixin, Dataset):
    def __init__(self, split_data, tokenizer, max_length=2048, pair_threshold=0.05):
        self.examples = []
        self._init_qwen_encoding(tokenizer, max_length)

        for sample_idx, (query, docs, metric_scores, latency_scores) in enumerate(
            zip(
                split_data["queries"],
                split_data["docs_per_retriever"],
                split_data["metric_scores"],
                split_data["latency_scores"],
            )
        ):
            for idx1, idx2 in combinations(range(len(docs)), 2):
                metric_diff = abs(metric_scores[idx2] - metric_scores[idx1])
                if pair_threshold is not None and pair_threshold > 0 and metric_diff < pair_threshold:
                    continue

                ordered_pair_ranking = rank_candidates(
                    [metric_scores[idx1], metric_scores[idx2]],
                    [latency_scores[idx1], latency_scores[idx2]],
                )
                better_pair_position = ordered_pair_ranking[0]
                label = 1 if better_pair_position == 1 else 0
                self.examples.append((query, docs[idx1], docs[idx2], label, sample_idx, idx1, idx2))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, doc1, doc2, label, sample_idx, idx1, idx2 = self.examples[idx]
        input_ids1, attention_mask1 = self.encode_text(query, doc1)
        input_ids2, attention_mask2 = self.encode_text(query, doc2)

        return {
            "input_ids1": torch.tensor(input_ids1, dtype=torch.long),
            "attention_mask1": torch.tensor(attention_mask1, dtype=torch.long),
            "input_ids2": torch.tensor(input_ids2, dtype=torch.long),
            "attention_mask2": torch.tensor(attention_mask2, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "sample_idx": torch.tensor(sample_idx, dtype=torch.long),
            "retriever_idx1": torch.tensor(idx1, dtype=torch.long),
            "retriever_idx2": torch.tensor(idx2, dtype=torch.long),
        }


def _score_qwen_candidates(model, input_ids, attention_mask, token_true_id, score_batch_size):
    batch_size, num_retrievers, seq_len = input_ids.shape
    flat_input_ids = input_ids.reshape(batch_size * num_retrievers, seq_len)
    flat_attention_mask = attention_mask.reshape(batch_size * num_retrievers, seq_len)
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
        score_chunks.append(final_token_logits[:, token_true_id].float().detach())
        del outputs, hidden_states, last_hidden_states, final_token_logits

    return torch.cat(score_chunks, dim=0).view(batch_size, num_retrievers)


def evaluate_qwen_selection(model, dataloader, device, retriever_names, token_true_id, score_batch_size=16):
    model.eval()

    all_predictions = []
    all_scores = []
    all_labels = []
    all_retriever_f1s = []
    all_retriever_metrics = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = _score_qwen_candidates(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_true_id=token_true_id,
                score_batch_size=score_batch_size,
            )
            predictions = torch.argmax(scores, dim=-1)

            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
            all_retriever_f1s.extend(batch["retriever_f1s"].cpu().numpy().tolist())
            all_retriever_metrics.extend(batch["retriever_metrics"].cpu().numpy().tolist())

    predictions = np.array(all_predictions, dtype=int)
    scores = np.array(all_scores, dtype=float)
    labels = np.array(all_labels, dtype=int)
    retriever_f1s = np.array(all_retriever_f1s, dtype=float)
    retriever_metrics = np.array(all_retriever_metrics, dtype=float)

    if len(labels) == 0:
        raise ValueError("Cannot evaluate an empty selection dataset.")

    scores_for_softmax = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores_for_softmax)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    classification_metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=list(range(len(retriever_names)))).tolist(),
    }

    try:
        if len(np.unique(labels)) > 1:
            classification_metrics["auc_ovr_macro"] = float(
                roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            )
        else:
            classification_metrics["auc_ovr_macro"] = None
    except ValueError:
        classification_metrics["auc_ovr_macro"] = None

    row_indices = np.arange(len(predictions))
    selected_f1 = retriever_f1s[row_indices, predictions]
    oracle_f1 = np.max(retriever_f1s, axis=1)
    selected_metric = retriever_metrics[row_indices, predictions]
    oracle_metric = np.max(retriever_metrics, axis=1)

    baseline_f1_by_retriever = {
        retriever_name: float(retriever_f1s[:, idx].mean())
        for idx, retriever_name in enumerate(retriever_names)
    }
    baseline_metric_by_retriever = {
        retriever_name: float(retriever_metrics[:, idx].mean())
        for idx, retriever_name in enumerate(retriever_names)
    }
    best_baseline_name = max(baseline_f1_by_retriever, key=baseline_f1_by_retriever.get)

    downstream_metrics = {
        "selected_f1": float(selected_f1.mean()),
        "oracle_f1": float(oracle_f1.mean()),
        "oracle_gap": float(oracle_f1.mean() - selected_f1.mean()),
        "selected_label_metric": float(selected_metric.mean()),
        "oracle_label_metric": float(oracle_metric.mean()),
        "label_metric_oracle_gap": float(oracle_metric.mean() - selected_metric.mean()),
        "baseline_f1_by_retriever": baseline_f1_by_retriever,
        "baseline_label_metric_by_retriever": baseline_metric_by_retriever,
        "best_baseline_name": best_baseline_name,
        "best_baseline_f1": float(baseline_f1_by_retriever[best_baseline_name]),
        "improvement_vs_best_baseline": float(selected_f1.mean() - baseline_f1_by_retriever[best_baseline_name]),
    }

    return {
        "classification": classification_metrics,
        "downstream": downstream_metrics,
    }


def resolve_run_dirs(
    method_dir_name,
    dataset_name,
    seed,
    retriever_names,
    run_name_suffix=None,
    runs_dir=None,
    checkpoints_dir=None,
    output_dir=None,
):
    """
    Where a run writes. Logs, metrics and plots go under
    <runs_dir>/<method>/<datasets>/<retrievers>/<run_name>/; the model goes to
    output_dir if given, else to the same sub-path under checkpoints_dir.
    """
    runs_dir = runs_dir or DEFAULT_RUNS_DIR
    checkpoints_dir = checkpoints_dir or DEFAULT_CHECKPOINTS_DIR
    tag = retriever_tag(retriever_names)
    run_name = f"seed_{seed}"
    if run_name_suffix:
        run_name = f"{run_name}-{run_name_suffix}"

    run_dir = os.path.abspath(
        os.path.join(runs_dir, method_dir_name, dataset_name, tag, run_name)
    )
    if output_dir:
        model_dir = os.path.abspath(output_dir)
    else:
        model_dir = os.path.abspath(
            os.path.join(checkpoints_dir, method_dir_name, dataset_name, tag, run_name, "checkpoints")
        )

    return {
        "run_dir": run_dir,
        "model_dir": model_dir,
        "log_dir": os.path.join(run_dir, "logs"),
        "metrics_dir": os.path.join(run_dir, "metrics"),
        "plots_dir": os.path.join(run_dir, "plots"),
        "retriever_tag": tag,
        "run_name": run_name,
        "run_name_suffix": run_name_suffix,
    }


class TeeStream:
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

    def isatty(self):
        return self.stream.isatty()


def install_file_logger(log_file):
    log_f = open(log_file, "w", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(log_f, sys.stdout)
    sys.stderr = TeeStream(log_f, sys.stderr)

    def restore():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_f.close()

    return restore


def plot_training_curves(training_history, plots_dir):
    epochs = training_history.get("epochs", [])
    if not epochs:
        return

    train_loss = training_history["train_loss"]
    val_loss = training_history.get("val_loss", [])
    val_accuracy = training_history["val_accuracy"]
    val_f1 = training_history["val_f1"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, train_loss, marker="o", linewidth=2, label="Train")
    if len(val_loss) == len(epochs):
        axes[0, 0].plot(epochs, val_loss, marker="s", linewidth=2, label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_ylabel("Cross-entropy")

    axes[0, 1].plot(epochs, val_accuracy, marker="s", linewidth=2, label="Accuracy")
    axes[0, 1].plot(epochs, val_f1, marker="^", linewidth=2, label="Macro F1")
    axes[0, 1].set_title("Validation Classification")
    axes[0, 1].set_ylim(0, 1)

    selected_f1 = training_history.get("val_selected_f1", [])
    oracle_f1 = training_history.get("val_oracle_f1", [])
    baseline_f1 = training_history.get("val_best_baseline_f1", [])
    if len(selected_f1) == len(epochs):
        axes[0, 2].plot(epochs, selected_f1, marker="o", linewidth=2, label="Selected")
        axes[0, 2].plot(epochs, oracle_f1, linestyle="--", linewidth=2, label="Oracle")
        axes[0, 2].plot(epochs, baseline_f1, linestyle=":", linewidth=2, label="Best fixed")
        best_epoch = training_history.get("best_epoch", 0)
        if best_epoch in epochs:
            best_idx = epochs.index(best_epoch)
            axes[0, 2].plot(
                best_epoch,
                selected_f1[best_idx],
                "r*",
                markersize=15,
                label=f"Best checkpoint ({best_epoch})",
            )
    axes[0, 2].set_title("Validation Downstream QA F1")

    oracle_gap = training_history.get("val_oracle_gap", [])
    improvement = training_history.get("val_improvement_vs_best_baseline", [])
    if len(oracle_gap) == len(epochs):
        axes[1, 0].plot(epochs, oracle_gap, marker="o", linewidth=2, label="Oracle regret")
        axes[1, 0].plot(epochs, improvement, marker="s", linewidth=2, label="Lift vs best fixed")
        axes[1, 0].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1, 0].set_title("Routing Value")

    confidence = training_history.get("val_mean_confidence", [])
    margin = training_history.get("val_score_margin", [])
    calibration = training_history.get("val_calibration_error", [])
    if len(confidence) == len(epochs):
        axes[1, 1].plot(epochs, confidence, marker="o", linewidth=2, label="Mean confidence")
        axes[1, 1].plot(epochs, margin, marker="s", linewidth=2, label="Top-1/top-2 margin")
        axes[1, 1].plot(epochs, calibration, marker="^", linewidth=2, label="ECE")
    axes[1, 1].set_title("Validation Confidence")

    truncation = training_history.get("val_truncation_fraction", [])
    if len(truncation) == len(epochs):
        axes[1, 2].plot(
            epochs,
            np.asarray(truncation) * 100,
            marker="o",
            linewidth=2,
            label="Truncated",
        )
    axes[1, 2].set_title("Validation Input Truncation")
    axes[1, 2].set_ylabel("Candidates truncated (%)")

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([min(epochs) - 0.5, max(epochs) + 0.5])
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plot_file = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {plot_file}")

    step_metrics_path = training_history.get("step_metrics_file")
    if step_metrics_path and os.path.exists(step_metrics_path):
        with open(step_metrics_path, "r", encoding="utf-8") as f:
            steps = [json.loads(line) for line in f if line.strip()]
        if steps:
            global_steps = [row["global_step"] for row in steps]
            fig, axes = plt.subplots(2, 2, figsize=(15, 9))
            axes[0, 0].plot(global_steps, [row["loss"] for row in steps], alpha=0.25, label="Raw")
            axes[0, 0].plot(global_steps, [row["loss_ema"] for row in steps], linewidth=2, label="EMA")
            axes[0, 0].set_title("Step Training Loss")
            axes[0, 1].plot(global_steps, [row["gradient_norm"] for row in steps], linewidth=1)
            axes[0, 1].set_title("Gradient L2 Norm (Pre-clipping)")
            axes[1, 0].plot(global_steps, [row["learning_rate"] for row in steps], linewidth=2)
            axes[1, 0].set_title("Learning Rate")
            axes[1, 1].plot(
                global_steps,
                [row["tokens_per_second"] for row in steps],
                alpha=0.6,
                label="Tokens/s",
            )
            memory_axis = axes[1, 1].twinx()
            memory_axis.plot(
                global_steps,
                [row["gpu_peak_memory_gib"] for row in steps],
                color="tab:red",
                alpha=0.7,
                label="Peak memory",
            )
            axes[1, 1].set_title("Throughput and GPU Memory")
            axes[1, 1].set_ylabel("Tokens/s")
            memory_axis.set_ylabel("Peak memory (GiB)", color="tab:red")
            for ax in axes.flat:
                ax.set_xlabel("Optimizer step")
                ax.grid(True, alpha=0.3)
                if ax.get_legend_handles_labels()[0]:
                    ax.legend(fontsize=9)
            plt.tight_layout()
            optimization_file = os.path.join(plots_dir, "optimization_steps.png")
            plt.savefig(optimization_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Optimization diagnostics saved to: {optimization_file}")

    test_classification = training_history.get("test_classification")
    actions = training_history.get("actions", [])
    if test_classification and actions:
        matrix = np.asarray(test_classification.get("confusion_matrix", []), dtype=int)
        if matrix.shape == (len(actions), len(actions)):
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            image = axes[0].imshow(matrix, cmap="Blues")
            axes[0].set_title("Test Confusion Matrix")
            axes[0].set_xlabel("Predicted action")
            axes[0].set_ylabel("Gold action")
            axes[0].set_xticks(range(len(actions)))
            axes[0].set_xticklabels(actions, rotation=45, ha="right")
            axes[0].set_yticks(range(len(actions)))
            axes[0].set_yticklabels(actions)
            for row_idx in range(len(actions)):
                for col_idx in range(len(actions)):
                    axes[0].text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center")
            fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)

            label_counts = test_classification.get("label_counts", {})
            prediction_counts = test_classification.get("prediction_counts", {})
            positions = np.arange(len(actions))
            width = 0.38
            axes[1].bar(positions - width / 2, [label_counts.get(name, 0) for name in actions], width, label="Gold")
            axes[1].bar(positions + width / 2, [prediction_counts.get(name, 0) for name in actions], width, label="Predicted")
            axes[1].set_title("Test Action Distribution")
            axes[1].set_xticks(positions)
            axes[1].set_xticklabels(actions, rotation=45, ha="right")
            axes[1].set_ylabel("Samples")
            axes[1].legend()
            axes[1].grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            selection_file = os.path.join(plots_dir, "test_selection_diagnostics.png")
            plt.savefig(selection_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Test selection diagnostics saved to: {selection_file}")
