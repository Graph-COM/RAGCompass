#!/usr/bin/env python3
"""
MoR-style baseline (Kalra et al. 2025, "MoR: Mixture of Sparse, Dense, and
Human Retrievers"): a zero-shot *weighted* combination of the 5 retrievers'
existing rankings, as opposed to RRF's rank-only fusion. Per-retriever weights
are static (each retriever's own mean F1 on its _val split, normalized to sum
to 1) -- no per-query training, matching MoR's "zero-shot" framing and keeping
this apples-to-apples with the weight-free RRF baseline.

Heavily reuses rrf.py's structure (I/O, resumability,
downstream generation backend); the only conceptual change is the fusion rule:
  RRF:  score(doc) = sum_r  1 / (rrf_k + rank_r(doc))
  MoR:  score(doc) = sum_r  weight_r * normalized_score_r(doc)
where normalized_score_r is that retriever's own top-k scores min-max scaled
to [0, 1] per query (raw scores aren't comparable across retriever paradigms
-- BM25 ~6-8, SPLADE ~20-30, cosine similarity 0-1, ...).
"""

import argparse
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from src.artifacts import ARTIFACT_ROOT_ENV, apply_cache_environment


DEFAULT_RESULTS_BASE_DIR = "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
DEFAULT_DATASETS = "nq,musique,hotpotqa,2wikimultihopqa"
DATASET_GROUPS = {
    "nq": ["nq_train", "nq_val", "nq"],
    "musique": ["musique_train", "musique_val", "musique"],
    "hotpotqa": ["hotpotqa_train", "hotpotqa_val", "hotpotqa"],
    "2wikimultihopqa": ["2wikimultihopqa_train", "2wikimultihopqa_val", "2wikimultihopqa"],
}
DEFAULT_MOR_SOURCES = [
    "rag_1_step",
    "hipporag2-0.5_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
]
MOR_OUTPUT_NAME = "mor_1_step"


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_datasets(dataset_arg, expand_groups=True):
    requested = [item.strip() for item in dataset_arg.split(",") if item.strip()]
    resolved = []
    seen = set()

    for item in requested:
        expanded = DATASET_GROUPS.get(item, [item]) if expand_groups else [item]
        for dataset in expanded:
            if dataset not in seen:
                resolved.append(dataset)
                seen.add(dataset)

    return resolved


def parse_retrievers(raw_value):
    if raw_value is None:
        return None

    normalized = raw_value.strip()
    if not normalized or normalized.lower() == "auto":
        return None

    retrievers = [item.strip() for item in normalized.split(",") if item.strip()]
    if not retrievers:
        return None
    return retrievers


def output_path(results_base_dir, dataset, output_name):
    return Path(results_base_dir) / dataset / f"{output_name}.jsonl"


def load_done_ids(path):
    ids = set()
    if not path.exists():
        log(f"Cached output not found, starting fresh: {path}")
        return ids

    start_time = time.time()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item.get("id")
            if item_id is None:
                raise ValueError(f"Missing id in cached output {path}:{line_no}")
            if item_id in ids:
                raise ValueError(f"Duplicate id {item_id!r} in cached output {path}")
            ids.add(item_id)

    log(f"Loaded {len(ids)} cached ids from {path} in {time.time() - start_time:.2f}s")
    return ids


def load_results_file(path):
    rows = []
    result_map = {}
    start_time = time.time()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            item_id = item.get("id")
            if item_id is None:
                raise ValueError(f"Missing id in retrieval result {path}:{line_no}")
            if item_id in result_map:
                raise ValueError(f"Duplicate id {item_id!r} in retrieval result {path}")
            rows.append(item)
            result_map[item_id] = item
    log(f"Loaded {len(rows)} retrieval rows from {path} in {time.time() - start_time:.2f}s")
    return rows, result_map


def resolve_mor_sources(results_base_dir, dataset, retriever_arg):
    explicit_retrievers = parse_retrievers(retriever_arg)
    dataset_dir = Path(results_base_dir) / dataset

    if explicit_retrievers is not None:
        missing = [
            name for name in explicit_retrievers
            if not (dataset_dir / f"{name}.jsonl").exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing explicit MoR source files for {dataset}: {missing}"
            )
        return explicit_retrievers

    sources = [
        name for name in DEFAULT_MOR_SOURCES
        if (dataset_dir / f"{name}.jsonl").exists()
    ]
    if not sources:
        raise FileNotFoundError(
            f"No active one-step MoR source files found under {dataset_dir}"
        )
    return sources


def load_source_maps(results_base_dir, dataset, source_names):
    source_maps = {}
    source_rows = {}
    source_counts = {}
    log(f"Dataset={dataset}: loading {len(source_names)} MoR source file(s)")
    for source_name in source_names:
        source_path = Path(results_base_dir) / dataset / f"{source_name}.jsonl"
        rows, result_map = load_results_file(source_path)
        source_maps[source_name] = result_map
        source_rows[source_name] = rows
        source_counts[source_name] = len(rows)
    return source_maps, source_rows, source_counts


def compute_mor_weights(results_base_dir, dataset, source_names):
    """Zero-shot static weights: each retriever's own mean F1 on its _val
    split, normalized to sum to 1. No per-query training and no new LLM
    calls -- reuses F1s already cached in the existing per-retriever _val
    jsonl files, matching MoR's 'zero-shot' framing."""
    val_dir = f"{dataset}_val"
    raw_weights = {}
    for source_name in source_names:
        path = Path(results_base_dir) / val_dir / f"{source_name}.jsonl"
        f1s = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                f1s.append(float(json.loads(line).get("f1", 0.0)))
        raw_weights[source_name] = sum(f1s) / len(f1s) if f1s else 0.0

    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError(f"All-zero MoR weights for dataset={dataset}, sources={source_names}")
    return {name: w / total for name, w in raw_weights.items()}


def get_pending_samples(samples, done_ids, max_samples):
    pending = [
        (row_idx, sample)
        for row_idx, sample in enumerate(samples)
        if sample["id"] not in done_ids
    ]
    if max_samples is not None:
        pending = pending[:max_samples]
    return pending


def validate_source_item(source_name, item, sample, corpus_size, top_k):
    if item.get("question") != sample["question"]:
        raise ValueError(
            f"Mismatched question for sample {sample['id']} in source {source_name}"
        )

    top_k_indices = item.get("top_k_indices")
    top_k_scores = item.get("top_k_scores")
    if not isinstance(top_k_indices, list):
        raise ValueError(f"Source {source_name} sample {sample['id']} has no top_k_indices list")
    if not isinstance(top_k_scores, list):
        raise ValueError(f"Source {source_name} sample {sample['id']} has no top_k_scores list")
    if len(top_k_indices) != len(top_k_scores):
        raise ValueError(
            f"Source {source_name} sample {sample['id']} has mismatched "
            "top_k_indices/top_k_scores lengths"
        )


def get_source_item(sample, row_idx, source_name, source_maps, source_rows):
    item = source_maps[source_name].get(sample["id"])
    if item is not None:
        return item, False

    rows = source_rows[source_name]
    if row_idx < len(rows):
        row_item = rows[row_idx]
        if row_item.get("question") == sample["question"]:
            return row_item, True

    raise ValueError(
        f"Sample {sample['id']} is missing from MoR source {source_name}, "
        "and row-order fallback was not possible because the row is absent or "
        "the question text does not match."
    )


def build_mor_candidate(
    sample,
    row_idx,
    corpus,
    source_names,
    source_weights,
    source_maps,
    source_rows,
    top_k,
    row_fallback_counts,
    invalid_doc_counts,
    invalid_doc_examples,
):
    fused_scores = defaultdict(float)
    best_positions = {}
    source_details_by_doc = defaultdict(list)

    for source_order, source_name in enumerate(source_names):
        item, used_row_fallback = get_source_item(
            sample, row_idx, source_name, source_maps, source_rows,
        )
        if used_row_fallback:
            row_fallback_counts[source_name] = row_fallback_counts.get(source_name, 0) + 1

        validate_source_item(source_name, item, sample, len(corpus), top_k)

        source_indices = item["top_k_indices"][:top_k]
        source_scores = item["top_k_scores"][:top_k]

        if len(source_scores) > 1 and max(source_scores) > min(source_scores):
            lo, hi = min(source_scores), max(source_scores)
            normalized = [(s - lo) / (hi - lo) for s in source_scores]
        else:
            normalized = [1.0] * len(source_scores)

        weight = source_weights[source_name]
        for rank_zero, doc_idx in enumerate(source_indices):
            rank = rank_zero + 1
            if not isinstance(doc_idx, int) or doc_idx < 0 or doc_idx >= len(corpus):
                invalid_doc_counts[source_name] = invalid_doc_counts.get(source_name, 0) + 1
                examples = invalid_doc_examples[source_name]
                if len(examples) < 5:
                    examples.append({
                        "sample_id": sample["id"],
                        "doc_index": doc_idx,
                        "rank": rank,
                    })
                continue

            contribution = weight * normalized[rank_zero]
            fused_scores[doc_idx] += contribution

            position = (source_order, rank, doc_idx)
            if doc_idx not in best_positions or position < best_positions[doc_idx]:
                best_positions[doc_idx] = position

            source_details_by_doc[doc_idx].append({
                "retriever": source_name,
                "rank": rank,
                "raw_score": float(source_scores[rank_zero]),
                "normalized_score": float(normalized[rank_zero]),
                "weight": float(weight),
                "mor_contribution": float(contribution),
            })

    if not fused_scores:
        raise ValueError(
            f"Sample {sample['id']} has no valid MoR documents after filtering invalid doc indices."
        )

    ranked_doc_indices = sorted(
        fused_scores,
        key=lambda doc_idx: (
            -fused_scores[doc_idx],
            best_positions[doc_idx][0],
            best_positions[doc_idx][1],
            doc_idx,
        ),
    )[:top_k]

    top_k_docs = [corpus[doc_idx] for doc_idx in ranked_doc_indices]
    top_k_scores = [float(fused_scores[doc_idx]) for doc_idx in ranked_doc_indices]
    final_doc_sources = [
        {
            "doc_index": int(doc_idx),
            "mor_score": float(fused_scores[doc_idx]),
            "sources": source_details_by_doc[doc_idx],
        }
        for doc_idx in ranked_doc_indices
    ]

    return ranked_doc_indices, top_k_docs, top_k_scores, final_doc_sources


def extract_answer_and_f1(response, gold_answers):
    from src.metric import get_f1

    try:
        pred_ans_raw = response.split("Answer:", 1)[1].strip()
        if "[ANS]" in pred_ans_raw:
            pred_ans = [ans.strip() for ans in pred_ans_raw.split("[ANS]") if ans.strip()]
            f1_scores = [get_f1(gold_answers, ans) for ans in pred_ans]
            f1 = float(max(f1_scores)) if f1_scores else 0.0
        else:
            pred_ans = pred_ans_raw
            f1 = float(get_f1(gold_answers, pred_ans))
    except Exception:
        pred_ans = response
        f1 = float(get_f1(gold_answers, pred_ans))

    return pred_ans, f1


def build_output_item(sample, top_k_docs, top_k_indices, top_k_scores, response, latency, extra_fields=None):
    from src.metric import get_ndcg, get_recall

    pred_ans, f1 = extract_answer_and_f1(response, sample["answers"])

    gold_doc_list = sample["gold_doc_list"]
    if top_k_docs and gold_doc_list is not None:
        recall = float(get_recall(gold_doc_list, top_k_docs))
        ndcg = float(get_ndcg(gold_doc_list, top_k_docs))
    else:
        recall = 0.0
        ndcg = 0.0

    out_item = {
        "id": sample["id"],
        "gold_doc_list": gold_doc_list,
        "recall": recall,
        "question": sample["question"],
        "answers": list(sample["answers"]),
        "ndcg": ndcg,
        "top_k_indices": [int(idx) for idx in top_k_indices],
        "top_k_scores": [float(score) for score in top_k_scores],
        "raw_response": [response],
        "pred_ans": pred_ans,
        "f1": f1,
        "latency": float(latency),
    }
    if extra_fields:
        out_item.update(extra_fields)
    return out_item


def write_jsonl_item(handle, item):
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()


class VLLMGenerator:
    def __init__(self, args):
        if args.llm_port is None:
            raise ValueError("--llm_port is required when --backend vllm")

        start_time = time.time()
        log(
            "Initializing vLLM generator: "
            f"model_path={args.model_path or args.llm}, port={args.llm_port}"
        )
        from openai import OpenAI
        from transformers import AutoTokenizer
        from src.llm import model_map

        self.model_name = args.model_path or args.llm
        tokenizer_name = args.model_path or model_map.get(args.llm, args.llm)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.client = OpenAI(
            base_url=f"http://localhost:{args.llm_port}/v1",
            api_key="token-abc123",
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        log(f"Initialized vLLM generator in {time.time() - start_time:.2f}s")

    def generate(self, prompt):
        from src.llm import query_llm

        return query_llm(
            prompt,
            self.tokenizer,
            self.client,
            self.model_name,
            self.max_new_tokens,
            temperature=self.temperature,
        )


class HFGenerator:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from src.llm import model_map

        start_time = time.time()
        model_id = args.model_id or model_map.get(args.llm)
        if model_id is None:
            raise ValueError(f"Unsupported llm alias without --model_id override: {args.llm}")

        llm_device = resolve_hf_device()
        torch_dtype = resolve_hf_dtype(args.hf_dtype, llm_device)
        log(
            "Initializing HF generator: "
            f"llm={args.llm}, model_id={model_id}, device={llm_device}, "
            f"dtype={torch_dtype}, max_input_tokens={args.max_input_tokens}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if llm_device == "auto":
            model_kwargs["device_map"] = "auto"
        elif llm_device != "cpu":
            model_kwargs["device_map"] = {"": llm_device}

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.model.eval()

        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature
        self.max_input_tokens = args.max_input_tokens
        log(f"Initialized HF generator in {time.time() - start_time:.2f}s")

    def generate(self, prompt):
        import torch

        inputs = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        original_input_tokens = input_ids.shape[-1]
        if self.max_input_tokens and original_input_tokens > self.max_input_tokens:
            input_ids = input_ids[:, -self.max_input_tokens:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_input_tokens:]
            log(
                "HF prompt truncated: "
                f"{original_input_tokens} -> {input_ids.shape[-1]} input tokens"
            )

        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        generation_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.max_new_tokens,
            "eos_token_id": terminators,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        if self.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.temperature
        else:
            generation_kwargs["do_sample"] = False
            generation_kwargs["temperature"] = None
            generation_kwargs["top_p"] = None

        with torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)

        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)


def resolve_hf_device():
    import os
    import torch

    requested_device = os.getenv("RAGCOMPASS_LLM_DEVICE", "").strip()
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def resolve_hf_dtype(hf_dtype, llm_device):
    import torch

    if hf_dtype == "float16":
        return torch.float16
    if hf_dtype == "bfloat16":
        return torch.bfloat16
    if hf_dtype == "float32":
        return torch.float32
    if hf_dtype != "auto":
        raise ValueError(f"Unsupported --hf_dtype: {hf_dtype}")

    if llm_device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    return torch.float16


def build_generator(args):
    if args.backend == "vllm":
        return VLLMGenerator(args)
    if args.backend == "hf":
        return HFGenerator(args)
    raise ValueError(f"Unsupported backend: {args.backend}")


def generate_for_docs(generator, prompt_manager, sample, top_k_docs, prompt_style):
    if prompt_style == "demo":
        prompt = prompt_manager.render_qa_ircot(top_k_docs, sample["question"], [])
    elif prompt_style == "no_demo":
        prompt = prompt_manager.render_qa_ircot_no_demo(top_k_docs, sample["question"], [])
    else:
        raise ValueError(f"Unsupported prompt style: {prompt_style}")
    return generator.generate(prompt)


def process_dataset(dataset, args, generator):
    from src.data import load_corpus, load_qa
    from src.prompt import PromptTemplateManager

    dataset_start_time = time.time()
    log(f"========== Dataset start: {dataset} ==========")
    dataset_dir = Path(args.results_base_dir) / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)

    mor_out_file = output_path(args.results_base_dir, dataset, MOR_OUTPUT_NAME)
    log(f"Dataset={dataset}: MoR output -> {mor_out_file}")

    mor_done_ids = load_done_ids(mor_out_file)

    qa_start_time = time.time()
    log(f"Dataset={dataset}: loading QA samples")
    samples = load_qa(dataset)
    log(f"Dataset={dataset}: loaded {len(samples)} QA samples in {time.time() - qa_start_time:.2f}s")

    pending_samples = get_pending_samples(samples, mor_done_ids, args.max_samples)

    log(
        f"Dataset={dataset}: total={len(samples)}, pending={len(pending_samples)}, "
        f"mor_done={len(mor_done_ids)}"
    )
    if not pending_samples:
        eval_existing_output(mor_out_file)
        log(f"========== Dataset done: {dataset} (no pending samples) ==========")
        return generator

    if generator is None:
        generator = build_generator(args)

    corpus_start_time = time.time()
    log(f"Dataset={dataset}: loading corpus for MoR preprocessing")
    corpus = load_corpus(dataset)
    log(f"Dataset={dataset}: loaded {len(corpus)} corpus docs in {time.time() - corpus_start_time:.2f}s")

    source_start_time = time.time()
    source_names = resolve_mor_sources(args.results_base_dir, dataset, args.retrievers)
    source_maps, source_rows, source_counts = load_source_maps(
        args.results_base_dir, dataset, source_names,
    )
    source_weights = compute_mor_weights(args.results_base_dir, dataset, source_names)
    log(
        f"Dataset={dataset}: MoR sources={source_names}, source_counts={source_counts}, "
        f"weights(val F1, normalized)={ {k: round(v, 4) for k, v in source_weights.items()} }, "
        f"loaded in {time.time() - source_start_time:.2f}s"
    )

    prompt_manager = PromptTemplateManager()
    generated_count = 0
    latency_sum = 0.0
    row_fallback_counts = defaultdict(int)
    invalid_doc_counts = defaultdict(int)
    invalid_doc_examples = defaultdict(list)
    log(
        f"Dataset={dataset}: starting downstream generation for {len(pending_samples)} pending sample(s); "
        f"log_every={args.log_every}"
    )
    with mor_out_file.open("a", encoding="utf-8") as mor_handle:
        for sample_idx, (row_idx, sample) in enumerate(
            tqdm(pending_samples, desc=f"Generating MoR outputs for {dataset}"),
            start=1,
        ):
            start_time = time.time()
            top_k_indices, top_k_docs, top_k_scores, final_doc_sources = build_mor_candidate(
                sample=sample,
                row_idx=row_idx,
                corpus=corpus,
                source_names=source_names,
                source_weights=source_weights,
                source_maps=source_maps,
                source_rows=source_rows,
                top_k=args.top_k,
                row_fallback_counts=row_fallback_counts,
                invalid_doc_counts=invalid_doc_counts,
                invalid_doc_examples=invalid_doc_examples,
            )
            response = generate_for_docs(
                generator, prompt_manager, sample, top_k_docs, args.prompt_style,
            )
            latency = time.time() - start_time
            out_item = build_output_item(
                sample=sample,
                top_k_docs=top_k_docs,
                top_k_indices=top_k_indices,
                top_k_scores=top_k_scores,
                response=response,
                latency=latency,
                extra_fields={
                    "fusion_sources": source_names,
                    "fusion_weights": source_weights,
                    "final_doc_sources": final_doc_sources,
                },
            )
            write_jsonl_item(mor_handle, out_item)
            mor_done_ids.add(sample["id"])
            generated_count += 1
            latency_sum += latency

            if args.log_every and (
                sample_idx == 1
                or sample_idx % args.log_every == 0
                or sample_idx == len(pending_samples)
            ):
                avg_latency = latency_sum / generated_count if generated_count else 0.0
                log(
                    f"Dataset={dataset}: progress {sample_idx}/{len(pending_samples)} "
                    f"latest_id={sample['id']} generated={generated_count} avg_latency={avg_latency:.2f}s"
                )

    log(f"Dataset={dataset}: generation complete; new_mor={generated_count}")
    if row_fallback_counts:
        log(f"Dataset={dataset}: row-order fallback counts={dict(row_fallback_counts)}")
    if invalid_doc_counts:
        log(f"Dataset={dataset}: skipped invalid doc-index counts={dict(invalid_doc_counts)}")
        log(f"Dataset={dataset}: invalid doc-index examples={dict(invalid_doc_examples)}")
    eval_existing_output(mor_out_file)
    log(f"========== Dataset done: {dataset} in {time.time() - dataset_start_time:.2f}s ==========")
    return generator


def eval_existing_output(mor_out_file):
    from utils import eval as eval_file

    if mor_out_file.exists() and mor_out_file.stat().st_size > 0:
        log(f"Evaluating {MOR_OUTPUT_NAME}: {mor_out_file}")
        eval_file(str(mor_out_file))


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate MoR-style (zero-shot weighted score fusion) action outputs from cached retriever results."
    )
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--llm", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--llm_port",
        type=int,
        default=None,
        help="OpenAI-compatible vLLM server port. Required for --backend vllm.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model name/path served by vLLM. Defaults to --llm.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Optional Hugging Face model id override for --backend hf.",
    )
    parser.add_argument(
        "--results_base_dir",
        type=str,
        default=DEFAULT_RESULTS_BASE_DIR,
        help="Base directory containing cached retriever JSONL files and action outputs.",
    )
    parser.add_argument(
        "--artifact_root",
        type=str,
        default=None,
        help="Root directory for model caches when this script is run directly.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=DEFAULT_DATASETS,
        help="Comma-separated dataset groups or explicit split names.",
    )
    parser.add_argument(
        "--no_expand_groups",
        action="store_true",
        help=(
            "Treat --datasets entries as literal subset directory names "
            "(skip train/val/test group expansion, e.g. to target only test splits)."
        ),
    )
    parser.add_argument(
        "--retrievers",
        type=str,
        default="auto",
        help=(
            "Comma-separated cached retriever result names to fuse. "
            "Default auto uses active one-step files when present."
        ),
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument(
        "--max_input_tokens",
        type=int,
        default=4096,
        help="Maximum HF input prompt tokens. Longer prompts are left-truncated. Set 0 to disable.",
    )
    parser.add_argument(
        "--hf_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for --backend hf. float16 is the safer default after BF16 CUBLAS failures.",
    )
    parser.add_argument(
        "--prompt_style",
        choices=["demo", "no_demo"],
        default="no_demo",
        help="QA prompt style. no_demo is shorter and safer for HF generation.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--log_every",
        type=int,
        default=50,
        help="Log downstream generation progress every N pending samples. Set 0 to disable progress logs.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(os.environ, args.artifact_root, override=True)
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.log_every < 0:
        raise ValueError("--log_every must be non-negative")

    datasets = resolve_datasets(args.datasets, expand_groups=not args.no_expand_groups)
    log(f"Datasets to process: {datasets}")
    log(f"Backend: {args.backend}")
    log(f"Results base dir: {args.results_base_dir}")
    log(
        f"Generation config: top_k={args.top_k}, "
        f"max_samples={args.max_samples}, max_new_tokens={args.max_new_tokens}, "
        f"temperature={args.temperature}"
    )

    generator = None
    for dataset in datasets:
        generator = process_dataset(dataset, args, generator)

    log("MoR action generation complete.")


if __name__ == "__main__":
    main()
