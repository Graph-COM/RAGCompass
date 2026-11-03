"""
Offline LLM-judge scoring pass for R3AG's retrieval-quality supervision signal
(Zhao et al. 2026, Section 4.2.1 / Appendix A.4: "Retrieval Quality Encoder").

R3AG trains its retrieval-quality encoder on positive/negative retriever sets
built from an external LLM judge's assessment of each retriever's documents,
independent of the reader's downstream answer. This script reproduces that
judge pass -- Qwen2.5-7B-Instruct by default (the original paper uses an 80B
judge, out of budget for this reproduction) --
scoring every (dataset, split, retriever, query) row's top-k documents on five
dimensions and writing the aggregate score to a cache file colocated with the
retrieval results, keyed by row id. It runs once; r3ag_train.py only reads
this cache, it never re-invokes the judge.

The no-retrieval action (no_retrieval_0_step) is skipped entirely: an empty
document set has nothing to judge on relevance/coverage/redundancy, so R0 is
assigned a fixed overall_score of 0.0 downstream (in r3ag_train.py) without
invoking the judge here -- R3AG's paper does not resolve this edge case
explicitly; treating "retrieved nothing" as the floor on retrieval quality is
the natural reading of the ranking objective in their Eq. 2.

Usage:
    python -m baselines.r3ag_judge \
        --datasets nq,musique,hotpotqa,2wikimultihopqa \
        --retrievers rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step \
        --judge_model Qwen2.5-7B-Instruct \
        --results_base_dir Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B \
        --output_base_dir r3ag_quality_judge_cache
"""

import argparse
import json
import os

from tqdm import tqdm

from kg_index import (
    env_bool,
    env_float,
    import_vllm_modules,
    vllm_batch_infer,
)
from src.data import load_corpus
from src.llm import model_map
from router.train_head import (
    discover_actions,
    docs_for_action_item,
    item_id,
    load_results_file,
    parse_dataset_names,
)
from router.shared_utils import DEFAULT_RESULTS_BASE_DIR, split_specs

NO_RETRIEVAL_ACTION = "no_retrieval_0_step"
SCORE_KEYS = (
    "relevance_score",
    "coverage_score",
    "accuracy_score",
    "ranking_score",
    "redundancy_score",
)
JUDGE_SYSTEM_PROMPT = (
    "You are a professional information retrieval quality evaluation expert. "
    "Please comprehensively evaluate the quality of documents returned by a "
    "retrieval system for the given question."
)
JUDGE_USER_TEMPLATE = """Question: {question}

Retrieved Documents:
{documents}

Please evaluate the overall retrieval quality from the following dimensions:
(1) Relevance, (2) Coverage, (3) Accuracy, (4) Ranking Quality, and (5) Redundancy.
Score each dimension from 1 (worst) to 5 (best)."""


def format_documents(docs):
    return "\n".join(f"Document {idx + 1}: {doc}" for idx, doc in enumerate(docs))


def build_judge_messages(question, docs):
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(question=question, documents=format_documents(docs)),
        },
    ]


def parse_judge_response(response):
    try:
        parsed = json.loads(response)
        scores = [float(parsed[key]) for key in SCORE_KEYS]
    except Exception:
        return None
    scores = [min(5.0, max(1.0, score)) for score in scores]
    return {key: score for key, score in zip(SCORE_KEYS, scores)}


def build_judge_client(judge_model, max_model_len, gpu_memory_utilization, enforce_eager):
    import torch

    _, LLM, _, _ = import_vllm_modules()
    client = LLM(
        model=model_map[judge_model],
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0,
        dtype="auto",
        enable_prefix_caching=True,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        quantization=None,
        load_format="auto",
        trust_remote_code=True,
    )
    return client, client.get_tokenizer()


def score_split_action(
    client,
    tokenizer,
    corpus,
    rows,
    top_k_docs,
    batch_size,
    max_tokens,
):
    scored_rows = []
    num_parse_failures = 0
    num_invalid_docs = 0

    pending_indices = []
    pending_messages = []

    def flush():
        nonlocal num_parse_failures
        if not pending_messages:
            return
        responses, _ = vllm_batch_infer(client, tokenizer, pending_messages, "retrieval_quality", max_tokens)
        for row_idx, response in zip(pending_indices, responses):
            parsed = parse_judge_response(response)
            if parsed is None:
                num_parse_failures += 1
                parsed = {key: 3.0 for key in SCORE_KEYS}
            overall_score = sum(parsed.values()) / len(parsed)
            row = rows[row_idx]
            scored_rows.append(
                {
                    "id": item_id(row, row_idx),
                    "overall_score": overall_score,
                    **parsed,
                }
            )
        pending_indices.clear()
        pending_messages.clear()

    for row_idx, row in enumerate(tqdm(rows, desc="Judging")):
        docs, invalid_indices = docs_for_action_item(row, corpus, top_k_docs)
        if docs is None:
            num_invalid_docs += 1
            continue
        pending_indices.append(row_idx)
        pending_messages.append(build_judge_messages(row["question"], docs))
        if len(pending_messages) >= batch_size:
            flush()
    flush()

    print(
        f"  scored={len(scored_rows)} parse_failures={num_parse_failures} "
        f"invalid_doc_indices={num_invalid_docs}"
    )
    return scored_rows


def main():
    parser = argparse.ArgumentParser(description="R3AG retrieval-quality LLM-judge scoring pass")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument(
        "--retrievers",
        type=str,
        default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step",
        help="Real retrievers to judge. no_retrieval_0_step is never judged (see module docstring).",
    )
    parser.add_argument("--judge_model", type=str, default="Qwen2.5-7B-Instruct", choices=list(model_map))
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--output_base_dir", type=str, default="r3ag_quality_judge_cache")
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if NO_RETRIEVAL_ACTION in args.retrievers:
        raise ValueError(f"{NO_RETRIEVAL_ACTION} is never judged; remove it from --retrievers.")

    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    retriever_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    print(f"Datasets: {dataset_names}")
    print(f"Retrievers to judge: {retriever_names}")
    print(f"Judge model: {args.judge_model} ({model_map[args.judge_model]})")

    enforce_eager = env_bool("RAGCOMPASS_KG_INDEX_VLLM_ENFORCE_EAGER", True)
    gpu_memory_utilization = env_float("RAGCOMPASS_KG_INDEX_GPU_MEMORY_UTILIZATION", 0.6)
    client, tokenizer = build_judge_client(
        args.judge_model, args.max_model_len, gpu_memory_utilization, enforce_eager
    )

    for dataset_name in dataset_names:
        for _, split_dir, corpus_name in split_specs(dataset_name):
            corpus = load_corpus(corpus_name)
            for retriever_name in retriever_names:
                results_path = os.path.join(args.results_base_dir, split_dir, f"{retriever_name}.jsonl")
                output_path = os.path.join(args.output_base_dir, split_dir, f"{retriever_name}.jsonl")
                if os.path.exists(output_path) and not args.overwrite:
                    print(f"Skipping {output_path} (already exists; pass --overwrite to redo).")
                    continue

                print(f"Judging {split_dir}/{retriever_name} ({results_path})...")
                rows = load_results_file(results_path)
                scored_rows = score_split_action(
                    client=client,
                    tokenizer=tokenizer,
                    corpus=corpus,
                    rows=rows,
                    top_k_docs=args.top_k_docs,
                    batch_size=args.batch_size,
                    max_tokens=args.max_tokens,
                )

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for row in scored_rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"  -> {output_path}")

    print("Done.")


if __name__ == "__main__":
    main()
