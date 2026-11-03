"""
Reader relabeling: re-answer each retriever's cached evidence with a different
reader LLM (unseen-reader experiment) or at a shallower retrieval depth
(unseen-depth experiment, via --top_k).

The router's per-query choice does not depend on the reader at all -- it scores
evidence sets. So this needs no retraining: only the utilities y_i change. We
therefore re-answer each retriever's ALREADY-RETRIEVED evidence with a new
reader and rescore the router's existing choices under the new utilities.

Retrieval is deliberately NOT re-run: the top_k_indices already stored in each
result JSONL are reused verbatim, so the evidence is identical and the reader is
the only thing that varies. That also avoids re-running HippoRAG2's KG pipeline.

Restricted by default to the filtered test ids in the decisions file, so the cost is
|test| x R generations rather than the full corpus.

Usage:
    python -m analysis.relabel_with_reader \
        --decisions outputs/diagnostics/routing_decisions_s42.jsonl \
        --dataset nq --model_id Qwen/Qwen2.5-7B-Instruct --reader_tag Qwen2.5-7B-Instruct \
        --output_dir Qwen2.5-7B-Instruct_Qwen3-Embedding-0.6B
"""

import argparse
import json
import os
import sys
import time

import torch
from tqdm import tqdm

# Qwen2 attention through PyTorch SDPA picks the cuDNN backend, which on some
# GPU/cuDNN/torch combinations fails with
#   "cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built".
# The other SDPA backends are fine, so
# disable only the cuDNN one rather than forcing eager attention everywhere.
# Scoped to this script; the shared reader path is untouched.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_hf import load_model  # noqa: E402
from src.artifacts import apply_cache_environment  # noqa: E402
from src.data import load_corpus  # noqa: E402
from src.llm_hf import query_llm_hf  # noqa: E402
from src.metric import get_f1  # noqa: E402
from src.prompt import PromptTemplateManager  # noqa: E402

DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]


def item_key(item, idx):
    for key in ("id", "_id", "qid", "question_id"):
        if key in item:
            return str(item[key])
    return f"row_{idx}"


def parse_answer(response, gold_answers):
    """
    Same answer extraction and scoring as main_hf.ircot, so the new reader's F1
    is computed exactly like the numbers it will be compared against.
    """
    try:
        raw = response.split("Answer:")[1].strip()
        if "[ANS]" in raw:
            parts = [part.strip() for part in raw.split("[ANS]") if part.strip()]
            scores = [get_f1(gold_answers, part) for part in parts]
            return parts, float(max(scores)) if scores else 0.0
        return raw, float(get_f1(gold_answers, raw))
    except IndexError:
        return response, float(get_f1(gold_answers, response))


def main():
    parser = argparse.ArgumentParser(description="Re-answer cached evidence with a second reader")
    parser.add_argument("--decisions", type=str, default=None,
                        help="Restrict to the ids in this per-query decisions JSONL (the filtered "
                             "test set). Omit to relabel every row in the dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument("--source_base_dir", type=str,
                        default="Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--reader_tag", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--artifact_root", type=str, default=None)
    args = parser.parse_args()

    apply_cache_environment(artifact_root=args.artifact_root)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    keep_ids = None
    if args.decisions:
        keep_ids = set()
        with open(args.decisions) as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record["dataset"] == args.dataset:
                        keep_ids.add(str(record["id"]))
        print(f"Restricting to {len(keep_ids)} ids from {args.decisions}")

    print(f"Loading corpus for {args.dataset}...")
    corpus = load_corpus(args.dataset)
    prompt_manager = PromptTemplateManager()
    tokenizer, model = load_model(args.reader_tag, args.model_id)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    for retriever in retrievers:
        src_path = os.path.join(args.source_base_dir, args.dataset, f"{retriever}.jsonl")
        out_path = os.path.join(out_dir, f"{retriever}.jsonl")

        rows = []
        with open(src_path) as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if keep_ids is not None and item_key(item, idx) not in keep_ids:
                    continue
                rows.append((item_key(item, idx), item))

        # Resume support: an interrupted or requeued job must not redo work,
        # and must not append duplicate ids.
        done = set()
        if os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        done.add(str(json.loads(line)["id"]))
            print(f"{retriever}: {len(done)} rows already done, resuming")
        pending = [(rid, item) for rid, item in rows if rid not in done]
        if args.max_samples:
            pending = pending[: args.max_samples]

        print(f"{retriever}: {len(pending)} rows to answer -> {out_path}")
        start = time.time()
        with open(out_path, "a", encoding="utf-8") as fout:
            for rid, item in tqdm(pending, desc=f"{args.dataset}/{retriever}"):
                indices = item.get("top_k_indices", [])[: args.top_k]
                docs = [corpus[i] for i in indices]
                # When --top_k is smaller than what the source run retrieved,
                # the evidence set genuinely changes, so the retrieval-side
                # fields must change with it. top_k_scores is truncated to stay
                # aligned with top_k_indices (a mismatched pair breaks every
                # downstream consumer), and recall/ndcg are dropped rather than
                # copied: they were computed over the source depth and would be
                # silently wrong at this one. f1 below IS recomputed, which is
                # the label the depth study reads.
                src_scores = item.get("top_k_scores")
                truncated = len(indices) < len(item.get("top_k_indices", []))
                out_scores = src_scores[: args.top_k] if src_scores else src_scores
                out_recall = None if truncated else item.get("recall")
                out_ndcg = None if truncated else item.get("ndcg")
                prompt = prompt_manager.render_qa_ircot(docs, item["question"], [])
                response = query_llm_hf(
                    prompt, tokenizer, model, args.max_new_tokens, args.temperature
                )
                pred_ans, f1 = parse_answer(response, item["answers"])
                # Retrieval-side fields are copied, not recomputed: the evidence
                # is byte-identical to the source run by construction.
                fout.write(json.dumps({
                    "id": rid,
                    "question": item["question"],
                    "answers": item["answers"],
                    "gold_doc_list": item.get("gold_doc_list"),
                    "recall": out_recall,
                    "ndcg": out_ndcg,
                    "top_k_indices": indices,
                    "top_k_scores": out_scores,
                    "raw_response": [response],
                    "pred_ans": pred_ans,
                    "f1": f1,
                    "reader": args.model_id,
                }, ensure_ascii=False) + "\n")
                fout.flush()
        elapsed = time.time() - start
        if pending:
            print(f"{retriever}: {len(pending)} rows in {elapsed / 60:.1f} min "
                  f"({elapsed / len(pending):.2f} s/query)")


if __name__ == "__main__":
    main()
