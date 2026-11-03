"""
Measure what the routing pipeline costs per query.

The pipeline does no generation: a routing decision is R backbone forward
passes plus one head forward.

What is timed, in the deployment shape (one query at a time, its R candidate
evidence sets encoded together):

  backbone : R forward passes over the routing prompt, no generation. This is
             the same code path the embedding cache uses
             (encode_flat_document_boundaries).
  head     : one LateInteractionHeadModel forward over the R pooled vectors.

Reported against two things already on disk, so the comparison needs no new
runs: the per-query retrieval `latency` field stored in every results row, and
the LLM-as-router costs measured by baselines/llm_router.py (0.358 s/query for
judge_evidence, 0.115 s/query for select_answers plus five reader generations).

CUDA is asynchronous, so every timed region synchronises before stopping the
clock; without that the backbone timing would silently measure kernel-launch
time. A warmup pass is discarded to exclude one-off allocator and autotuner
costs, which otherwise land entirely on the first query.

Usage (needs a GPU; ~20 min for the default 200 queries):
    python -m analysis.measure_latency \
        --pretrained_model_path <backbone checkpoint> \
        --head_dir <head checkpoint dir> \
        --output_json outputs/diagnostics/routing_latency.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.cache_embeddings import (  # noqa: E402
    encode_flat_document_boundaries,
)
from router.train_head import (  # noqa: E402
    LateInteractionHeadModel,
    QwenActionDataset,
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from router.shared_utils import (  # noqa: E402
    DEFAULT_RESULTS_BASE_DIR,
    load_results_file,
    split_specs,
)


def find_head_dir(root):
    """A run dir may store the head at its root or under checkpoints/best.

    Same resolution order as dump_decisions.py, duplicated rather
    than imported because that module is a script with side effects at import.
    """
    for cand in (root, os.path.join(root, "checkpoints", "best"),
                 os.path.join(root, "best"), os.path.join(root, "checkpoints")):
        if os.path.exists(os.path.join(cand, "hybrid_model.bin")):
            return cand
    raise FileNotFoundError(f"no hybrid_model.bin under {root}")

# Measured with baselines/llm_router.py on an A100, same 3735-query test set.
LLM_JUDGE_EVIDENCE_S = 0.358
LLM_SELECT_ANSWERS_S = 0.115


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def retrieval_latencies(base_dir, dataset_names, retrievers):
    """Per-query retrieval+generation latency already stored in the results rows.

    This is the cost the router is being compared against: what one fixed
    retriever's pipeline already spends per query, reader generation included.
    """
    per_retriever = {}
    for r in retrievers:
        vals = []
        for ds in dataset_names:
            for _, split_dir, _ in split_specs(ds):
                path = os.path.join(base_dir, split_dir, f"{r}.jsonl")
                if not os.path.exists(path):
                    continue
                for item in load_results_file(path):
                    lat = item.get("latency")
                    if lat is not None:
                        vals.append(float(lat))
        if vals:
            per_retriever[r] = {
                "n": len(vals),
                "mean_s": statistics.mean(vals),
                "median_s": statistics.median(vals),
            }
    return per_retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen3-Reranker-0.6B")
    ap.add_argument("--pretrained_model_path", required=True)
    ap.add_argument("--head_dir", required=True)
    ap.add_argument("--datasets", default="nq,musique,hotpotqa,2wikimultihopqa")
    ap.add_argument("--retrievers",
                    default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step")
    ap.add_argument("--results_base_dir", default=DEFAULT_RESULTS_BASE_DIR)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--top_k_docs", type=int, default=5)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--encode_batch_size", type=int, default=5)
    ap.add_argument("--n_queries", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible. A CPU number does not describe deployment.")

    dataset_names = parse_dataset_names(args.datasets, None)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)

    print("Loading test split...")
    splits = load_action_splits(
        dataset_names=dataset_names, action_names=action_names,
        threshold=args.threshold, max_samples=None,
        base_dir=args.results_base_dir, label_metric="f1",
        top_k_docs=args.top_k_docs,
    )
    test_data = splits["test"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    head = LateInteractionHeadModel.load_pretrained(
        find_head_dir(args.head_dir), action_names, device=device).eval()

    dataset = QwenActionDataset(
        test_data, tokenizer, max_length=args.max_length,
        top_k_docs=args.top_k_docs, routing_prompt=True,
        track_doc_boundaries=True,
    )
    # batch_size=1 is the deployment shape: one query, its R candidates encoded
    # together. Larger batches would measure throughput, not per-query latency.
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    backbone_times, head_times, total_times = [], [], []
    n_done = 0
    limit = args.warmup + args.n_queries

    with torch.no_grad():
        for batch in loader:
            if n_done >= limit:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            b, n_actions, seq_len = input_ids.shape
            flat_ids = input_ids.view(b * n_actions, seq_len)
            flat_mask = attention_mask.view(b * n_actions, seq_len)
            flat_offsets = batch["doc_end_offsets"].to(device).view(b * n_actions, -1)

            sync(device)
            t0 = time.perf_counter()
            encoded, _ = encode_flat_document_boundaries(
                base_model=base_model, input_ids=flat_ids, attention_mask=flat_mask,
                doc_end_offsets=flat_offsets, encode_batch_size=args.encode_batch_size,
            )
            sync(device)
            t1 = time.perf_counter()

            # encode_flat_document_boundaries returns CPU tensors -- the cache
            # builder wants them that way for storage -- so the head, which is on
            # the GPU, needs them moved back. The move sits OUTSIDE both timed
            # regions: it is an artefact of reusing the caching code path, not a
            # cost the deployed pipeline would pay (there the embeddings never
            # leave the device).
            emb = encoded.view(b, n_actions, -1) if encoded.dim() == 2 else encoded
            emb = emb.to(device).float()
            sync(device)
            t2 = time.perf_counter()
            _ = head(emb)
            sync(device)
            t3 = time.perf_counter()

            n_done += 1
            if n_done <= args.warmup:      # discard allocator/autotuner warmup
                continue
            backbone_times.append(t1 - t0)
            head_times.append(t3 - t2)
            total_times.append((t1 - t0) + (t3 - t2))

    def stats(xs):
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "mean_s": statistics.mean(xs),
            "median_s": statistics.median(xs),
            "p95_s": xs_sorted[int(0.95 * len(xs_sorted)) - 1] if xs else None,
        }

    routing = {
        "backbone_forwards": stats(backbone_times),
        "head_forward": stats(head_times),
        "total_routing_decision": stats(total_times),
        "n_retrievers": len(action_names),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "generation_tokens": 0,
    }

    retrieval = retrieval_latencies(args.results_base_dir, dataset_names, retrievers)
    mean_fixed = (statistics.mean(v["mean_s"] for v in retrieval.values())
                  if retrieval else None)

    out = {
        "routing": routing,
        "fixed_pipeline_latency_per_retriever": retrieval,
        "fixed_pipeline_mean_s": mean_fixed,
        "llm_router_baselines_s": {
            "judge_evidence": LLM_JUDGE_EVIDENCE_S,
            "select_answers_decision_only": LLM_SELECT_ANSWERS_S,
            "note": "select_answers additionally requires five full reader generations",
        },
    }
    if mean_fixed:
        out["routing_overhead_as_fraction_of_one_retriever_pipeline"] = (
            routing["total_routing_decision"]["mean_s"] / mean_fixed
        )

    print("\n=== Routing decision cost, reported pipeline (no generation) ===")
    print(f"  backbone ({len(action_names)} forwards): "
          f"{routing['backbone_forwards']['mean_s'] * 1000:.1f} ms mean, "
          f"{routing['backbone_forwards']['p95_s'] * 1000:.1f} ms p95")
    print(f"  head (1 forward):                {routing['head_forward']['mean_s'] * 1000:.2f} ms mean")
    print(f"  TOTAL per query:                 "
          f"{routing['total_routing_decision']['mean_s'] * 1000:.1f} ms mean, "
          f"{routing['total_routing_decision']['median_s'] * 1000:.1f} ms median")
    if mean_fixed:
        print(f"\n  one fixed retriever's pipeline:  {mean_fixed * 1000:.1f} ms mean "
              f"(retrieval + reader generation, from stored `latency`)")
        print(f"  routing adds:                    "
              f"{out['routing_overhead_as_fraction_of_one_retriever_pipeline']:.1%} of that")
    print(f"\n  LLM-as-router: judge_evidence {LLM_JUDGE_EVIDENCE_S * 1000:.0f} ms, "
          f"select_answers {LLM_SELECT_ANSWERS_S * 1000:.0f} ms + 5 generations")

    with open(args.output_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
