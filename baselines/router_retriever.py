"""
ROUTERRETRIEVER-style baseline (Lee et al. 2025, arXiv:2409.02685): training-free
nearest-centroid routing, adapted to our heterogeneous 5-retriever pool.

Their mechanism: each expert gets a "pilot embedding" -- the centroid of that
expert's own (pre-existing, independently available) domain-training-query
embeddings, computed once with no gradient step; a query is routed to the
expert whose pilot embedding it is nearest to (cosine similarity). Adding or
removing an expert costs one centroid computation, no retraining of anything
else.

Adapting this to our setting requires one real design decision, since our
"experts" are not domain-specialized instances of one embedding architecture
(their setting) but five structurally different retrieval mechanisms (BM25,
SPLADE, a dense retriever, ColBERTv2, HippoRAG2) with no shared embedding
space of their own. We build each retriever's pilot embedding as the centroid
of a SHARED query encoder's (Qwen3-Embedding-0.6B) embeddings for the training
queries where that retriever's OWN F1 is highest -- deliberately using each
retriever's OWN score column only, never a cross-retriever comparison, since
that is the closest analogue to "an expert's own domain data" and the only
version of this design that could even in principle stay defined under
leave-one-retriever-out.

Why --holdout_retriever is refused rather than computed: a held-out
retriever's pilot embedding needs SOME supervision derived from that
retriever's own F1 outcomes -- comparative or not -- to be built at all. There
is no retriever-agnostic fallback the way our own comparison head (no
per-retriever parameters) or a shared-feature scorer has. Under a fair reading
of "held out" (this retriever's outcomes are not used during training), the
centroid cannot be built, full stop -- the same structural wall LTRR's plain
R-way classifier hits (Section 5.2), not a limitation of this implementation.
Silently reusing the held-out retriever's own labels to build its centroid
anyway would answer a different, easier question than the one the leave-one-
retriever-out protocol asks of every other baseline in the paper.

Usage:
    python -m baselines.router_retriever \
        --datasets nq,musique,hotpotqa,2wikimultihopqa \
        --retrievers rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step \
        --seed 42 --output_json <path>
"""

import argparse
import json
import os

import numpy as np
import torch

from src.emb.encoder import Qwen3EmbeddingModel
from router.train_e2e import resplit_split_data
from router.train_head import (
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from baselines.ltrr import downstream_metrics, filter_split_by_dataset
from router.shared_utils import DEFAULT_RESULTS_BASE_DIR, set_seed

# Same reference numbers / tolerance ltrr_shared_scorer.py and r3ag_train.py
# gate against: the main results table's best fixed retriever and oracle on the pooled,
# no-holdout, 5-retriever test population.
REF_BEST_FIXED = 0.2781745126131807
REF_ORACLE = 0.7507691749237375
GATE_TOL = 1e-3


def build_embedding_cache(cache_path, dataset_names, action_names, results_base_dir, threshold, top_k_docs):
    """id -> unit-norm query embedding (numpy [D]), for every query in the full
    (non-holdout) 4-dataset train+val+test population. Computed once; every
    seed and every --holdout_dataset fold draws from the same id universe,
    since holdout filtering (see ltrr.filter_split_by_dataset) happens
    AFTER load_action_splits, not before."""
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    print("No embedding cache found; loading full (non-holdout) splits to enumerate queries...")
    splits = load_action_splits(
        dataset_names=dataset_names, action_names=action_names, threshold=threshold,
        max_samples=None, base_dir=results_base_dir, label_metric="f1", top_k_docs=top_k_docs,
    )
    ids, questions = [], []
    seen = set()
    for split_name in ("train", "val", "test"):
        for rid, q in zip(splits[split_name]["ids"], splits[split_name]["queries"]):
            if rid not in seen:
                seen.add(rid)
                ids.append(rid)
                questions.append(q)
    print(f"Encoding {len(questions)} unique queries with Qwen3-Embedding-0.6B...")

    encoder = Qwen3EmbeddingModel()
    embeddings = encoder.batch_encode_query_for_doc(questions)  # already unit-normalized numpy [N, D]

    cache = {rid: embeddings[i] for i, rid in enumerate(ids)}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Saved -> {cache_path}")
    return cache


def embed_split(split_data, cache):
    return np.stack([cache[rid] for rid in split_data["ids"]])


def compute_pilot_embeddings(query_emb, f1_matrix, top_fraction):
    """[R, D] centroids. Each retriever's centroid is built from ONLY that
    retriever's own F1 column (its own top-`top_fraction` training queries by
    its own F1) -- never a cross-retriever comparison. See module docstring."""
    n_examples, n_retrievers = f1_matrix.shape
    k = max(1, int(round(n_examples * top_fraction)))
    pilots = []
    for r in range(n_retrievers):
        top_idx = np.argsort(-f1_matrix[:, r])[:k]
        centroid = query_emb[top_idx].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        pilots.append(centroid)
    return np.stack(pilots, axis=0)


def route(query_emb, pilot_embeddings):
    norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12)
    sims = norm @ pilot_embeddings.T
    return sims.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description="ROUTERRETRIEVER-style nearest-centroid baseline")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str,
                         default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--resplit_train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--top_fractions", type=str, default="0.1,0.2,0.35,0.5,1.0",
        help="Grid for the one hyperparameter (fraction of training queries, ranked by a "
             "retriever's OWN F1, that make up its pilot centroid); selected on validation F1. "
             "1.0 = a naive full-mean centroid, ignoring F1 entirely, as a sanity floor.",
    )
    parser.add_argument(
        "--holdout_dataset", type=str, default=None,
        help="Leave-one-dataset-out: train on the other datasets, test only on this one.",
    )
    parser.add_argument(
        "--holdout_retriever", type=str, default=None,
        help="Leave-one-retriever-out. Refused: see module docstring for why a nearest-centroid "
             "router cannot be fairly evaluated under this protocol.",
    )
    parser.add_argument("--embedding_cache_path", type=str, default="router_retriever_query_embeddings.pt")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    if args.holdout_retriever:
        raise SystemExit(
            "--holdout_retriever is not implemented: a nearest-centroid router's per-retriever "
            "pilot embedding needs supervision derived from that retriever's own F1 outcomes, which "
            "is exactly what 'held out' withholds, so it cannot be fairly built for the held-out "
            "retriever without reusing information the protocol is meant to withhold (see this "
            "script's module docstring). R.RETRIEVER is not defined for an unseen retriever under "
            "this protocol."
        )

    set_seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    assert action_names == retrievers, f"discover_actions mismatch: {action_names} vs {retrievers}"

    cache = build_embedding_cache(
        args.embedding_cache_path, dataset_names, action_names, args.results_base_dir,
        args.threshold, args.top_k_docs,
    )

    print("Loading splits (identical filtering/test-set to the main pipeline)...")
    splits = load_action_splits(
        dataset_names=dataset_names, action_names=action_names, threshold=args.threshold,
        max_samples=None, base_dir=args.results_base_dir, label_metric="f1", top_k_docs=args.top_k_docs,
    )
    if args.holdout_dataset:
        if args.holdout_dataset not in dataset_names:
            raise ValueError(f"--holdout_dataset {args.holdout_dataset!r} not in {dataset_names}")
        splits = dict(splits)
        splits["train"] = filter_split_by_dataset(splits["train"], args.holdout_dataset, keep=False)
        splits["val"] = filter_split_by_dataset(splits["val"], args.holdout_dataset, keep=False)
        splits["test"] = filter_split_by_dataset(splits["test"], args.holdout_dataset, keep=True)
        print(f"Leave-one-dataset-out: holding out {args.holdout_dataset}")

    train_data, val_data = resplit_split_data(
        splits["train"], splits["val"], train_ratio=args.resplit_train_ratio, seed=args.seed,
    )
    test_data = splits["test"]
    print(f"Train: {len(train_data['labels'])}  Val: {len(val_data['labels'])}  Test: {len(test_data['labels'])}")

    emb_train = embed_split(train_data, cache)
    emb_val = embed_split(val_data, cache)
    emb_test = embed_split(test_data, cache)
    f1_train = np.array(train_data["f1_scores"])
    f1_val = np.array(val_data["f1_scores"])
    f1_test = np.array(test_data["f1_scores"])

    top_fractions = [float(x) for x in args.top_fractions.split(",") if x.strip()]
    sweep = {}
    for frac in top_fractions:
        pilots = compute_pilot_embeddings(emb_train, f1_train, frac)
        val_metrics = downstream_metrics(route(emb_val, pilots), f1_val)
        sweep[str(frac)] = val_metrics
        print(f"top_fraction={frac}: val selected_f1={val_metrics['selected_f1']:.4f}")

    best_frac = max(top_fractions, key=lambda f: sweep[str(f)]["selected_f1"])
    print(f"\nBest top_fraction on validation: {best_frac}")

    # Refit on train+val for the reported test number, same pattern as
    # ltrr.py's val-selected-then-refit protocol.
    emb_trainval = np.concatenate([emb_train, emb_val], axis=0)
    f1_trainval = np.concatenate([f1_train, f1_val], axis=0)
    pilots_final = compute_pilot_embeddings(emb_trainval, f1_trainval, best_frac)
    test_metrics = downstream_metrics(route(emb_test, pilots_final), f1_test)
    print(f"Test: {test_metrics}")

    gate_passed = None
    if not args.holdout_dataset:
        gate_passed = bool(
            abs(test_metrics["best_fixed_f1"] - REF_BEST_FIXED) < GATE_TOL
            and abs(test_metrics["oracle_f1"] - REF_ORACLE) < GATE_TOL
        )
        print(f"Example-set gate: best_fixed={test_metrics['best_fixed_f1']:.4f} (ref {REF_BEST_FIXED:.4f}), "
              f"oracle={test_metrics['oracle_f1']:.4f} (ref {REF_ORACLE:.4f}) -> {'PASS' if gate_passed else 'FAIL'}")
        if not gate_passed:
            raise SystemExit("Example-set gate FAILED: not the reported test set. Refusing to write output.")

    results = {
        "method": "router_retriever",
        "datasets": dataset_names,
        "holdout_dataset": args.holdout_dataset,
        "selection": "val_selected_f1",
        "seed": args.seed,
        "top_fraction_sweep": sweep,
        "best_top_fraction": best_frac,
        "n_train": int(len(train_data["labels"])),
        "n_val": int(len(val_data["labels"])),
        "n_test": int(len(test_data["labels"])),
        "val_metrics": sweep[str(best_frac)],
        "test_metrics": test_metrics,
        "best_model_metrics": test_metrics,
        "example_set_gate_passed": gate_passed,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
