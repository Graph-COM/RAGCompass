"""
"Why not just aggregate the answers?" -- training-free baselines that consume
the five retrievers' already-generated answers instead of routing to one.

Why not have the reader answer from each evidence set and then pick? Part of
that question is free: the five answers are already on disk (main_hf.py writes
`pred_ans` per query per retriever), so majority voting over them costs
nothing. The paid part (an LLM choosing among the answers, or judging the
evidence) is baselines/llm_router.py.

Baselines reported here, all on the same filtered test set as the router:
  * majority vote over the 5 normalized answers (ties -> best fixed retriever)
  * random retriever (mean over the 5, i.e. the no-information baseline)
  * best fixed retriever / oracle, reproduced as the sanity gate
  * "any retriever correct" rate -- how often SOME retriever gets F1 above a
    threshold, i.e. the ceiling any selection method could reach

Usage (CPU):
    python -m baselines.majority_vote \
        --decisions outputs/diagnostics/routing_decisions_s42.jsonl \
        --output_json outputs/diagnostics/majority_vote_s42.json
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metric import get_f1, normalize_answer  # noqa: E402

DEFAULT_RESULTS_BASE_DIR = "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]

EXPECTED_BEST_FIXED_F1 = 0.2782
EXPECTED_ORACLE_F1 = 0.7508
GATE_TOLERANCE = 5e-4


def as_answer_text(pred_ans):
    """
    `pred_ans` is a string for almost every row, but a few rows store a list
    (the reader's answer-extraction occasionally returns multiple spans). Join
    rather than drop, so voting sees the same text the F1 in the file scored.
    """
    if pred_ans is None:
        return ""
    if isinstance(pred_ans, str):
        return pred_ans
    if isinstance(pred_ans, (list, tuple)):
        return " ".join(as_answer_text(part) for part in pred_ans)
    return str(pred_ans)


def item_key(item, idx):
    for key in ("id", "_id", "qid", "question_id"):
        if key in item:
            return str(item[key])
    return f"row_{idx}"


def load_answers(base_dir, datasets, retrievers):
    """(dataset, retriever) -> {id: {"pred_ans": str, "answers": [...], "f1": float}}"""
    answers = defaultdict(dict)
    for dataset in datasets:
        for retriever in retrievers:
            path = os.path.join(base_dir, dataset, f"{retriever}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            with open(path) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    answers[(dataset, retriever)][item_key(item, idx)] = {
                        "pred_ans": as_answer_text(item.get("pred_ans")),
                        "answers": item.get("answers", []),
                        "f1": item.get("f1"),
                    }
    return answers


def main():
    parser = argparse.ArgumentParser(description="Training-free answer-aggregation baselines")
    parser.add_argument("--decisions", type=str, required=True)
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument("--correct_threshold", type=float, default=0.5,
                        help="F1 above which an answer counts as 'correct' for the any-correct rate.")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    decisions = []
    with open(args.decisions) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    datasets = sorted({row["dataset"] for row in decisions})
    print(f"Loaded {len(decisions)} test queries across {datasets}")

    # ---- Gate on the published numbers before reporting anything ------------
    per_retriever_f1 = {r: np.array([row["all_f1s"][r] for row in decisions]) for r in retrievers}
    best_fixed_name = max(per_retriever_f1, key=lambda r: per_retriever_f1[r].mean())
    best_fixed_f1 = float(per_retriever_f1[best_fixed_name].mean())
    oracle_f1 = float(np.mean([row["oracle_f1"] for row in decisions]))
    routed_f1 = float(np.mean([row["chosen_f1"] for row in decisions]))
    if (abs(best_fixed_f1 - EXPECTED_BEST_FIXED_F1) > GATE_TOLERANCE
            or abs(oracle_f1 - EXPECTED_ORACLE_F1) > GATE_TOLERANCE):
        raise SystemExit(
            f"Gate FAILED: best_fixed={best_fixed_f1:.4f} (expect {EXPECTED_BEST_FIXED_F1}), "
            f"oracle={oracle_f1:.4f} (expect {EXPECTED_ORACLE_F1})."
        )
    print(f"Gate PASSED: best fixed {best_fixed_f1:.4f} ({best_fixed_name}), oracle {oracle_f1:.4f}")

    answers = load_answers(args.results_base_dir, datasets, retrievers)

    majority_f1, random_f1, any_correct, unanimous = [], [], [], []
    # A "majority" needs at least two retrievers producing the same normalized
    # answer. Without this split the headline is uninterpretable: if agreement
    # is rare, majority voting is mostly its own tie-break rule.
    majority_exists, agreed_f1, agreed_best_fixed_f1 = [], [], []
    skipped = 0
    for row in decisions:
        dataset, qid = row["dataset"], str(row["id"])
        records = [answers[(dataset, r)].get(qid) for r in retrievers]
        if any(rec is None for rec in records):
            skipped += 1
            continue

        gold = records[0]["answers"]
        normalized = [normalize_answer(rec["pred_ans"]) for rec in records]
        counts = Counter(normalized)
        top_count = max(counts.values())
        # Deterministic tie-break: among the answers tied for most frequent,
        # take the one produced by the best fixed retriever if it is among them,
        # otherwise the first retriever in the canonical order that produced one.
        tied = {answer for answer, count in counts.items() if count == top_count}
        best_fixed_pos = retrievers.index(best_fixed_name)
        if normalized[best_fixed_pos] in tied:
            winner_pos = best_fixed_pos
        else:
            winner_pos = next(i for i, a in enumerate(normalized) if a in tied)

        voted_f1 = get_f1(gold, records[winner_pos]["pred_ans"])
        majority_f1.append(voted_f1)
        random_f1.append(float(np.mean([row["all_f1s"][r] for r in retrievers])))
        any_correct.append(float(max(row["all_f1s"][r] for r in retrievers) >= args.correct_threshold))
        unanimous.append(float(len(counts) == 1))
        majority_exists.append(float(top_count >= 2))
        if top_count >= 2:
            agreed_f1.append(voted_f1)
            agreed_best_fixed_f1.append(row["all_f1s"][best_fixed_name])

    if skipped:
        print(f"WARNING: skipped {skipped} queries with no matching answer row.")

    results = {
        "n": len(majority_f1),
        "n_skipped": skipped,
        "correct_threshold": args.correct_threshold,
        "majority_vote_f1": float(np.mean(majority_f1)),
        "random_retriever_f1": float(np.mean(random_f1)),
        "best_fixed_f1": best_fixed_f1,
        "best_fixed_retriever": best_fixed_name,
        "routed_f1": routed_f1,
        "oracle_f1": oracle_f1,
        "any_retriever_correct_rate": float(np.mean(any_correct)),
        "unanimous_answer_rate": float(np.mean(unanimous)),
        "majority_exists_rate": float(np.mean(majority_exists)),
        "majority_f1_where_majority_exists": float(np.mean(agreed_f1)) if agreed_f1 else None,
        "best_fixed_f1_where_majority_exists": float(np.mean(agreed_best_fixed_f1)) if agreed_best_fixed_f1 else None,
    }
    results["majority_vote_vs_best_fixed"] = results["majority_vote_f1"] - best_fixed_f1
    results["routed_vs_majority_vote"] = routed_f1 - results["majority_vote_f1"]

    print("\n=== Training-free aggregation over the 5 cached answers ===")
    print(f"  random retriever (mean)      {results['random_retriever_f1']:.4f}")
    print(f"  best fixed retriever         {best_fixed_f1:.4f}  ({best_fixed_name})")
    print(f"  majority vote over 5 answers {results['majority_vote_f1']:.4f}"
          f"  ({results['majority_vote_vs_best_fixed']:+.4f} vs best fixed)")
    print(f"  ROUTED (ours)                {routed_f1:.4f}"
          f"  ({results['routed_vs_majority_vote']:+.4f} vs majority vote)")
    print(f"  oracle over retrievers       {oracle_f1:.4f}")
    print(f"\n  >=2 retrievers agree on      {results['majority_exists_rate'] * 100:.1f}% of queries")
    if results["majority_f1_where_majority_exists"] is not None:
        print(f"    on those, voted answer F1  {results['majority_f1_where_majority_exists']:.4f} "
              f"vs best fixed {results['best_fixed_f1_where_majority_exists']:.4f}")
    print(f"  all 5 answers identical on   {results['unanimous_answer_rate'] * 100:.1f}% of queries")
    print(f"  some retriever reaches F1>={args.correct_threshold}: "
          f"{results['any_retriever_correct_rate'] * 100:.1f}% of queries")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
