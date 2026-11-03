"""
Does routing improve INTERMEDIATE retrieval metrics (recall@k, nDCG@k), not
just answer F1?

No GPU, no model, no re-retrieval. Every retriever's result JSONL already stores
per-query `recall` and `ndcg` (written by main_hf.py), and the router's per-query
choice is already dumped by analysis/dump_decisions.py, so this only joins the
two on query id.

Sanity gate: F1 recomputed from the per-query records must reproduce the
published best-fixed (0.2782) and oracle (0.7508) numbers. If it does not, the
id join is wrong and no retrieval number below should be believed.

Usage (CPU):
    python -m analysis.retrieval_metrics \
        --decisions outputs/diagnostics/routing_decisions_s42.jsonl \
        --output_json outputs/diagnostics/retrieval_metrics_s42.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

DEFAULT_RESULTS_BASE_DIR = "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]
# Fusion/mixture baselines are stored as extra pseudo-retrievers in the same
# layout by baselines/rrf.py / baselines/mor.py.
DEFAULT_FUSION_BASELINES = ["rrf_1_step", "mor_1_step"]

# Reference numbers (main results table) this script must reproduce before its
# own output is trusted.
EXPECTED_BEST_FIXED_F1 = 0.2782
EXPECTED_ORACLE_F1 = 0.7508
GATE_TOLERANCE = 5e-4


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def item_key(item, idx):
    """Same id convention as shared_utils.item_id."""
    for key in ("id", "_id", "qid", "question_id"):
        if key in item:
            return str(item[key])
    return f"row_{idx}"


def load_metric_lookup(base_dir, datasets, retrievers, metrics):
    """(dataset, retriever) -> {query_id: {metric: value}} over the TEST split dirs."""
    lookup = defaultdict(dict)
    missing_files = []
    for dataset in datasets:
        for retriever in retrievers:
            # split_specs() maps the test split to the bare dataset directory.
            path = os.path.join(base_dir, dataset, f"{retriever}.jsonl")
            if not os.path.exists(path):
                missing_files.append(path)
                continue
            # Streamed rather than list-loaded: these files are 9-27MB each and
            # only two floats per row are needed, so nothing else is retained.
            with open(path) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    values = {m: item[m] for m in metrics if m in item}
                    lookup[(dataset, retriever)][item_key(item, idx)] = values
    return lookup, missing_files


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "n": int(arr.size)}


def main():
    parser = argparse.ArgumentParser(description="Retrieval metrics of the routed choice")
    parser.add_argument("--decisions", type=str, required=True,
                        help="Per-query decisions JSONL from analysis/dump_decisions.py")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument("--fusion_baselines", type=str, default=",".join(DEFAULT_FUSION_BASELINES))
    parser.add_argument("--metrics", type=str, default="recall,ndcg")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--skip_gate", action="store_true",
                        help="Report even if the F1 reproduction gate fails (for debugging only).")
    args = parser.parse_args()

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    fusion = [r.strip() for r in args.fusion_baselines.split(",") if r.strip()]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    decisions = load_jsonl(args.decisions)
    print(f"Loaded {len(decisions)} routing decisions from {args.decisions}")
    datasets = sorted({row["dataset"] for row in decisions})
    print(f"Datasets: {datasets}")

    # ---- Gate: reproduce the published F1 numbers from these records --------
    chosen_f1 = np.array([row["chosen_f1"] for row in decisions])
    oracle_f1 = np.array([row["oracle_f1"] for row in decisions])
    per_retriever_f1 = {
        r: np.array([row["all_f1s"][r] for row in decisions]) for r in retrievers
    }
    best_fixed_name = max(per_retriever_f1, key=lambda r: per_retriever_f1[r].mean())
    best_fixed_f1 = per_retriever_f1[best_fixed_name].mean()

    print(f"\nGate: best-fixed F1 = {best_fixed_f1:.4f} ({best_fixed_name}), "
          f"oracle F1 = {oracle_f1.mean():.4f}, routed F1 = {chosen_f1.mean():.4f}")
    gate_ok = (
        abs(best_fixed_f1 - EXPECTED_BEST_FIXED_F1) < GATE_TOLERANCE
        and abs(oracle_f1.mean() - EXPECTED_ORACLE_F1) < GATE_TOLERANCE
    )
    if not gate_ok:
        message = (
            f"Gate FAILED: expected best_fixed={EXPECTED_BEST_FIXED_F1}, "
            f"oracle={EXPECTED_ORACLE_F1}; got {best_fixed_f1:.4f}, {oracle_f1.mean():.4f}. "
            "The example set is not the reported one."
        )
        if not args.skip_gate:
            raise SystemExit(message)
        print(f"WARNING: {message}")
    else:
        print("Gate PASSED: this is the same example set as the main results table.")

    # ---- Join retrieval metrics onto each decision --------------------------
    all_names = retrievers + fusion
    lookup, missing_files = load_metric_lookup(
        args.results_base_dir, datasets, all_names, metrics
    )
    if missing_files:
        print(f"\nNote: {len(missing_files)} result file(s) absent (baseline not run for that "
              f"dataset); those rows are skipped for that baseline only:")
        for path in missing_files[:6]:
            print(f"  {path}")

    routed = defaultdict(list)
    per_retriever = {r: defaultdict(list) for r in all_names}
    oracle_retrieval = defaultdict(list)
    per_dataset = defaultdict(lambda: defaultdict(list))
    unmatched = 0

    for row in decisions:
        dataset, qid, chosen = row["dataset"], str(row["id"]), row["chosen_retriever"]
        values_by_retriever = {}
        for name in all_names:
            values = lookup.get((dataset, name), {}).get(qid)
            if values:
                values_by_retriever[name] = values

        if chosen not in values_by_retriever:
            unmatched += 1
            continue

        for metric in metrics:
            chosen_value = values_by_retriever[chosen].get(metric)
            if chosen_value is None:
                continue
            routed[metric].append(chosen_value)
            per_dataset[dataset][metric].append(chosen_value)
            candidate_values = [
                values_by_retriever[r][metric]
                for r in retrievers
                if r in values_by_retriever and metric in values_by_retriever[r]
            ]
            if len(candidate_values) == len(retrievers):
                oracle_retrieval[metric].append(max(candidate_values))
            for name in all_names:
                value = values_by_retriever.get(name, {}).get(metric)
                if value is not None:
                    per_retriever[name][metric].append(value)

    if unmatched:
        print(f"\nWARNING: {unmatched}/{len(decisions)} decisions had no matching result row "
              f"for the chosen retriever (id join gap).")

    # ---- Report -------------------------------------------------------------
    results = {
        "n_decisions": len(decisions),
        "n_unmatched": unmatched,
        "gate_passed": bool(gate_ok),
        "f1": {
            "routed": float(chosen_f1.mean()),
            "best_fixed": float(best_fixed_f1),
            "best_fixed_retriever": best_fixed_name,
            "oracle": float(oracle_f1.mean()),
        },
        "retrieval": {},
        "per_dataset": {},
    }

    for metric in metrics:
        by_retriever = {
            name: summarize(per_retriever[name][metric])
            for name in all_names
            if per_retriever[name][metric]
        }
        fixed_only = {k: v for k, v in by_retriever.items() if k in retrievers}
        best_fixed_ret = max(fixed_only, key=lambda k: fixed_only[k]["mean"]) if fixed_only else None
        entry = {
            "routed": summarize(routed[metric]) if routed[metric] else None,
            "by_retriever": by_retriever,
            "best_fixed_retriever": best_fixed_ret,
            "best_fixed": fixed_only[best_fixed_ret]["mean"] if best_fixed_ret else None,
            "oracle": summarize(oracle_retrieval[metric])["mean"] if oracle_retrieval[metric] else None,
        }
        if entry["routed"] and entry["best_fixed"] is not None:
            entry["improvement_vs_best_fixed"] = entry["routed"]["mean"] - entry["best_fixed"]
        results["retrieval"][metric] = entry

        print(f"\n=== {metric}@5 ===")
        for name, stats in sorted(by_retriever.items(), key=lambda kv: -kv[1]["mean"]):
            marker = " (best fixed)" if name == best_fixed_ret else ""
            marker += " [fusion baseline]" if name in fusion else ""
            print(f"  {name:<28} {stats['mean']:.4f}  (n={stats['n']}){marker}")
        if entry["routed"]:
            print(f"  {'ROUTED (ours)':<28} {entry['routed']['mean']:.4f}  (n={entry['routed']['n']})")
        if entry["oracle"] is not None:
            print(f"  {'oracle over retrievers':<28} {entry['oracle']:.4f}")
        if "improvement_vs_best_fixed" in entry:
            print(f"  -> routed vs best fixed: {entry['improvement_vs_best_fixed']:+.4f}")

    for dataset in datasets:
        results["per_dataset"][dataset] = {
            metric: summarize(values) for metric, values in per_dataset[dataset].items()
        }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
