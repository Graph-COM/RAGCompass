"""
Aggregate R3AG per-seed output JSONs (from r3ag_train.py, one per seed via
`scripts/baselines.sh r3ag`) into a mean/std summary.

Usage:
    python -m baselines.r3ag_aggregate_seeds \
        --input_jsons out/r3ag_main_seed42.json out/r3ag_main_seed43.json out/r3ag_main_seed44.json \
        --output_json out/r3ag_main_aggregate.json
"""

import argparse
import json

import numpy as np

METRIC_KEYS = (
    "selected_f1",
    "oracle_f1",
    "best_fixed_f1",
    "improvement_vs_best_fixed",
    "picks_held_out_retriever",
    "picks_held_out_when_oracle",
)


def mean_std(values):
    clean_values = [float(value) for value in values if value is not None]
    if not clean_values:
        return None
    arr = np.asarray(clean_values, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "values": clean_values, "n": len(clean_values)}


def main():
    parser = argparse.ArgumentParser(description="Aggregate R3AG per-seed results")
    parser.add_argument("--input_jsons", type=str, nargs="+", required=True)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    per_seed = []
    for path in args.input_jsons:
        with open(path, "r", encoding="utf-8") as f:
            per_seed.append(json.load(f))

    seeds = [run.get("seed") for run in per_seed]
    print(f"Aggregating {len(per_seed)} runs, seeds={seeds}")

    aggregate = {"n_runs": len(per_seed), "seeds": seeds, "metrics": {}}
    for metric_key in METRIC_KEYS:
        test_values = [run.get("test_metrics", {}).get(metric_key) for run in per_seed]
        summary = mean_std(test_values)
        if summary is not None:
            aggregate["metrics"][metric_key] = summary
            print(f"test/{metric_key}: {summary['mean']:.4f} +/- {summary['std']:.4f} (n={summary['n']})")

    aggregate["per_seed"] = [
        {"seed": run.get("seed"), "test_metrics": run.get("test_metrics")} for run in per_seed
    ]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
