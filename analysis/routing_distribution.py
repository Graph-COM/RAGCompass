"""
Oracle-gap breakdown and routing-decision distribution (chosen vs. oracle
retriever).

Consumes the per-query JSONL written by analysis/dump_decisions.py (one record
per test query: chosen/oracle/best-fixed retriever, their F1s,
oracle_gain = oracle_f1 - chosen_f1, dataset). Pure analysis, no GPU/training.

Usage:
    python -m analysis.routing_distribution \
        --input outputs/diagnostics/routing_decisions_s42.jsonl \
        --output_dir outputs/diagnostics/figures
"""

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def oracle_gap_breakdown(records):
    by_dataset = {}
    datasets = sorted({r["dataset"] for r in records})
    for ds in datasets:
        gains = [r["oracle_gain"] for r in records if r["dataset"] == ds]
        by_dataset[ds] = {
            "n": len(gains),
            "mean_oracle_gain": float(np.mean(gains)),
            "median_oracle_gain": float(np.median(gains)),
            "frac_matches_oracle": float(
                np.mean([r["matches_oracle"] for r in records if r["dataset"] == ds])
            ),
        }
    all_gains = [r["oracle_gain"] for r in records]
    overall = {
        "n": len(all_gains),
        "mean_oracle_gain": float(np.mean(all_gains)),
        "median_oracle_gain": float(np.median(all_gains)),
        "frac_matches_oracle": float(np.mean([r["matches_oracle"] for r in records])),
    }
    return by_dataset, overall, all_gains


def routing_decision_distribution(records):
    chosen_counts = Counter(r["chosen_retriever"] for r in records)
    oracle_counts = Counter(r["oracle_retriever"] for r in records)
    best_fixed = records[0]["best_fixed_retriever"]
    n = len(records)

    retrievers = sorted(set(chosen_counts) | set(oracle_counts))
    confusion = {r: Counter() for r in retrievers}
    for rec in records:
        confusion[rec["oracle_retriever"]][rec["chosen_retriever"]] += 1

    return {
        "best_fixed_retriever": best_fixed,
        "chosen_distribution": {r: chosen_counts.get(r, 0) / n for r in retrievers},
        "oracle_distribution": {r: oracle_counts.get(r, 0) / n for r in retrievers},
        "confusion_oracle_rows_chosen_cols": {
            oracle_r: {chosen_r: confusion[oracle_r].get(chosen_r, 0) for chosen_r in retrievers}
            for oracle_r in retrievers
        },
        "retrievers": retrievers,
    }


def plot_oracle_gap_cdf(all_gains, output_path):
    sorted_gains = np.sort(all_gains)
    cdf = np.arange(1, len(sorted_gains) + 1) / len(sorted_gains)
    plt.figure(figsize=(5, 4))
    plt.plot(sorted_gains, cdf)
    plt.xlabel("Oracle gain (oracle F1 - chosen F1)")
    plt.ylabel("Cumulative fraction of test queries")
    plt.title("Oracle-gap CDF")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_routing_distribution(dist, output_path):
    retrievers = dist["retrievers"]
    chosen = [dist["chosen_distribution"][r] for r in retrievers]
    oracle = [dist["oracle_distribution"][r] for r in retrievers]
    x = np.arange(len(retrievers))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, oracle, width, label="Oracle")
    plt.bar(x + width / 2, chosen, width, label="Ours (chosen)")
    plt.xticks(x, retrievers, rotation=30, ha="right")
    plt.ylabel("Fraction of test queries")
    plt.title("Routing-decision distribution vs. oracle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} per-query records")

    by_dataset, overall, all_gains = oracle_gap_breakdown(records)
    dist = routing_decision_distribution(records)

    print("\n=== Oracle-gap breakdown ===")
    for ds, m in by_dataset.items():
        print(f"{ds:20s} n={m['n']:5d}  mean_gain={m['mean_oracle_gain']:.4f}  "
              f"median_gain={m['median_oracle_gain']:.4f}  matches_oracle={m['frac_matches_oracle']:.3f}")
    print(f"{'OVERALL':20s} n={overall['n']:5d}  mean_gain={overall['mean_oracle_gain']:.4f}  "
          f"median_gain={overall['median_oracle_gain']:.4f}  matches_oracle={overall['frac_matches_oracle']:.3f}")

    print("\n=== Routing-decision distribution ===")
    print(f"Best fixed retriever: {dist['best_fixed_retriever']}")
    for r in dist["retrievers"]:
        print(f"{r:25s} oracle={dist['oracle_distribution'][r]:.3f}  chosen={dist['chosen_distribution'][r]:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    plot_oracle_gap_cdf(all_gains, os.path.join(args.output_dir, "oracle_gap_cdf.pdf"))
    plot_routing_distribution(dist, os.path.join(args.output_dir, "routing_decision_distribution.pdf"))
    print(f"\nSaved figures -> {args.output_dir}/oracle_gap_cdf.pdf, routing_decision_distribution.pdf")

    output_json = args.output_json or os.path.join(args.output_dir, "routing_distribution.json")
    with open(output_json, "w") as f:
        json.dump({
            "oracle_gap_by_dataset": by_dataset,
            "oracle_gap_overall": overall,
            "routing_distribution": dist,
        }, f, indent=2)
    print(f"Saved -> {output_json}")


if __name__ == "__main__":
    main()
