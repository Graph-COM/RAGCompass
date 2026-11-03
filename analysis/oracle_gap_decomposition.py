"""
Decompose the router's remaining gap to the oracle, and analyse its errors.

The gap's overall size and the routing distribution (routing_distribution.py)
do not say how much of the gap was *reachable* by any routing decision. That
is the question this script answers.

Every query is put in exactly one bucket, by the gap between the best and
second-best retriever's utility (the top-two gap):

  unroutable   -- no retriever answers the query at all (oracle F1 ~ 0). No
                  routing decision can win these; they are pure floor.
  near-tie     -- a routing decision exists but the top two candidates are
                  within `--tie_threshold` F1 of each other, so choosing
                  correctly is worth almost nothing.
  routable     -- a real choice with real stakes. This is the only bucket
                  where the router can meaningfully win or lose, and the only
                  one whose gap is honestly "missed".

Reported per bucket: share of queries, the router's F1, the oracle's, the best
fixed retriever's, and how much of the total oracle gap the bucket accounts
for. The headline number is the routable bucket's share of the gap -- the
part that is genuinely on the table.

The error analysis then asks whether the misses in the routable bucket are
systematic: which retriever the router reaches for when it is wrong, which one
it should have taken, and whether errors concentrate in a dataset or a
confusion pair rather than spreading evenly.

Reads the same per-query dump as its sibling scripts, so it needs no GPU and no
checkpoint:
  python -m analysis.oracle_gap_decomposition \
      --input  outputs/diagnostics/routing_decisions_s42_unfiltered.jsonl \
      --output_json outputs/diagnostics/oracle_gap_decomposition_s42.json
"""
import argparse
import json
import statistics
from collections import Counter, defaultdict


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def top_two_gap(all_f1s):
    """Utility gap between the best and second-best retriever for one query."""
    vals = sorted(all_f1s.values(), reverse=True)
    if len(vals) < 2:
        return 0.0
    return vals[0] - vals[1]


def bucket_of(row, tie_threshold, dead_threshold):
    oracle = row["oracle_f1"]
    if oracle <= dead_threshold:
        return "unroutable"
    if top_two_gap(row["all_f1s"]) <= tie_threshold:
        return "near-tie"
    return "routable"


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        return {"bucket": label, "n": 0, "share": 0.0}
    chosen = statistics.mean(r["chosen_f1"] for r in rows)
    oracle = statistics.mean(r["oracle_f1"] for r in rows)
    fixed = statistics.mean(r["best_fixed_f1"] for r in rows)
    gap = statistics.mean(r["oracle_f1"] - r["chosen_f1"] for r in rows)
    return {
        "bucket": label,
        "n": n,
        "chosen_f1": chosen,
        "oracle_f1": oracle,
        "best_fixed_f1": fixed,
        "gain_over_fixed": chosen - fixed,
        "mean_gap_to_oracle": gap,
        "total_gap_mass": gap * n,
        "matches_oracle": statistics.mean(1.0 if r["matches_oracle"] else 0.0 for r in rows),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="per_query_routing_decisions_*.jsonl")
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--tie_threshold", type=float, default=0.05,
                    help="Top-two utility gap at or below which the choice is a near-tie. "
                         "Defaults to 0.05, the same value the training example filter uses.")
    ap.add_argument("--dead_threshold", type=float, default=0.0,
                    help="Oracle F1 at or below which no retriever answers the query.")
    args = ap.parse_args()

    rows = load(args.input)
    print(f"Loaded {len(rows)} per-query records from {args.input}")

    buckets = defaultdict(list)
    for r in rows:
        buckets[bucket_of(r, args.tie_threshold, args.dead_threshold)].append(r)

    order = ["unroutable", "near-tie", "routable"]
    summaries = [summarize(buckets[b], b) for b in order]
    overall = summarize(rows, "ALL")
    total_gap_mass = overall["total_gap_mass"]

    print(f"\n=== Oracle-gap decomposition "
          f"(tie<={args.tie_threshold}, dead<={args.dead_threshold}) ===")
    hdr = (f"{'bucket':<12} {'n':>6} {'share':>7} {'routed':>8} {'oracle':>8} "
           f"{'fixed':>8} {'gap':>8} {'%of gap':>8} {'hit@1':>7}")
    print(hdr)
    for s in summaries:
        if s["n"] == 0:
            print(f"{s['bucket']:<12} {0:>6}")
            continue
        share = s["n"] / len(rows)
        pct_gap = s["total_gap_mass"] / total_gap_mass if total_gap_mass else 0.0
        print(f"{s['bucket']:<12} {s['n']:>6} {share:>7.1%} {s['chosen_f1']:>8.4f} "
              f"{s['oracle_f1']:>8.4f} {s['best_fixed_f1']:>8.4f} "
              f"{s['mean_gap_to_oracle']:>8.4f} {pct_gap:>8.1%} {s['matches_oracle']:>7.1%}")
    print(f"{'ALL':<12} {overall['n']:>6} {1.0:>7.1%} {overall['chosen_f1']:>8.4f} "
          f"{overall['oracle_f1']:>8.4f} {overall['best_fixed_f1']:>8.4f} "
          f"{overall['mean_gap_to_oracle']:>8.4f} {1.0:>8.1%} "
          f"{overall['matches_oracle']:>7.1%}")

    # --- Error analysis, restricted to the bucket where errors cost something ---
    routable = buckets["routable"]
    misses = [r for r in routable if not r["matches_oracle"]]
    print(f"\n=== Error analysis on the routable bucket "
          f"({len(misses)}/{len(routable)} misses) ===")

    if misses:
        loss = [r["oracle_f1"] - r["chosen_f1"] for r in misses]
        print(f"Mean F1 lost per miss: {statistics.mean(loss):.4f} "
              f"(median {statistics.median(loss):.4f}, max {max(loss):.4f})")

        beat_fixed = sum(1 for r in misses if r["chosen_f1"] >= r["best_fixed_f1"])
        print(f"Misses that still beat or match the best fixed retriever: "
              f"{beat_fixed}/{len(misses)} ({beat_fixed / len(misses):.1%})")

        by_ds = defaultdict(lambda: [0, 0])
        for r in routable:
            by_ds[r["dataset"]][1] += 1
            if not r["matches_oracle"]:
                by_ds[r["dataset"]][0] += 1
        print("\nMiss rate by dataset (routable only):")
        for ds in sorted(by_ds):
            m, tot = by_ds[ds]
            print(f"  {ds:<20} {m:>5}/{tot:<5} {m / tot:>6.1%}")

        pairs = Counter((r["oracle_retriever"], r["chosen_retriever"]) for r in misses)
        print("\nTop confusions (should have taken -> took):")
        for (want, got), c in pairs.most_common(8):
            print(f"  {want:<24} -> {got:<24} {c:>5} ({c / len(misses):>5.1%})")

        over = Counter(r["chosen_retriever"] for r in misses)
        under = Counter(r["oracle_retriever"] for r in misses)
        print("\nPer-retriever error balance (picked-when-wrong vs missed-when-right):")
        for r in sorted(set(over) | set(under)):
            print(f"  {r:<24} over={over.get(r, 0):>5}  under={under.get(r, 0):>5}  "
                  f"net={over.get(r, 0) - under.get(r, 0):>+5}")

    out = {
        "input": args.input,
        "n": len(rows),
        "tie_threshold": args.tie_threshold,
        "dead_threshold": args.dead_threshold,
        "buckets": summaries,
        "overall": overall,
        "routable_misses": len(misses),
        "routable_n": len(routable),
        "top_confusions": [
            {"oracle": w, "chosen": g, "count": c}
            for (w, g), c in Counter(
                (r["oracle_retriever"], r["chosen_retriever"]) for r in misses
            ).most_common(20)
        ],
    }
    if args.output_json:
        with open(args.output_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
