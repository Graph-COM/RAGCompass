"""
Does the trained router still help when a DIFFERENT LLM reads the evidence?

The router's per-query choice is a function of the evidence, not of the reader,
so nothing is retrained: `relabel_with_reader.py` re-answers the same cached
evidence with a second reader, and this script re-scores the router's ALREADY
MADE choices under the new reader's utilities.

Reports, under each reader: the routed F1, that reader's best fixed retriever
(whose identity may change), and the oracle -- so "the router still beats the
best fixed choice" is checked against the new reader's own baseline, not the
old one.

Usage (CPU):
    python -m analysis.second_reader_eval \
        --decisions outputs/diagnostics/routing_decisions_s42.jsonl \
        --reader_dir Qwen2.5-7B-Instruct_Qwen3-Embedding-0.6B \
        --output_json outputs/diagnostics/second_reader_eval_s42.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]
EXPECTED_BEST_FIXED_F1 = 0.2781745126131807
EXPECTED_ORACLE_F1 = 0.750769178988225
GATE_TOLERANCE = 5e-4


def item_key(item, idx):
    for key in ("id", "_id", "qid", "question_id"):
        if key in item:
            return str(item[key])
    return f"row_{idx}"


def load_f1(base_dir, datasets, retrievers):
    """(dataset, retriever) -> {id: f1}"""
    out = defaultdict(dict)
    missing = []
    for dataset in datasets:
        for retriever in retrievers:
            path = os.path.join(base_dir, dataset, f"{retriever}.jsonl")
            if not os.path.exists(path):
                missing.append(path)
                continue
            with open(path) as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if item.get("f1") is not None:
                        out[(dataset, retriever)][item_key(item, idx)] = float(item["f1"])
    return out, missing


def summarize(rows, retrievers, label):
    """rows: list of (chosen_retriever, {retriever: f1})"""
    routed = np.array([f1s[chosen] for chosen, f1s in rows])
    per_retriever = {
        r: np.array([f1s[r] for _, f1s in rows]) for r in retrievers
    }
    best_name = max(per_retriever, key=lambda r: per_retriever[r].mean())
    best = float(per_retriever[best_name].mean())
    oracle = float(np.mean([max(f1s.values()) for _, f1s in rows]))
    result = {
        "reader": label,
        "n": len(rows),
        "routed_f1": float(routed.mean()),
        "best_fixed_f1": best,
        "best_fixed_retriever": best_name,
        "oracle_f1": oracle,
        "improvement_vs_best_fixed": float(routed.mean()) - best,
        "per_retriever_f1": {r: float(v.mean()) for r, v in per_retriever.items()},
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fixed router under a second reader")
    parser.add_argument("--decisions", type=str, required=True)
    parser.add_argument("--original_dir", type=str,
                        default="Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B")
    parser.add_argument("--reader_dir", type=str, required=True)
    parser.add_argument("--original_label", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--reader_label", type=str, default="Qwen2.5-7B-Instruct")
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    decisions = []
    with open(args.decisions) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    datasets = sorted({d["dataset"] for d in decisions})
    print(f"{len(decisions)} routing decisions across {datasets}")

    new_f1, missing = load_f1(args.reader_dir, datasets, retrievers)
    if missing:
        print(f"NOTE: {len(missing)} result file(s) absent under {args.reader_dir}:")
        for path in missing[:6]:
            print(f"  {path}")

    # Only queries the second reader has answered for ALL retrievers are usable:
    # a partial row would change the best-fixed and oracle baselines rather than
    # just adding noise.
    old_rows, new_rows, incomplete = [], [], 0
    for row in decisions:
        ds, qid, chosen = row["dataset"], str(row["id"]), row["chosen_retriever"]
        f1s_new = {r: new_f1.get((ds, r), {}).get(qid) for r in retrievers}
        if any(v is None for v in f1s_new.values()):
            incomplete += 1
            continue
        new_rows.append((chosen, f1s_new))
        old_rows.append((chosen, {r: row["all_f1s"][r] for r in retrievers}))

    if incomplete:
        print(f"{incomplete}/{len(decisions)} queries not yet complete under the second reader "
              f"-- excluded from BOTH readers so the comparison stays paired.")
    if not new_rows:
        raise SystemExit("No queries are complete under the second reader yet.")

    original = summarize(old_rows, retrievers, args.original_label)
    second = summarize(new_rows, retrievers, args.reader_label)

    # Gate only when the full set is present: on a subset the published numbers
    # legitimately differ, so a mismatch would not indicate an error.
    gate = None
    if not incomplete:
        gate = (abs(original["best_fixed_f1"] - EXPECTED_BEST_FIXED_F1) < GATE_TOLERANCE
                and abs(original["oracle_f1"] - EXPECTED_ORACLE_F1) < GATE_TOLERANCE)
        print(f"Gate (original reader reproduces published numbers): {'PASS' if gate else 'FAIL'}")

    print(f"\n{'':<26}{original['reader']:>22}{second['reader']:>22}")
    print(f"{'routed (ours)':<26}{original['routed_f1']:>22.4f}{second['routed_f1']:>22.4f}")
    print(f"{'best fixed':<26}{original['best_fixed_f1']:>22.4f}{second['best_fixed_f1']:>22.4f}")
    print(f"{'  (which retriever)':<26}{original['best_fixed_retriever']:>22}{second['best_fixed_retriever']:>22}")
    print(f"{'oracle':<26}{original['oracle_f1']:>22.4f}{second['oracle_f1']:>22.4f}")
    print(f"{'gain vs best fixed':<26}{original['improvement_vs_best_fixed']:>+22.4f}"
          f"{second['improvement_vs_best_fixed']:>+22.4f}")

    results = {
        "n_paired": len(new_rows),
        "n_incomplete": incomplete,
        "gate_passed": gate,
        "original": original,
        "second": second,
        "best_fixed_identity_changed": (
            original["best_fixed_retriever"] != second["best_fixed_retriever"]
        ),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
