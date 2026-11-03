"""Test F1 of the methods whose per-query outputs are stored in the result JSONLs
(best fixed retriever, RRF, MoR, oracle), with 95% bootstrap confidence
intervals over test queries, on the routable test population of the main
results table (top-two utility gap >= 0.05 over the five retrievers).

Usage: python -m analysis.fixed_and_fusion_scores [RESULTS_DIR]
       (default: Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B)
Expected with seed 0 and 2000 resamples: n=3735; best fixed 0.2782, oracle
0.7508, RRF 0.3002, MoR 0.3203.
"""
import json
import os
import random
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
RETRIEVERS = ["rag_1_step", "bm25_1_step", "splade_v3_1_step", "colbertv2_1_step", "hipporag2-0.5_1_step"]
DATASETS = ["nq", "musique", "hotpotqa", "2wikimultihopqa"]
N_BOOT, SEED, GAP = 2000, 0, 0.05


def rows(path):
    with open(path) as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


per_retriever = [[] for _ in RETRIEVERS]
scores = {"oracle": [], "rrf": [], "mor": []}
for ds in DATASETS:
    res = [rows(os.path.join(BASE, ds, f"{r}.jsonl")) for r in RETRIEVERS]
    rrf = rows(os.path.join(BASE, ds, "rrf_1_step.jsonl"))
    mor = rows(os.path.join(BASE, ds, "mor_1_step.jsonl"))
    for qid in res[0]:
        f1 = [res[i][qid]["f1"] for i in range(len(RETRIEVERS))]
        top = sorted(f1, reverse=True)
        if top[0] - top[1] < GAP:
            continue
        for i, v in enumerate(f1):
            per_retriever[i].append(v)
        scores["oracle"].append(top[0])
        scores["rrf"].append(rrf[qid]["f1"])
        scores["mor"].append(mor[qid]["f1"])

means = [sum(v) / len(v) for v in per_retriever]
best = max(range(len(RETRIEVERS)), key=means.__getitem__)
scores = {"best_fixed": per_retriever[best], **scores}
n = len(scores["oracle"])
print(f"n={n}  best fixed retriever={RETRIEVERS[best]}")

rng = random.Random(SEED)
for name, v in scores.items():
    boots = sorted(sum(v[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT)]
    print(f"{name:10s} {sum(v) / n:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  half-width {(hi - lo) / 2:.4f}")
