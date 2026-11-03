"""
Emit per-query routing decisions for the comparison head trained on frozen
backbone representations, plus the per-dataset breakdown.

Every downstream analysis -- retrieval metrics, majority vote, the
LLM-as-router comparison, the oracle-gap decomposition, the second reader and
the retrieval-depth evaluation -- joins against the per-query decisions file
this script writes.

Usage:
    python -m analysis.dump_decisions \
        --cache_dir  outputs/cache/main \
        --head_dir   <run dir containing hybrid_model.bin> \
        --output     outputs/diagnostics/routing_decisions_s42.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router.train_head import (  # noqa: E402
    CachedActionEmbeddingDataset,
    LateInteractionHeadModel,
    load_embedding_cache_metadata,
)

# The paper's filtered test set; used as a gate, not as a filter.
EXPECTED_BEST_FIXED = 0.2781745126131807
EXPECTED_ORACLE = 0.750769178988225
TOL = 5e-4


def find_head_dir(root):
    """A run dir may store the head at its root or under checkpoints/best."""
    for cand in (root, os.path.join(root, "checkpoints", "best"),
                 os.path.join(root, "best"), os.path.join(root, "checkpoints")):
        if os.path.exists(os.path.join(cand, "hybrid_model.bin")):
            return cand
    raise FileNotFoundError(
        f"no hybrid_model.bin under {root} (tried root, checkpoints/best, best, checkpoints)")


def main():
    ap = argparse.ArgumentParser(description="Per-query decisions for a frozen head")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--head_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--skip_gate", action="store_true")
    args = ap.parse_args()

    meta = load_embedding_cache_metadata(args.cache_dir)
    actions = meta["action_names"]
    ds = CachedActionEmbeddingDataset(os.path.join(args.cache_dir, "test.pt"),
                                      threshold=args.threshold)
    print(f"test rows after utility-gap filter: {len(ds)} | retrievers: {actions}")

    head_dir = find_head_dir(args.head_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = LateInteractionHeadModel.load_pretrained(head_dir, actions, device=device)
    head.eval()
    print(f"head: {head_dir} (device {device})")

    emb = ds.data["embeddings"]
    preds = []
    with torch.no_grad():
        for i in range(0, len(ds), args.batch_size):
            logits = head(emb[i:i + args.batch_size].to(device))
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)

    f1s = ds.data["action_f1s"].float().numpy()
    ids = ds.data["ids"]
    dsets = ds.data["dataset_names"]

    rows = np.arange(len(preds))
    routed = float(f1s[rows, preds].mean())
    oracle = float(f1s.max(axis=1).mean())
    per_ret = {a: float(f1s[:, i].mean()) for i, a in enumerate(actions)}
    best_name = max(per_ret, key=per_ret.get)
    best_fixed = per_ret[best_name]

    print(f"\nrouted {routed:.4f} | best fixed {best_fixed:.4f} ({best_name}) | oracle {oracle:.4f}"
          f" | gain {routed - best_fixed:+.4f}")

    ok = abs(best_fixed - EXPECTED_BEST_FIXED) < TOL and abs(oracle - EXPECTED_ORACLE) < TOL
    print(f"gate (same example set as the main results table): {'PASS' if ok else 'FAIL'}")
    if not ok and not args.skip_gate:
        raise SystemExit("Gate failed -- this is not the reported example set; refusing to write.")

    best_idx = actions.index(best_name)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for i in range(len(preds)):
            oi = int(np.argmax(f1s[i]))
            f.write(json.dumps({
                "id": ids[i],
                "dataset": dsets[i],
                "chosen_retriever": actions[int(preds[i])],
                "oracle_retriever": actions[oi],
                "best_fixed_retriever": best_name,
                "chosen_f1": float(f1s[i, preds[i]]),
                "oracle_f1": float(f1s[i, oi]),
                "best_fixed_f1": float(f1s[i, best_idx]),
                "all_f1s": {actions[j]: float(f1s[i, j]) for j in range(f1s.shape[1])},
                "matches_oracle": bool(preds[i] == oi),
                "oracle_gain": float(f1s[i, oi] - f1s[i, preds[i]]),
            }) + "\n")
    print(f"wrote {len(preds)} rows -> {args.output}")

    # Per-dataset breakdown falls straight out of the same arrays.
    print("\nper-dataset:")
    for d in sorted(set(dsets)):
        m = np.array([x == d for x in dsets])
        bf = max(f1s[m][:, i].mean() for i in range(f1s.shape[1]))
        print(f"  {d:<18} n={m.sum():<5} routed {f1s[m][np.arange(m.sum()), preds[m]].mean():.4f}"
              f"  best-fixed {bf:.4f}  oracle {f1s[m].max(axis=1).mean():.4f}"
              f"  gain {f1s[m][np.arange(m.sum()), preds[m]].mean() - bf:+.4f}")


if __name__ == "__main__":
    main()
