"""
Diagnose the routing-prompt late-interaction pipeline bottleneck.

Four questions:
  Q1. Are backbone embeddings discriminative across retrievers for the same query?
      → Pairwise cosine similarity statistics
  Q2. Do embedding distances actually reflect F1 differences?
      → Spearman(|F1_i - F1_j|, cosine_distance_ij) across all query-pair combinations
  Q3. Does the yes-logit alone already solve routing?
      → argmax(yes_logit) downstream F1 + Spearman(yes_logit, F1) per query
  Q4. What can a linear probe achieve vs the trained cross-attention head?
      → LogisticRegression on [N, 5×1024] trained on train+val, evaluated on test
      → LogisticRegression on [N, 5] yes-logits only (baseline)

Run:
  python -m analysis.representation_diagnostics \
      --cache_dir outputs/cache/single_vector \
      --head_f1 0.3518 \
      --output_json /path/to/diagnostics.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_split(cache_dir, split_name):
    path = os.path.join(cache_dir, f"{split_name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cache split: {path}")
    data = torch_load(path)
    print(f"  {split_name}: embeddings={tuple(data['embeddings'].shape)}, "
          f"yes_logits={tuple(data['yes_logits'].shape)}, n={data['labels'].shape[0]}")
    return data


def sep(title=""):
    bar = "=" * 72
    print(f"\n{bar}")
    if title:
        print(f"  {title}")
        print(bar)


# --------------------------------------------------------------------------- #
# Q1 — Pairwise cosine similarity
# --------------------------------------------------------------------------- #

def pairwise_cosine_stats(embeddings, action_metrics, split_name):
    """
    embeddings: [N, C, H]
    Returns a dict of stats and the raw (N, num_pairs) similarity tensor.
    """
    N, C, H = embeddings.shape
    emb_norm = F.normalize(embeddings.float(), dim=-1)
    sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))  # [N, C, C]
    triu = torch.triu_indices(C, C, offset=1)
    pair_sims = sim_matrix[:, triu[0], triu[1]]  # [N, num_pairs]

    best_idx = action_metrics.argmax(dim=-1)
    worst_idx = action_metrics.argmin(dim=-1)
    best_emb = emb_norm[torch.arange(N), best_idx]
    worst_emb = emb_norm[torch.arange(N), worst_idx]
    best_worst_sim = (best_emb * worst_emb).sum(dim=-1)

    # Per-retriever mean outward cosine distance from all others
    cos_dist_matrix = 1.0 - sim_matrix  # [N, C, C]
    mask = 1.0 - torch.eye(C)
    mean_outward = (cos_dist_matrix * mask.unsqueeze(0)).sum(dim=-1) / (C - 1)  # [N, C]
    per_retriever_mean_dist = mean_outward.mean(dim=0)  # [C]

    stats = {
        "mean_pairwise_sim": pair_sims.mean().item(),
        "median_pairwise_sim": pair_sims.median().item(),
        "std_pairwise_sim": pair_sims.std().item(),
        "min_pairwise_sim": pair_sims.min().item(),
        "max_pairwise_sim": pair_sims.max().item(),
        "pct_pairs_above_095": (pair_sims > 0.95).float().mean().item(),
        "pct_pairs_above_099": (pair_sims > 0.99).float().mean().item(),
        "best_worst_sim_mean": best_worst_sim.mean().item(),
        "best_worst_sim_std": best_worst_sim.std().item(),
        "per_retriever_mean_outward_dist": per_retriever_mean_dist.tolist(),
    }

    print(f"\n[{split_name}]  Pairwise cosine similarity (C(5,2)=10 pairs per query):")
    print(f"  mean={stats['mean_pairwise_sim']:.4f}  median={stats['median_pairwise_sim']:.4f}  "
          f"std={stats['std_pairwise_sim']:.4f}  min={stats['min_pairwise_sim']:.4f}  max={stats['max_pairwise_sim']:.4f}")
    print(f"  % pairs with sim > 0.95: {stats['pct_pairs_above_095']*100:.1f}%")
    print(f"  % pairs with sim > 0.99: {stats['pct_pairs_above_099']*100:.1f}%")
    print(f"  Best vs worst retriever cosine sim: {stats['best_worst_sim_mean']:.4f} ± {stats['best_worst_sim_std']:.4f}")
    print(f"  Per-retriever mean cosine distance from others: "
          + "  ".join(f"r{i}={d:.4f}" for i, d in enumerate(stats['per_retriever_mean_outward_dist'])))

    return stats, pair_sims, triu


# --------------------------------------------------------------------------- #
# Q2 — F1 gap vs cosine distance correlation
# --------------------------------------------------------------------------- #

def f1_gap_distance_correlation(embeddings, action_metrics, pair_sims, triu, split_name):
    N, C = action_metrics.shape
    f1_i = action_metrics[:, triu[0]]
    f1_j = action_metrics[:, triu[1]]
    f1_gaps = (f1_i - f1_j).abs()          # [N, num_pairs]
    cos_dist = 1.0 - pair_sims             # [N, num_pairs]

    gaps_flat = f1_gaps.numpy().ravel()
    dist_flat = cos_dist.numpy().ravel()

    # Overall Spearman
    mask_nonzero = gaps_flat > 0.0
    rho_all, pval_all = spearmanr(gaps_flat[mask_nonzero], dist_flat[mask_nonzero])

    # Bucket: does mean distance increase as F1 gap grows?
    buckets = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
    bucket_stats = {}
    for lo, hi in buckets:
        m = (gaps_flat >= lo) & (gaps_flat < hi)
        if m.sum() > 0:
            bucket_stats[f"gap_{lo:.2f}_{hi:.2f}"] = {
                "n_pairs": int(m.sum()),
                "mean_cos_dist": float(dist_flat[m].mean()),
            }

    stats = {
        "spearman_rho": rho_all,
        "spearman_pval": pval_all,
        "buckets": bucket_stats,
    }

    print(f"\n[{split_name}]  Spearman(F1 gap, cosine distance): rho={rho_all:.4f}  p={pval_all:.2e}")
    print(f"  Mean cosine distance by F1 gap bucket:")
    for key, v in bucket_stats.items():
        lo, hi = key.replace("gap_", "").split("_")
        print(f"    {lo} ≤ gap < {hi}: mean_dist={v['mean_cos_dist']:.4f}  (n={v['n_pairs']:,})")

    return stats


# --------------------------------------------------------------------------- #
# Q3 — Yes-logit routing
# --------------------------------------------------------------------------- #

def yes_logit_diagnostics(yes_logits, action_metrics, split_name):
    N, C = yes_logits.shape

    # Downstream F1 from argmax routing
    selected = yes_logits.argmax(dim=-1)
    downstream_f1 = action_metrics[torch.arange(N), selected].mean().item()

    # Per-query Spearman(yes_logit, F1)
    corrs = []
    for i in range(N):
        yl = yes_logits[i].numpy()
        f1 = action_metrics[i].numpy()
        if len(np.unique(f1)) < 2:
            continue
        r, _ = spearmanr(yl, f1)
        if not np.isnan(r):
            corrs.append(r)
    mean_spearman = float(np.mean(corrs))
    std_spearman = float(np.std(corrs))

    # Yes-logit spread: std across retrievers per query
    yl_std = yes_logits.float().std(dim=-1)

    stats = {
        "argmax_routing_downstream_f1": downstream_f1,
        "per_query_spearman_mean": mean_spearman,
        "per_query_spearman_std": std_spearman,
        "yes_logit_spread_mean": yl_std.mean().item(),
        "yes_logit_spread_std": yl_std.std().item(),
    }

    print(f"\n[{split_name}]  argmax(yes_logit) downstream F1: {downstream_f1:.4f}")
    print(f"  Per-query Spearman(yes_logit rank, F1 rank): {mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"  Yes-logit std across retrievers per query: {yl_std.mean():.4f} ± {yl_std.std():.4f}")

    return stats


# --------------------------------------------------------------------------- #
# Q4 — Linear probe vs trained head
# --------------------------------------------------------------------------- #

def linear_probe_embeddings(train_emb, train_labels, train_metrics,
                             test_emb, test_labels, test_metrics, seed=42):
    N_tr, C, H = train_emb.shape
    N_te = test_emb.shape[0]

    X_tr = train_emb.view(N_tr, C * H).float().numpy().astype(np.float32)
    X_te = test_emb.view(N_te, C * H).float().numpy().astype(np.float32)
    y_tr = train_labels.numpy().astype(int)
    y_te = test_labels.numpy().astype(int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    print("  Fitting logistic regression on [N, 5×1024]...")
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=0.1, solver="saga", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    pred_te = clf.predict(X_te)
    acc = (pred_te == y_te).mean()
    downstream_f1 = test_metrics[torch.arange(N_te), torch.tensor(pred_te)].mean().item()

    return {
        "probe_test_accuracy": float(acc),
        "probe_test_downstream_f1": float(downstream_f1),
    }


def linear_probe_yes_logits(train_yes, train_labels, train_metrics,
                             test_yes, test_labels, test_metrics, seed=42):
    X_tr = train_yes.float().numpy().astype(np.float32)
    X_te = test_yes.float().numpy().astype(np.float32)
    y_tr = train_labels.numpy().astype(int)
    y_te = test_labels.numpy().astype(int)
    N_te = len(y_te)

    print("  Fitting logistic regression on [N, 5] yes-logits...")
    clf = LogisticRegression(max_iter=2000, random_state=seed, C=0.1)
    clf.fit(X_tr, y_tr)

    pred_te = clf.predict(X_te)
    acc = (pred_te == y_te).mean()
    downstream_f1 = test_metrics[torch.arange(N_te), torch.tensor(pred_te)].mean().item()

    return {
        "yl_probe_test_accuracy": float(acc),
        "yl_probe_test_downstream_f1": float(downstream_f1),
    }


# --------------------------------------------------------------------------- #
# Oracle / baseline
# --------------------------------------------------------------------------- #

def baseline_stats(action_metrics, retriever_names=None):
    oracle_f1 = action_metrics.max(dim=-1).values.mean().item()
    per_retriever = action_metrics.mean(dim=0)
    best_fixed_f1 = per_retriever.max().item()
    best_fixed_idx = per_retriever.argmax().item()
    best_fixed_name = retriever_names[best_fixed_idx] if retriever_names else str(best_fixed_idx)
    return {
        "oracle_f1": oracle_f1,
        "best_fixed_f1": best_fixed_f1,
        "best_fixed_retriever": best_fixed_name,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--head_f1", type=float, default=None,
                        help="Downstream test F1 of the trained cross-attention head (for comparison table).")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    sep("Loading cache")
    train = load_split(args.cache_dir, "train")
    val = load_split(args.cache_dir, "val")
    test = load_split(args.cache_dir, "test")

    # Multi-vector caches ([N, C, K, H], K hidden states
    # per retriever) mean-pool over K to [N, C, H] so the rest of this script's
    # single-vector analysis (Q1-Q4) applies unchanged. This measures whether
    # the *pooled* per-retriever representation is less collapsed, not the
    # finer token-level geometry the late-interaction head actually attends
    # over -- a coarser but still informative proxy.
    for split in (train, val, test):
        if split["embeddings"].dim() == 4:
            split["embeddings"] = split["embeddings"].mean(dim=2)

    # Mirror the head's training setup: combine train+val
    train_emb = torch.cat([train["embeddings"], val["embeddings"]], dim=0)
    train_labels = torch.cat([train["labels"], val["labels"]], dim=0)
    train_metrics = torch.cat([train["action_metrics"], val["action_metrics"]], dim=0)
    train_yes = torch.cat([train["yes_logits"], val["yes_logits"]], dim=0)

    test_emb = test["embeddings"]
    test_labels = test["labels"]
    test_metrics = test["action_metrics"]
    test_yes = test["yes_logits"]

    results = {}

    # -------- Q1: Embedding similarity ------------------------------------ #
    sep("Q1 — Are embeddings discriminative? (pairwise cosine similarity)")
    tv_stats, tv_pair_sims, triu = pairwise_cosine_stats(train_emb, train_metrics, "train+val")
    te_stats, te_pair_sims, _ = pairwise_cosine_stats(test_emb, test_metrics, "test")
    results["embedding_similarity"] = {"train_val": tv_stats, "test": te_stats}

    # -------- Q2: F1 gap vs distance correlation -------------------------- #
    sep("Q2 — Do distances reflect F1 differences? (Spearman correlation)")
    tv_corr = f1_gap_distance_correlation(train_emb, train_metrics, tv_pair_sims, triu, "train+val")
    te_corr = f1_gap_distance_correlation(test_emb, test_metrics, te_pair_sims, triu, "test")
    results["f1_gap_distance_correlation"] = {"train_val": tv_corr, "test": te_corr}

    # -------- Q3: Yes-logit routing --------------------------------------- #
    sep("Q3 — Does yes-logit alone solve routing?")
    yl_stats = yes_logit_diagnostics(test_yes, test_metrics, "test")
    results["yes_logit"] = yl_stats

    # -------- Q4: Linear probes ------------------------------------------ #
    sep("Q4 — Linear probe vs cross-attention head")
    emb_probe = linear_probe_embeddings(train_emb, train_labels, train_metrics,
                                         test_emb, test_labels, test_metrics)
    yl_probe = linear_probe_yes_logits(train_yes, train_labels, train_metrics,
                                        test_yes, test_labels, test_metrics)
    results["linear_probe"] = {**emb_probe, **yl_probe}

    # -------- Baselines --------------------------------------------------- #
    bl = baseline_stats(test_metrics)
    results["baselines"] = bl

    # -------- Summary table ----------------------------------------------- #
    sep("Summary — Test downstream F1 by routing method")
    print(f"\n{'Method':<45} {'Test F1':>10}")
    print("-" * 57)
    print(f"{'Oracle (best retriever per query)':<45} {bl['oracle_f1']:>10.4f}")
    print(f"{'Best fixed retriever (' + bl['best_fixed_retriever'] + ')':<45} {bl['best_fixed_f1']:>10.4f}")
    print(f"{'argmax(yes_logit)':<45} {yl_stats['argmax_routing_downstream_f1']:>10.4f}")
    print(f"{'LogReg on yes-logits only (5-dim)':<45} {yl_probe['yl_probe_test_downstream_f1']:>10.4f}")
    print(f"{'LogReg probe on embeddings (5×1024)':<45} {emb_probe['probe_test_downstream_f1']:>10.4f}")
    if args.head_f1 is not None:
        print(f"{'Trained LateInterHead (cross-attn)':<45} {args.head_f1:>10.4f}")
    print()
    print(f"Key diagnostics:")
    print(f"  Mean pairwise cosine sim (test):       {te_stats['mean_pairwise_sim']:.4f}")
    print(f"  Best vs worst retriever cos sim (test): {te_stats['best_worst_sim_mean']:.4f}")
    print(f"  Spearman(F1 gap, cos distance) (test): {te_corr['spearman_rho']:.4f}  p={te_corr['spearman_pval']:.2e}")
    print(f"  Per-query Spearman(yes_logit, F1):      {yl_stats['per_query_spearman_mean']:.4f}")

    if args.output_json:
        # Convert non-serialisable types
        def to_serialisable(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: to_serialisable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_serialisable(x) for x in obj]
            return obj

        with open(args.output_json, "w") as f:
            json.dump(to_serialisable(results), f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
