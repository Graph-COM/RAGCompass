"""
Query-only routing baseline in a form that survives an unseen retriever.

WHY THIS EXISTS, given ltrr.py already exists
------------------------------------------------------
`ltrr.py` fits a 5-way classifier over which retriever to pick. That
formulation cannot express the leave-one-retriever-out protocol at all: a class
absent from `fit` can never be returned by `predict`, so a held-out mechanism is
unreachable by construction rather than as a finding. Reporting "the baseline
never picks the unseen retriever" from that model would be a statement about
sklearn's label space, not about query-only routing.

So this file re-poses the same baseline as a SHARED PER-RETRIEVER SCORER, which
is the strongest honest form of it and the one that makes the comparison fair:

    one row per (query, retriever)
    features = query features  +  THAT retriever's own confidence statistics
    one model, shared across retrievers, fitted over all rows
    at test: score all R candidates, take the argmax

Nothing in the model is tied to a retriever's identity, so a mechanism held out
of training is still scoreable -- exactly the open-pool property our own scalar
scorer has. That parity is the point: if the query-only router still degrades on
an unseen mechanism, it degrades for a reason about *what it reads*, not because
its output layer had no slot for the answer.

The paper's analytical claim is the hypothesis under test here. Confidence
scores are internal to each mechanism and calibrated on the corpus it was fitted
to, so a newly added mechanism's scores share no scale with the ones the scorer
was fitted on, and a shared scorer over raw confidence values should transfer
poorly. This measures that instead of asserting it.

DELIBERATE FEATURE CHOICE
-------------------------
The cross-retriever pool features in `ltrr.py` (mean pairwise top-1
match, mean pairwise top-3 Jaccard) are dropped here, for two reasons. They are
constant across retrievers within a query, so they cannot affect an argmax over
retrievers -- they carry no discriminative signal in this formulation. And
computing them over the full 5-retriever pool would leak the held-out mechanism
into training rows, quietly breaking the protocol. Everything else -- the query
features and each retriever's own confidence block -- is carried over unchanged.

Usage:
    python -m baselines.ltrr_shared_scorer \
        --retrievers rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step \
        --holdout_action hipporag2-0.5_1_step --seed 42 \
        --output_json <path>
"""

import argparse
import json

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from router.train_e2e import resplit_split_data
from router.train_head import (
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from baselines.ltrr import (
    QUESTION_TYPE_VOCAB,
    build_raw_score_lookup,
    downstream_metrics,
    extract_query_features,
    filter_split_by_dataset,
)
from router.shared_utils import DEFAULT_RESULTS_BASE_DIR

# The filtered example set's reference values, used as a gate when no holdout is
# in play. Same constants dump_decisions.py checks.
REF_BEST_FIXED = 0.2781745126131807
REF_ORACLE = 0.7507691
GATE_TOL = 1e-3


def own_score_features(item):
    """The confidence block for ONE retriever, under retriever-agnostic names.

    Identical statistics to ltrr.retriever_score_features, minus the
    `{retriever}__` prefix -- that prefix is precisely what ties the feature
    space to a fixed pool and has to go for a shared scorer.
    """
    scores = item.get("top_k_scores", []) or [0.0]
    return {
        "own_top1_score": float(scores[0]),
        "own_top3_mean": float(np.mean(scores[:3])) if scores else 0.0,
        "own_top5_mean": float(np.mean(scores[:5])) if scores else 0.0,
        "own_score_std": float(np.std(scores)) if len(scores) > 1 else 0.0,
        "own_top1_gap": float(scores[0] - scores[1]) if len(scores) > 1 else 0.0,
    }


def build_pairwise_matrix(split_data, raw_lookup, retrievers, use_identity=False):
    """Explode a split into one row per (query, retriever).

    Returns X, feature_names, and index/label arrays aligned to X's rows:
      query_idx[i]     -- which query row i came from
      retriever_idx[i] -- which retriever row i scores
      y_binary[i]      -- 1.0 if that retriever is the query's oracle choice
      y_util[i]        -- that retriever's downstream utility (the F1 label)

    With use_identity, a one-hot over retrievers is appended. That restores the
    per-retriever prior a 5-way classifier gets for free ("SPLADE is usually
    best") -- and reintroduces exactly the failure our own type-embedding head
    shows on an unseen mechanism, because the held-out retriever's column is
    never active during training.
    """
    f1s = np.asarray(split_data["f1_scores"])  # [n_queries, n_retrievers]
    n_queries = len(split_data["queries"])
    oracle_idx = f1s.argmax(axis=1)

    rows, query_idx, retriever_idx, y_binary, y_util = [], [], [], [], []
    for qi in range(n_queries):
        rid = split_data["ids"][qi]
        qfeats = extract_query_features(split_data["queries"][qi])
        qtype = qfeats.pop("question_type")
        by_retriever = raw_lookup.get(rid, {})
        for ri, r in enumerate(retrievers):
            feats = dict(qfeats)
            feats.update(own_score_features(by_retriever.get(r, {})))
            feats["__qtype"] = qtype
            rows.append(feats)
            query_idx.append(qi)
            retriever_idx.append(ri)
            y_binary.append(1.0 if ri == oracle_idx[qi] else 0.0)
            y_util.append(float(f1s[qi, ri]))

    qtypes = [row.pop("__qtype") for row in rows]
    numeric_keys = sorted(rows[0].keys())
    X_numeric = np.array([[row[k] for k in numeric_keys] for row in rows], dtype=np.float64)
    X_qtype = np.zeros((len(rows), len(QUESTION_TYPE_VOCAB)), dtype=np.float64)
    for i, qt in enumerate(qtypes):
        X_qtype[i, QUESTION_TYPE_VOCAB.index(qt)] = 1.0
    X = np.concatenate([X_numeric, X_qtype], axis=1)
    feature_names = numeric_keys + [f"qtype_{q}" for q in QUESTION_TYPE_VOCAB]
    if use_identity:
        X_ident = np.zeros((len(rows), len(retrievers)), dtype=np.float64)
        X_ident[np.arange(len(rows)), np.asarray(retriever_idx)] = 1.0
        X = np.concatenate([X, X_ident], axis=1)
        feature_names = feature_names + [f"is_{r}" for r in retrievers]
    return (
        X,
        feature_names,
        np.asarray(query_idx),
        np.asarray(retriever_idx),
        np.asarray(y_binary),
        np.asarray(y_util),
    )


def argmax_predictions(scores, query_idx, retriever_idx, n_queries, n_retrievers):
    """Fold per-(query, retriever) scores back into one choice per query."""
    grid = np.full((n_queries, n_retrievers), -np.inf, dtype=np.float64)
    grid[query_idx, retriever_idx] = scores
    return grid.argmax(axis=1)


def main():
    ap = argparse.ArgumentParser(description="Shared per-retriever query-only scorer")
    ap.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    ap.add_argument("--dataset", type=str, default=None)
    ap.add_argument("--retrievers", type=str,
                    default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step")
    ap.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    ap.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    ap.add_argument("--threshold", type=float, default=0.05)
    ap.add_argument("--top_k_docs", type=int, default=5)
    ap.add_argument("--resplit_train_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout_dataset", type=str, default=None,
                    help="Leave-one-dataset-out, as in ltrr.py.")
    ap.add_argument("--holdout_action", type=str, default=None,
                    help="Leave-one-RETRIEVER-out: drop this retriever's rows from training. "
                         "Test always scores the full pool, so the model must rank a "
                         "mechanism it never trained on. Mirrors "
                         "train_head.py --holdout_action.")
    ap.add_argument("--use_retriever_identity", action="store_true",
                    help="Append a one-hot retriever ID to the features. Recovers the "
                         "per-retriever prior a 5-way classifier gets for free, at the "
                         "cost of open-pool support: under --holdout_action the held-out "
                         "retriever's column is never active in training, so its score is "
                         "whatever the unfitted weight implies. This is the query-only "
                         "analogue of the type-embedding head's degeneracy.")
    ap.add_argument("--output_json", type=str, required=True)
    args = ap.parse_args()

    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    assert action_names == retrievers, (
        f"discover_actions order/set mismatch: {action_names} vs {retrievers}"
    )
    if args.holdout_action and args.holdout_action not in retrievers:
        raise ValueError(f"--holdout_action {args.holdout_action!r} not among {retrievers}")

    print(f"Datasets:   {dataset_names}")
    print(f"Retrievers: {retrievers}")

    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=args.threshold,
        max_samples=None,
        base_dir=args.results_base_dir,
        label_metric=args.label_metric,
        top_k_docs=args.top_k_docs,
    )

    # Dataset holdout is applied BEFORE the resplit, so held-out rows cannot
    # re-enter training through the train/val merge. Same ordering as
    # ltrr.py and train_head.py.
    if args.holdout_dataset:
        if args.holdout_dataset not in dataset_names:
            raise ValueError(f"--holdout_dataset {args.holdout_dataset!r} not in {dataset_names}")
        splits = dict(splits)
        splits["train"] = filter_split_by_dataset(splits["train"], args.holdout_dataset, keep=False)
        splits["val"] = filter_split_by_dataset(splits["val"], args.holdout_dataset, keep=False)
        splits["test"] = filter_split_by_dataset(splits["test"], args.holdout_dataset, keep=True)
        print(f"Leave-one-dataset-out: holding out {args.holdout_dataset}")

    train_data, val_data = resplit_split_data(
        splits["train"], splits["val"],
        train_ratio=args.resplit_train_ratio, seed=args.seed,
    )
    test_data = splits["test"]
    print(f"Train {len(train_data['labels'])}  Val {len(val_data['labels'])}  "
          f"Test {len(test_data['labels'])}")

    raw_lookup = build_raw_score_lookup(args.results_base_dir, dataset_names, retrievers)

    ident = args.use_retriever_identity
    Xtr, feature_names, qi_tr, ri_tr, yb_tr, yu_tr = build_pairwise_matrix(
        train_data, raw_lookup, retrievers, use_identity=ident)
    Xva, _, qi_va, ri_va, yb_va, yu_va = build_pairwise_matrix(
        val_data, raw_lookup, retrievers, use_identity=ident)
    Xte, _, qi_te, ri_te, _, _ = build_pairwise_matrix(
        test_data, raw_lookup, retrievers, use_identity=ident)

    # Train and val are fitted together for the test-time model, matching
    # ltrr.py so the two formulations are compared like for like.
    X_fit = np.concatenate([Xtr, Xva], axis=0)
    ri_fit = np.concatenate([ri_tr, ri_va], axis=0)
    yb_fit = np.concatenate([yb_tr, yb_va], axis=0)
    yu_fit = np.concatenate([yu_tr, yu_va], axis=0)

    # Leave-one-retriever-out acts here: the held-out mechanism's ROWS leave
    # training, while the test matrix keeps all R retrievers. The point is that
    # the scorer must rank a mechanism it never saw, not that it never meets one.
    if args.holdout_action:
        hold_idx = retrievers.index(args.holdout_action)
        keep = ri_fit != hold_idx
        dropped = int((~keep).sum())
        X_fit, yb_fit, yu_fit, ri_fit = X_fit[keep], yb_fit[keep], yu_fit[keep], ri_fit[keep]
        print(f"Leave-one-retriever-out: holding out {args.holdout_action} "
              f"({dropped} training rows dropped, test keeps all {len(retrievers)})")

    # Train-only fit set (same retriever holdout) for picking the model family
    # on VALIDATION; the chosen family's reported test number still comes from
    # the train+val fit above. Picking by test improvement, as before, made the
    # reported number optimistic.
    Xtr_sel, yb_tr_sel, yu_tr_sel = Xtr, yb_tr, yu_tr
    if args.holdout_action:
        keep_tr = ri_tr != retrievers.index(args.holdout_action)
        Xtr_sel, yb_tr_sel, yu_tr_sel = Xtr[keep_tr], yb_tr[keep_tr], yu_tr[keep_tr]
    f1_val = np.asarray(val_data["f1_scores"])
    n_val_q = f1_val.shape[0]
    sel_scaler = StandardScaler().fit(Xtr_sel)

    f1_test = np.asarray(test_data["f1_scores"])
    n_test_q, n_r = f1_test.shape

    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_te_s = scaler.transform(Xte)

    # Two ways to turn query-only features into a ranking, both reported: fit
    # "is this the oracle choice" as a classifier and use P(yes) as the score,
    # or regress the retriever's utility directly and score with the prediction.
    models = {
        "logreg_isbest": (LogisticRegression(max_iter=2000, class_weight="balanced"),
                          "clf", True),
        "rf_isbest": (RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                             random_state=args.seed), "clf", False),
        "gb_isbest": (GradientBoostingClassifier(random_state=args.seed), "clf", False),
        "ridge_util": (Ridge(alpha=1.0, random_state=args.seed), "reg", True),
        "rf_util": (RandomForestRegressor(n_estimators=500, random_state=args.seed),
                    "reg", False),
        "gb_util": (GradientBoostingRegressor(random_state=args.seed), "reg", False),
    }

    results = {
        "formulation": "shared_per_retriever_scorer",
        "use_retriever_identity": bool(args.use_retriever_identity),
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "holdout_dataset": args.holdout_dataset,
        "holdout_action": args.holdout_action,
        "seed": args.seed,
        "n_train_rows": int(len(yb_fit)),
        "n_test_queries": int(n_test_q),
        "retrievers": retrievers,
        "models": {},
    }

    for name, (model, kind, needs_scaling) in models.items():
        print(f"\nFitting {name} ({kind})...")
        Xf = X_fit_s if needs_scaling else X_fit
        Xp = X_te_s if needs_scaling else Xte
        target = yb_fit if kind == "clf" else yu_fit
        model.fit(Xf, target)
        if kind == "clf":
            scores = model.predict_proba(Xp)[:, 1]
        else:
            scores = model.predict(Xp)
        preds = argmax_predictions(scores, qi_te, ri_te, n_test_q, n_r)
        metrics = downstream_metrics(preds, f1_test)

        pick_counts = np.bincount(preds, minlength=n_r)
        metrics["pick_rate_by_retriever"] = {
            r: float(pick_counts[i] / n_test_q) for i, r in enumerate(retrievers)
        }
        if args.holdout_action:
            hi = retrievers.index(args.holdout_action)
            oracle_is_held = f1_test.argmax(axis=1) == hi
            metrics["picks_held_out_retriever"] = float(pick_counts[hi] / n_test_q)
            metrics["picks_held_out_when_oracle"] = (
                float((preds[oracle_is_held] == hi).mean()) if oracle_is_held.any() else 0.0
            )
        sel_model = clone(model)
        Xs_fit = sel_scaler.transform(Xtr_sel) if needs_scaling else Xtr_sel
        Xs_val = sel_scaler.transform(Xva) if needs_scaling else Xva
        sel_model.fit(Xs_fit, yb_tr_sel if kind == "clf" else yu_tr_sel)
        val_scores = sel_model.predict_proba(Xs_val)[:, 1] if kind == "clf" else sel_model.predict(Xs_val)
        val_preds = argmax_predictions(val_scores, qi_va, ri_va, n_val_q, n_r)
        metrics["val_selected_f1"] = downstream_metrics(val_preds, f1_val)["selected_f1"]

        results["models"][name] = metrics
        print(f"  val_selected_f1={metrics['val_selected_f1']:.4f}  selected_f1={metrics['selected_f1']:.4f}  "
              f"best_fixed={metrics['best_fixed_f1']:.4f}  "
              f"improvement={metrics['improvement_vs_best_fixed']:+.4f}")
        if args.holdout_action:
            print(f"  picks {args.holdout_action}: "
                  f"{metrics['picks_held_out_retriever']:.1%} overall, "
                  f"{metrics['picks_held_out_when_oracle']:.1%} when it is the oracle")

    results["selection"] = "val_selected_f1"
    best_name = max(results["models"],
                    key=lambda k: results["models"][k]["val_selected_f1"])
    results["best_model"] = best_name
    results["best_model_metrics"] = results["models"][best_name]

    # Gate: with no holdout the example set must be the reported one. This is the
    # same check ltrr.py relies on, and it is what makes the shared
    # scorer's number comparable to the main results table's rather than a different population.
    bf = results["best_model_metrics"]["best_fixed_f1"]
    orc = results["best_model_metrics"]["oracle_f1"]
    if not args.holdout_dataset:
        ok = abs(bf - REF_BEST_FIXED) < GATE_TOL and abs(orc - REF_ORACLE) < GATE_TOL
        results["example_set_gate_passed"] = bool(ok)
        print(f"\nExample-set gate: best_fixed={bf:.4f} (ref {REF_BEST_FIXED:.4f}), "
              f"oracle={orc:.4f} (ref {REF_ORACLE:.4f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(
                "Example-set gate FAILED: this run is not on the reported test set, "
                "so its number is not comparable to the main results table. Refusing to write output."
            )
    else:
        results["example_set_gate_passed"] = None
        print(f"\nHoldout fold: best_fixed={bf:.4f}, oracle={orc:.4f} "
              f"(dataset-specific; the pooled gate does not apply)")

    print(f"\nBest: {best_name}  "
          f"improvement={results['best_model_metrics']['improvement_vs_best_fixed']:+.4f}")

    with open(args.output_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved -> {args.output_json}")


if __name__ == "__main__":
    main()
