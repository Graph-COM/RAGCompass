"""
LTRR-style learned-ranker baseline.

Prior retriever-routing work (LTRR) trains a feature-based learned ranker
(gradient-boosted / logistic-regression over query and retrieval-score
features). This trains such a ranker (sklearn's GradientBoostingClassifier/
RandomForest/LogisticRegression + lightweight regex query features) as a 5-way
classifier over which retriever to route to, using ONLY query-side and raw
per-retriever retrieval-confidence features (never the downstream f1/ndcg/
recall labels used to supervise it -- that would leak the answer).

Reuses load_action_splits + resplit_split_data so the filtered example set and
train/val/test partition are IDENTICAL to the router's (same
threshold=0.05 f1gap filter, same deterministic test split) -- best_fixed_f1
and oracle_f1 computed here should reproduce 0.2782 / 0.7508 as a sanity check
that this is a fair, apples-to-apples comparison.

Usage:
    python -m baselines.ltrr \
        --datasets nq,musique,hotpotqa,2wikimultihopqa \
        --retrievers rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step \
        --output_json <path>
"""

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from router.train_e2e import resplit_split_data
from router.train_head import (
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from router.shared_utils import (
    DEFAULT_RESULTS_BASE_DIR,
    item_id,
    load_results_file,
    split_specs,
)

QUESTION_WORDS = ["who", "what", "when", "where", "why", "how", "which", "whom", "whose"]
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
}


def extract_query_features(question):
    features = {}
    words = question.split()
    features["char_count"] = len(question)
    features["word_count"] = len(words)
    q_lower = question.lower().strip()
    features["question_type"] = "other"
    for qw in QUESTION_WORDS:
        if q_lower.startswith(qw):
            features["question_type"] = qw
            break
    features["has_numerics"] = float(bool(re.search(r"\d+", question)))
    features["avg_word_length"] = (
        sum(len(w) for w in words) / len(words) if words else 0.0
    )
    words_lower = [w.lower() for w in words]
    features["stopword_ratio"] = (
        sum(1 for w in words_lower if w in STOPWORDS) / len(words_lower) if words_lower else 0.0
    )
    return features


def build_raw_score_lookup(base_dir, dataset_names, retrievers):
    """id -> {retriever: item} across every split-dir of every dataset, so a
    row's raw top_k_scores can be found regardless of which resplit bucket
    (train/val) it ended up in after resplit_split_data's merge+reshuffle."""
    lookup = {}
    for dataset_name in dataset_names:
        for _, split_dir, _ in split_specs(dataset_name):
            for retriever in retrievers:
                path = os.path.join(base_dir, split_dir, f"{retriever}.jsonl")
                rows = load_results_file(path)
                for idx, item in enumerate(rows):
                    rid = item_id(item, idx)
                    lookup.setdefault(rid, {})[retriever] = item
    return lookup


def retriever_score_features(items_by_retriever, retrievers):
    features = {}
    top1_docs = {}
    top3_doc_sets = {}
    for r in retrievers:
        item = items_by_retriever.get(r, {})
        scores = item.get("top_k_scores", []) or [0.0]
        indices = item.get("top_k_indices", []) or []
        features[f"{r}__top1_score"] = float(scores[0])
        features[f"{r}__top3_mean"] = float(np.mean(scores[:3])) if scores else 0.0
        features[f"{r}__top5_mean"] = float(np.mean(scores[:5])) if scores else 0.0
        features[f"{r}__score_std"] = float(np.std(scores)) if len(scores) > 1 else 0.0
        features[f"{r}__top1_gap"] = float(scores[0] - scores[1]) if len(scores) > 1 else 0.0
        top1_docs[r] = indices[0] if indices else None
        top3_doc_sets[r] = set(indices[:3])

    pair_top1_matches, pair_top3_jaccards = [], []
    for i in range(len(retrievers)):
        for j in range(i + 1, len(retrievers)):
            r1, r2 = retrievers[i], retrievers[j]
            pair_top1_matches.append(float(top1_docs[r1] == top1_docs[r2] and top1_docs[r1] is not None))
            union = top3_doc_sets[r1] | top3_doc_sets[r2]
            inter = top3_doc_sets[r1] & top3_doc_sets[r2]
            pair_top3_jaccards.append(len(inter) / len(union) if union else 0.0)

    features["mean_pairwise_top1_match"] = float(np.mean(pair_top1_matches))
    features["mean_pairwise_top3_jaccard"] = float(np.mean(pair_top3_jaccards))
    return features


QUESTION_TYPE_VOCAB = QUESTION_WORDS + ["other"]


def build_feature_matrix(split_data, raw_lookup, retrievers):
    rows = []
    for idx, question in enumerate(split_data["queries"]):
        rid = split_data["ids"][idx]
        feats = extract_query_features(question)
        feats.update(retriever_score_features(raw_lookup.get(rid, {}), retrievers))
        rows.append(feats)

    question_types = [row.pop("question_type") for row in rows]
    numeric_keys = sorted(rows[0].keys())
    X_numeric = np.array([[row[k] for k in numeric_keys] for row in rows], dtype=np.float64)
    X_qtype = np.zeros((len(rows), len(QUESTION_TYPE_VOCAB)), dtype=np.float64)
    for i, qt in enumerate(question_types):
        X_qtype[i, QUESTION_TYPE_VOCAB.index(qt)] = 1.0
    X = np.concatenate([X_numeric, X_qtype], axis=1)
    feature_names = numeric_keys + [f"qtype_{q}" for q in QUESTION_TYPE_VOCAB]
    return X, feature_names


def filter_split_by_dataset(split_data, holdout, keep):
    """
    Row-filter a split dict on its per-row source dataset.

    Used for the leave-one-dataset-out protocol, so this query-feature baseline
    is held to exactly the same train/test partition as the evidence-reading
    router it is being compared against -- otherwise the comparison would say
    nothing about generalization.
    """
    if "dataset_names" not in split_data:
        raise ValueError(f"split has no per-row dataset_names (keys: {sorted(split_data)})")
    names = split_data["dataset_names"]
    n_rows = len(names)
    keep_idx = [i for i, name in enumerate(names) if (name == holdout) == bool(keep)]
    if not keep_idx:
        raise ValueError(
            f"Dataset filter (holdout={holdout!r}, keep={keep}) left 0 rows. "
            f"Present: {sorted(set(names))}."
        )
    filtered = {}
    for key, value in split_data.items():
        if isinstance(value, np.ndarray) and value.shape[0] == n_rows:
            filtered[key] = value[keep_idx]
        elif isinstance(value, (list, tuple)) and len(value) == n_rows:
            filtered[key] = [value[i] for i in keep_idx]
        else:
            filtered[key] = value
    return filtered


def downstream_metrics(preds, f1s):
    preds = np.asarray(preds)
    f1s = np.asarray(f1s)
    selected_f1 = float(f1s[np.arange(len(preds)), preds].mean())
    oracle_f1 = float(f1s.max(axis=1).mean())
    best_fixed_f1 = float(max(f1s[:, i].mean() for i in range(f1s.shape[1])))
    return {
        "selected_f1": selected_f1,
        "oracle_f1": oracle_f1,
        "best_fixed_f1": best_fixed_f1,
        "improvement_vs_best_fixed": selected_f1 - best_fixed_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="LTRR-style learned-ranker baseline")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str,
                         default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--resplit_train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout_dataset", type=str, default=None,
        help="Leave-one-dataset-out: train on the other datasets, test only on this one. "
             "Same protocol as train_head.py --holdout_dataset.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    assert action_names == retrievers, (
        f"discover_actions returned a different order/set than --retrievers: "
        f"{action_names} vs {retrievers}"
    )
    print(f"Datasets: {dataset_names}")
    print(f"Retrievers (=classes): {retrievers}")

    print("Loading splits (identical filtering/test-set to the main pipeline)...")
    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=args.threshold,
        max_samples=None,
        base_dir=args.results_base_dir,
        label_metric=args.label_metric,
        top_k_docs=args.top_k_docs,
    )
    if args.holdout_dataset:
        if args.holdout_dataset not in dataset_names:
            raise ValueError(
                f"--holdout_dataset {args.holdout_dataset!r} is not among {dataset_names}."
            )
        # Filter BEFORE the resplit, so held-out rows cannot re-enter training
        # through the train/val merge-and-reshuffle.
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
    print(f"Train: {len(train_data['labels'])}  Val: {len(val_data['labels'])}  Test: {len(test_data['labels'])}")

    print("Building raw per-retriever score lookup (query-time features only, no label leakage)...")
    raw_lookup = build_raw_score_lookup(args.results_base_dir, dataset_names, retrievers)

    X_train, feature_names = build_feature_matrix(train_data, raw_lookup, retrievers)
    X_val, _ = build_feature_matrix(val_data, raw_lookup, retrievers)
    X_test, _ = build_feature_matrix(test_data, raw_lookup, retrievers)
    y_train = np.array(train_data["labels"])
    y_val = np.array(val_data["labels"])
    f1_train = np.array(train_data["f1_scores"])
    f1_val = np.array(val_data["f1_scores"])
    f1_test = np.array(test_data["f1_scores"])

    # Model selection on VALIDATION: each model family is fit on train and
    # scored on val; the val-best family is then refit on train+val for the
    # final test number. (Previously the family was picked by test F1, which
    # made the reported LTRR number optimistic.)
    def make_models():
        return {
            "logistic_regression": LogisticRegression(
                max_iter=2000, class_weight="balanced",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=500, max_depth=None, class_weight="balanced", random_state=args.seed,
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=args.seed),
        }

    def fit_predict(name, model, X_fit, y_fit, X_pred):
        if name == "logistic_regression":
            scaler = StandardScaler().fit(X_fit)
            X_fit, X_pred = scaler.transform(X_fit), scaler.transform(X_pred)
        model.fit(X_fit, y_fit)
        return model.predict(X_pred)

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    results = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "holdout_dataset": args.holdout_dataset,
        "selection": "val_selected_f1",
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(f1_test)),
        "models": {},
    }
    for name, model in make_models().items():
        print(f"\nFitting {name} on train, scoring val...")
        val_metrics = downstream_metrics(fit_predict(name, model, X_train, y_train, X_val), f1_val)
        print(f"\nRefitting {name} on train+val, scoring test...")
        test_metrics = downstream_metrics(
            fit_predict(name, make_models()[name], X_trainval, y_trainval, X_test), f1_test
        )
        print(f"{name}: val selected_f1={val_metrics['selected_f1']:.4f}  "
              f"test selected_f1={test_metrics['selected_f1']:.4f} "
              f"(best_fixed={test_metrics['best_fixed_f1']:.4f}, oracle={test_metrics['oracle_f1']:.4f}, "
              f"vs_best_fixed={test_metrics['improvement_vs_best_fixed']:+.4f})")
        results["models"][name] = {**test_metrics, "val_metrics": val_metrics}

    best_name = max(results["models"], key=lambda n: results["models"][n]["val_metrics"]["selected_f1"])
    results["best_model"] = best_name
    results["best_model_metrics"] = {
        k: v for k, v in results["models"][best_name].items() if k != "val_metrics"
    }
    print(f"\nBest LTRR-style model: {best_name} -> {results['best_model_metrics']}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
