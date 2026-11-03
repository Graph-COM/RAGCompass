"""
R3AG baseline (Zhao et al. 2026): two-stage training + evaluation.

Reproduces R3AG's method -- not its numbers -- on our own datasets, retrievers,
and reader, so it sits on the identical train/val/test partition as our router
and the LTRR-style baseline (same f1-gap filter, same resplit) and can be
compared against them fairly. See r3ag_model.py's module docstring:
unlike our own router, R3AG never reads this query's retrieved evidence -- each
retriever is a learned identity token, fused with the query via attention, and
document quality only shapes which retriever is a training positive/negative
(Appendix B of the paper). This script measures what that architecture
actually does on our setup, including under the two generalization protocols
(--holdout_dataset, --holdout_retriever).

Prerequisites:
  1. `python -m baselines.r3ag_judge ...` has
     already produced the retrieval-quality judge cache this script reads.
  2. The retrieval-result JSONLs already carry `answers` / `pred_ans` (used
     here to compute EM -- see src/metric.py:get_em -- since no `em` field is
     stored in the raw files, only `f1`).

Two-stage recipe (Section 4.2 of the paper):
  Stage 1: two independent contrastive encoders, phi_r (retrieval quality,
    supervised by the judge cache) and phi_g (generation utility, supervised
    by EM + beta*F1 + gamma*sigma(retriever)), each trained with a top-k
    multi-positive InfoNCE loss over the full candidate pool.
  Stage 2: freeze phi_r/phi_g, cache their embeddings, train the
    R3AGFusionHead (query-conditioned attention over each candidate's own
    [r_Ri, g_Ri] pair) with BCE against the same top-k utility target plus an
    L2 regularizer pulling the fused representation toward g_Ri (Eq. 8-9).

Usage:
    python -m baselines.r3ag_train \
        --datasets nq,musique,hotpotqa,2wikimultihopqa \
        --retrievers rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step \
        --judge_cache_base_dir r3ag_quality_judge_cache \
        --seed 42 \
        --output_json <path>
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.optim import AdamW

from src.metric import get_em
from router.train_e2e import resplit_split_data
from router.train_head import (
    discover_actions,
    item_id,
    load_action_splits,
    load_results_file,
    parse_dataset_names,
)
from baselines.ltrr import filter_split_by_dataset
from baselines.r3ag_model import R3AGCapabilityEncoder, R3AGFusionHead, routing_scores
from router.shared_utils import DEFAULT_RESULTS_BASE_DIR, set_seed, split_specs

DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]
DEFAULT_NO_RETRIEVAL_ACTION = "no_retrieval_0_step"


# --------------------------------------------------------------------------- #
# EM / judge-score lookups
# --------------------------------------------------------------------------- #

def em_value(answers, pred_ans):
    """Mirrors the list-vs-string handling every other script in this repo
    applies at the call site (e.g. main_hf.py's [ANS]-split branch): a
    multi-answer prediction counts as correct if any candidate exact-matches."""
    if isinstance(pred_ans, list):
        return max((get_em(answers, candidate) for candidate in pred_ans), default=0.0)
    return get_em(answers, pred_ans)


def build_raw_row_lookup(base_dir, dataset_names, action_names):
    """id -> {action: row}, scanning every split dir of every dataset (mirrors
    ltrr.py's build_raw_score_lookup) so a row's answers/pred_ans can
    be found regardless of which resplit bucket it lands in after
    resplit_split_data's merge+reshuffle."""
    lookup = {}
    for dataset_name in dataset_names:
        for _, split_dir, _ in split_specs(dataset_name):
            for action_name in action_names:
                path = os.path.join(base_dir, split_dir, f"{action_name}.jsonl")
                rows = load_results_file(path)
                for idx, row in enumerate(rows):
                    lookup.setdefault(item_id(row, idx), {})[action_name] = row
    return lookup


def build_judge_lookup(judge_cache_base_dir, dataset_names, retriever_names):
    """id -> {retriever: overall_score}, scanning every split dir. Only covers
    the real retrievers -- the no-retrieval action is never judged (see
    r3ag_judge.py's module docstring)."""
    lookup = {}
    for dataset_name in dataset_names:
        for _, split_dir, _ in split_specs(dataset_name):
            for retriever_name in retriever_names:
                path = os.path.join(judge_cache_base_dir, split_dir, f"{retriever_name}.jsonl")
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        f"Missing judge cache {path}. Run "
                        "r3ag_judge.py before r3ag_train.py."
                    )
                rows = load_results_file(path)
                for idx, row in enumerate(rows):
                    lookup.setdefault(item_id(row, idx), {})[retriever_name] = row["overall_score"]
    return lookup


def compute_em_and_quality_matrices(split_data, raw_lookup, judge_lookup, action_names, no_retrieval_action):
    """Returns (em_matrix, quality_matrix), each a [N, len(action_names)] list
    of lists aligned with split_data["ids"] / split_data["f1_scores"]."""
    em_matrix = []
    quality_matrix = []
    for row_id in split_data["ids"]:
        row_lookup = raw_lookup.get(row_id)
        if row_lookup is None:
            raise ValueError(f"No raw rows found for id={row_id!r}; EM cannot be computed.")
        em_row = []
        quality_row = []
        for action_name in action_names:
            raw_item = row_lookup.get(action_name)
            if raw_item is None:
                raise ValueError(f"Missing raw row for id={row_id!r} action={action_name!r}.")
            em_row.append(em_value(raw_item.get("answers"), raw_item.get("pred_ans")))
            if action_name == no_retrieval_action:
                quality_row.append(0.0)  # nothing retrieved -> floor on retrieval quality
            else:
                judge_row = judge_lookup.get(row_id)
                if judge_row is None or action_name not in judge_row:
                    raise ValueError(f"Missing judge score for id={row_id!r} action={action_name!r}.")
                quality_row.append(judge_row[action_name])
        em_matrix.append(em_row)
        quality_matrix.append(quality_row)
    return em_matrix, quality_matrix


def compute_utility_matrix(em_matrix, f1_matrix, sigma_by_action, action_names, beta, gamma):
    """u(q, Ri) = EM(q, Ri) + beta * F1(q, Ri) + gamma * sigma(Ri) -- Eq. 5."""
    return [
        [
            em_matrix[i][a] + beta * f1_matrix[i][a] + gamma * sigma_by_action[action_names[a]]
            for a in range(len(action_names))
        ]
        for i in range(len(em_matrix))
    ]


def compute_sigma_by_action(em_matrix, action_names):
    """Global average correctness per retriever (Eq. 5's sigma(Ri)), computed
    once from the training split's EM matrix and reused for val/test."""
    arr = np.asarray(em_matrix, dtype=float)
    return {action_names[a]: float(arr[:, a].mean()) for a in range(len(action_names))}


# --------------------------------------------------------------------------- #
# Stage 1: contrastive capability encoders
# --------------------------------------------------------------------------- #

def top_k_positive_mask(matrix, top_k):
    """Eq. 2/5's V+(q): the top_k highest-scoring candidates per row.

    Ties across the top-k boundary are broken at random (np.random, seeded by
    set_seed). A plain argsort breaks them by column order, and the judge's
    integer-averaged scores tie at the boundary on ~64% of training queries, so
    that silently made the first-listed retrievers the quality positives."""
    scores = np.asarray(matrix, dtype=float)
    tie_break = np.random.random(scores.shape)
    order = np.lexsort((tie_break, -scores), axis=1)
    mask = np.zeros_like(scores, dtype=bool)
    for row_idx in range(scores.shape[0]):
        mask[row_idx, order[row_idx, :top_k]] = True
    return mask


def multi_positive_infonce(query_emb, candidate_embs, positive_mask, temperature):
    """Eq. 2/5's contrastive loss, computed over the full candidate pool at
    once (V+(q) union V-(q) is always the whole pool here, since every query
    is scored against the same fixed R candidates): a multi-positive InfoNCE
    with the positive set given by top_k_positive_mask.

    query_emb: [B, H] (already L2-normalized). candidate_embs: [R, H]
    (already L2-normalized, shared across the batch since retriever identity
    embeddings do not depend on the query). positive_mask: [B, R] bool.
    """
    sims = (query_emb @ candidate_embs.t()) / temperature  # [B, R]
    log_z = torch.logsumexp(sims, dim=1, keepdim=True)
    log_probs = sims - log_z
    pos = positive_mask.float()
    num_pos = pos.sum(dim=1).clamp(min=1.0)
    loss_per_example = -(log_probs * pos).sum(dim=1) / num_pos
    return loss_per_example.mean()


def train_stage1(
    quality_encoder,
    utility_encoder,
    questions,
    quality_targets,
    utility_targets,
    train_action_names,
    top_k,
    temperature,
    epochs,
    batch_size,
    learning_rate,
    device,
):
    quality_encoder.backbone.to(device)
    utility_encoder.backbone.to(device)
    quality_pos_mask = torch.from_numpy(top_k_positive_mask(quality_targets, top_k))
    utility_pos_mask = torch.from_numpy(top_k_positive_mask(utility_targets, top_k))

    trainable_params = [p for p in quality_encoder.backbone.parameters() if p.requires_grad]
    trainable_params += [p for p in utility_encoder.backbone.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=learning_rate)

    num_examples = len(questions)
    for epoch in range(epochs):
        quality_encoder.backbone.train()
        utility_encoder.backbone.train()
        permutation = np.random.permutation(num_examples)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_examples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_questions = [questions[i] for i in batch_idx]

            quality_query_emb = quality_encoder.encode_query(batch_questions)
            quality_candidate_emb = quality_encoder.encode_retrievers_subset(train_action_names)
            loss_qual = multi_positive_infonce(
                quality_query_emb, quality_candidate_emb, quality_pos_mask[batch_idx].to(device), temperature
            )

            utility_query_emb = utility_encoder.encode_query(batch_questions)
            utility_candidate_emb = utility_encoder.encode_retrievers_subset(train_action_names)
            loss_util = multi_positive_infonce(
                utility_query_emb, utility_candidate_emb, utility_pos_mask[batch_idx].to(device), temperature
            )

            loss = loss_qual + loss_util
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
        print(f"[Stage 1] epoch {epoch + 1}/{epochs}: mean loss = {total_loss / max(num_batches, 1):.4f}")


@torch.no_grad()
def encode_questions(encoder, questions, batch_size, device):
    encoder.backbone.to(device)
    encoder.backbone.eval()
    embeddings = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        embeddings.append(encoder.encode_query(batch).cpu())
    return torch.cat(embeddings, dim=0)


@torch.no_grad()
def encode_retrievers(encoder, action_names, device):
    encoder.backbone.to(device)
    encoder.backbone.eval()
    return encoder.encode_retrievers_subset(action_names).cpu()


# --------------------------------------------------------------------------- #
# Stage 2: fusion head
# --------------------------------------------------------------------------- #

def train_stage2(
    fusion_head,
    r_q,
    g_q,
    r_candidates,
    g_candidates,
    utility_targets,
    top_k,
    lambda_reg,
    epochs,
    batch_size,
    learning_rate,
    device,
    val_fn=None,
):
    """val_fn(fusion_head) -> validation selected F1. When given, the epoch
    with the best validation F1 is restored at the end -- the same
    best-epoch-on-val selection the main pipeline (train_head.py)
    applies to our own router."""
    fusion_head.to(device)
    best_val, best_state, best_epoch = -float("inf"), None, None
    positive_mask = torch.from_numpy(top_k_positive_mask(utility_targets, top_k)).float()
    optimizer = AdamW(fusion_head.parameters(), lr=learning_rate)
    num_examples = r_q.shape[0]
    r_candidates = r_candidates.to(device)
    g_candidates = g_candidates.to(device)

    for epoch in range(epochs):
        fusion_head.train()
        permutation = np.random.permutation(num_examples)
        total_loss = 0.0
        num_batches = 0
        for start in range(0, num_examples, batch_size):
            batch_idx = permutation[start : start + batch_size]
            batch_size_actual = len(batch_idx)
            batch_r_q = r_q[batch_idx].to(device)
            batch_g_q = g_q[batch_idx].to(device)
            batch_targets = positive_mask[batch_idx].to(device)

            batch_r_candidates = r_candidates.unsqueeze(0).expand(batch_size_actual, -1, -1)
            batch_g_candidates = g_candidates.unsqueeze(0).expand(batch_size_actual, -1, -1)

            fused = fusion_head(batch_r_q, batch_r_candidates, batch_g_candidates)  # [B, R, H]
            scores = routing_scores(batch_g_q, fused)  # [B, R] -- cosine sim, used as the BCE logit (Eq. 6)

            loss_cls = torch.nn.functional.binary_cross_entropy_with_logits(scores, batch_targets)
            # Eq. 8: squared L2 norm per candidate (summed over the hidden dim),
            # not F.mse_loss's per-element mean, which shrinks lambda_reg by 1/H.
            loss_reg = ((fused - batch_g_candidates) ** 2).sum(dim=-1).mean()
            loss = loss_cls + lambda_reg * loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1
        msg = f"[Stage 2] epoch {epoch + 1}/{epochs}: mean loss = {total_loss / max(num_batches, 1):.4f}"
        if val_fn is not None:
            val_f1 = val_fn(fusion_head)
            msg += f"  val selected_f1 = {val_f1:.4f}"
            if val_f1 > best_val:
                best_val, best_epoch = val_f1, epoch + 1
                best_state = {k: v.detach().clone() for k, v in fusion_head.state_dict().items()}
        print(msg)
    if best_state is not None:
        fusion_head.load_state_dict(best_state)
        print(f"[Stage 2] restored best epoch {best_epoch} (val selected_f1 = {best_val:.4f})")
    return best_epoch


@torch.no_grad()
def route(fusion_head, r_q, g_q, r_candidates, g_candidates, device, batch_size=64):
    fusion_head.to(device)
    fusion_head.eval()
    r_candidates = r_candidates.to(device)
    g_candidates = g_candidates.to(device)
    all_scores = []
    for start in range(0, r_q.shape[0], batch_size):
        batch_r_q = r_q[start : start + batch_size].to(device)
        batch_g_q = g_q[start : start + batch_size].to(device)
        batch_size_actual = batch_r_q.shape[0]
        batch_r_candidates = r_candidates.unsqueeze(0).expand(batch_size_actual, -1, -1)
        batch_g_candidates = g_candidates.unsqueeze(0).expand(batch_size_actual, -1, -1)
        fused = fusion_head(batch_r_q, batch_r_candidates, batch_g_candidates)
        scores = routing_scores(batch_g_q, fused)
        all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0)


# --------------------------------------------------------------------------- #
# Evaluation (downstream F1 -- matches \sys / LTRR reporting, not the
# EM+F1+sigma utility composite, which is a training signal only)
# --------------------------------------------------------------------------- #

def downstream_metrics(scores, f1_matrix, action_names, held_out_retriever=None):
    scores = scores.numpy()
    predictions = np.argmax(scores, axis=1)
    f1s = np.asarray(f1_matrix, dtype=float)
    selected_f1 = float(f1s[np.arange(len(predictions)), predictions].mean())
    oracle_f1 = float(f1s.max(axis=1).mean())
    best_fixed_f1 = float(max(f1s[:, i].mean() for i in range(f1s.shape[1])))
    metrics = {
        "selected_f1": selected_f1,
        "oracle_f1": oracle_f1,
        "best_fixed_f1": best_fixed_f1,
        "improvement_vs_best_fixed": selected_f1 - best_fixed_f1,
        # Fraction of queries routed to each action: a router that collapses to
        # one action shows up here as a single entry near 1.0.
        "pick_distribution": {
            name: float((predictions == idx).mean()) for idx, name in enumerate(action_names)
        },
    }
    if held_out_retriever is not None:
        held_out_idx = action_names.index(held_out_retriever)
        picks_held_out = predictions == held_out_idx
        metrics["picks_held_out_retriever"] = float(picks_held_out.mean())
        oracle_is_held_out = np.argmax(f1s, axis=1) == held_out_idx
        if oracle_is_held_out.any():
            metrics["picks_held_out_when_oracle"] = float(
                picks_held_out[oracle_is_held_out].mean()
            )
        else:
            metrics["picks_held_out_when_oracle"] = None
    return metrics


def main():
    parser = argparse.ArgumentParser(description="R3AG baseline: two-stage training + evaluation")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument(
        "--no_retrieval_action", type=str, default=DEFAULT_NO_RETRIEVAL_ACTION,
        help="R3AG's null retriever R0. Pass 'none' to route over the real retrievers only -- "
             "the pool (and hence the filtered test set) our own router and LTRR are evaluated on.",
    )
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--judge_cache_base_dir", type=str, default="r3ag_quality_judge_cache")
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--top_k_contrastive", type=int, default=2, help="R3AG's k in Top-k_R(...); paper uses 2.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Contrastive temperature tau.")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument(
        "--lambda_reg", type=float, default=0.1,
        help="Weight on the L2 term pulling the fused representation toward g_Ri (Eq. 8-9). "
             "Not numerically specified in the paper; 0.1 is a reasonable default, exposed for tuning.",
    )
    parser.add_argument("--num_trainable_layers", type=int, default=2)
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage1_batch_size", type=int, default=16)
    parser.add_argument("--stage1_lr", type=float, default=2e-6)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage2_batch_size", type=int, default=64)
    parser.add_argument("--stage2_lr", type=float, default=1e-4)
    parser.add_argument("--resplit_train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--holdout_dataset", type=str, default=None,
        help="Leave-one-dataset-out: train on the other datasets, test only on this one.",
    )
    parser.add_argument(
        "--holdout_retriever", type=str, default=None,
        help="Leave-one-retriever-out: exclude this retriever's token from both training "
             "stages (its embedding never receives gradient) while keeping it scoreable at "
             "test time -- the mechanism-shift protocol.",
    )
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    real_retrievers = [name.strip() for name in args.retrievers.split(",") if name.strip()]
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    assert action_names == real_retrievers, (
        f"discover_actions returned a different order/set than --retrievers: {action_names} vs {real_retrievers}"
    )
    if args.no_retrieval_action.lower() in ("", "none"):
        args.no_retrieval_action = None
    else:
        action_names = action_names + [args.no_retrieval_action]
    print(f"Datasets: {dataset_names}")
    print(f"Retriever pool: {action_names}")

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
            raise ValueError(f"--holdout_dataset {args.holdout_dataset!r} is not among {dataset_names}.")
        splits = dict(splits)
        splits["train"] = filter_split_by_dataset(splits["train"], args.holdout_dataset, keep=False)
        splits["val"] = filter_split_by_dataset(splits["val"], args.holdout_dataset, keep=False)
        splits["test"] = filter_split_by_dataset(splits["test"], args.holdout_dataset, keep=True)
        print(f"Leave-one-dataset-out: holding out {args.holdout_dataset}")

    train_data, val_data = resplit_split_data(
        splits["train"], splits["val"], train_ratio=args.resplit_train_ratio, seed=args.seed
    )
    test_data = splits["test"]
    print(f"Train: {len(train_data['labels'])}  Val: {len(val_data['labels'])}  Test: {len(test_data['labels'])}")

    print("Building raw-row and judge-score lookups...")
    raw_lookup = build_raw_row_lookup(args.results_base_dir, dataset_names, action_names)
    judge_lookup = build_judge_lookup(args.judge_cache_base_dir, dataset_names, real_retrievers)

    def prepare(split_data):
        em_matrix, quality_matrix = compute_em_and_quality_matrices(
            split_data, raw_lookup, judge_lookup, action_names, args.no_retrieval_action
        )
        return em_matrix, quality_matrix

    train_em, train_quality = prepare(train_data)
    val_em, val_quality = prepare(val_data)
    test_em, test_quality = prepare(test_data)

    sigma_by_action = compute_sigma_by_action(train_em, action_names)
    print(f"sigma(Ri) (mean training EM per retriever): {sigma_by_action}")

    train_utility = compute_utility_matrix(train_em, train_data["f1_scores"], sigma_by_action, action_names, args.beta, args.gamma)
    val_utility = compute_utility_matrix(val_em, val_data["f1_scores"], sigma_by_action, action_names, args.beta, args.gamma)
    test_utility = compute_utility_matrix(test_em, test_data["f1_scores"], sigma_by_action, action_names, args.beta, args.gamma)

    if args.holdout_retriever:
        if args.holdout_retriever not in real_retrievers:
            raise ValueError(f"--holdout_retriever must be one of {real_retrievers}, got {args.holdout_retriever!r}")
        held_out_idx = action_names.index(args.holdout_retriever)
        train_action_names = [name for name in action_names if name != args.holdout_retriever]
        train_quality_cols = [i for i in range(len(action_names)) if i != held_out_idx]
        train_quality_train = [[row[i] for i in train_quality_cols] for row in train_quality]
        train_utility_train = [[row[i] for i in train_quality_cols] for row in train_utility]
        print(f"Leave-one-retriever-out: holding out {args.holdout_retriever} from both training stages")
    else:
        train_action_names = action_names
        train_quality_train = train_quality
        train_utility_train = train_utility

    quality_encoder = R3AGCapabilityEncoder(retriever_names=action_names, num_trainable_layers=args.num_trainable_layers)
    utility_encoder = R3AGCapabilityEncoder(retriever_names=action_names, num_trainable_layers=args.num_trainable_layers)

    train_stage1(
        quality_encoder=quality_encoder,
        utility_encoder=utility_encoder,
        questions=train_data["queries"],
        quality_targets=train_quality_train,
        utility_targets=train_utility_train,
        train_action_names=train_action_names,
        top_k=args.top_k_contrastive,
        temperature=args.temperature,
        epochs=args.stage1_epochs,
        batch_size=args.stage1_batch_size,
        learning_rate=args.stage1_lr,
        device=device,
    )

    print("Caching Stage 1 embeddings for Stage 2...")
    for param in list(quality_encoder.backbone.parameters()) + list(utility_encoder.backbone.parameters()):
        param.requires_grad = False

    r_q_train = encode_questions(quality_encoder, train_data["queries"], args.stage1_batch_size, device)
    g_q_train = encode_questions(utility_encoder, train_data["queries"], args.stage1_batch_size, device)
    r_candidates_train = encode_retrievers(quality_encoder, train_action_names, device)
    g_candidates_train = encode_retrievers(utility_encoder, train_action_names, device)

    # Stage 1 is frozen from here, so every split's embeddings are final.
    r_candidates_full = encode_retrievers(quality_encoder, action_names, device)
    g_candidates_full = encode_retrievers(utility_encoder, action_names, device)
    split_embeddings = {
        name: (
            encode_questions(quality_encoder, data["queries"], args.stage1_batch_size, device),
            encode_questions(utility_encoder, data["queries"], args.stage1_batch_size, device),
        )
        for name, data in (("val", val_data), ("test", test_data))
    }

    def eval_split(name, split_data, head):
        r_q, g_q = split_embeddings[name]
        scores = route(head, r_q, g_q, r_candidates_full, g_candidates_full, device)
        return downstream_metrics(scores, split_data["f1_scores"], action_names, held_out_retriever=args.holdout_retriever)

    fusion_head = R3AGFusionHead(hidden_size=r_q_train.shape[-1])
    best_stage2_epoch = train_stage2(
        fusion_head=fusion_head,
        r_q=r_q_train,
        g_q=g_q_train,
        r_candidates=r_candidates_train,
        g_candidates=g_candidates_train,
        utility_targets=train_utility_train,
        top_k=args.top_k_contrastive,
        lambda_reg=args.lambda_reg,
        epochs=args.stage2_epochs,
        batch_size=args.stage2_batch_size,
        learning_rate=args.stage2_lr,
        device=device,
        val_fn=lambda head: eval_split("val", val_data, head)["selected_f1"],
    )

    print("Evaluating on val/test with the full retriever pool (including any held-out retriever)...")
    val_metrics = eval_split("val", val_data, fusion_head)
    test_metrics = eval_split("test", test_data, fusion_head)
    print(f"Val:  {val_metrics}")
    print(f"Test: {test_metrics}")

    # Example-set gate (same constants as ltrr_shared_scorer.py): on the pooled
    # 5-retriever setting the test population must be the one the main results table reports,
    # or the number is not comparable to our router / LTRR.
    gate_passed = None
    if args.no_retrieval_action is None and not args.holdout_dataset:
        from baselines.ltrr_shared_scorer import GATE_TOL, REF_BEST_FIXED, REF_ORACLE
        gate_passed = bool(
            abs(test_metrics["best_fixed_f1"] - REF_BEST_FIXED) < GATE_TOL
            and abs(test_metrics["oracle_f1"] - REF_ORACLE) < GATE_TOL
        )
        print(f"Example-set gate: best_fixed={test_metrics['best_fixed_f1']:.4f} (ref {REF_BEST_FIXED:.4f}), "
              f"oracle={test_metrics['oracle_f1']:.4f} (ref {REF_ORACLE:.4f}) -> {'PASS' if gate_passed else 'FAIL'}")
        if not gate_passed:
            raise SystemExit("Example-set gate FAILED: not the reported test set. Refusing to write output.")

    results = {
        "method": "r3ag",
        "datasets": dataset_names,
        "action_names": action_names,
        "holdout_dataset": args.holdout_dataset,
        "holdout_retriever": args.holdout_retriever,
        "seed": args.seed,
        "best_stage2_epoch": best_stage2_epoch,
        "example_set_gate_passed": gate_passed,
        "hyperparameters": {k: v for k, v in vars(args).items() if k != "output_json"},
        "n_train": len(train_data["labels"]),
        "n_val": len(val_data["labels"]),
        "n_test": len(test_data["labels"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        # Same key as the LTRR baselines' output JSON, so results can be
        # collected uniformly.
        "best_model_metrics": test_metrics,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")


if __name__ == "__main__":
    main()
