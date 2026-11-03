"""
Joint (end-to-end) training of the reranker backbone and the comparison head.

Every step encodes each (query, passage set) prompt with the trainable backbone,
feeds the per-retriever representations to the head, and backpropagates the
routing loss through both. This is the alternative to the two-stage recipe
(train_scorer.py -> cache_embeddings.py -> train_head.py), where the backbone
is frozen before the head is trained.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# PyTorch SDPA can select a cuDNN attention backend that has no valid execution
# plan on some GPU/cuDNN/torch combinations, failing with
#   "cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built"
# Backbone training is the only path in this file that runs
# attention, and the other SDPA backends are fine, so disable just the cuDNN one
# rather than forcing eager attention everywhere.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

try:
    import wandb
except ImportError:
    wandb = None

from router.train_head import (
    LateInteractionHeadModel,
    LateInterHeadModel,
    QwenActionDataset,
    _wandb_log_eval,
    compute_training_loss,
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from router.shared_utils import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_RESULTS_BASE_DIR,
    DEFAULT_RUNS_DIR,
    install_file_logger,
    retriever_tag,
    sanitize_name,
    set_seed,
)

METHOD_DIR = "e2e_ft"


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #

def resplit_split_data(train_data, val_data, train_ratio=0.7, seed=42):
    """Merge source train+val dicts, resplit at train_ratio (test unchanged)."""
    keys = list(train_data.keys())
    pool = {k: train_data[k] + val_data[k] for k in keys}
    n = len(pool["labels"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(n * train_ratio))
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    new_train = {k: [pool[k][i] for i in train_idx] for k in keys}
    new_val = {k: [pool[k][i] for i in val_idx] for k in keys}
    return new_train, new_val


# --------------------------------------------------------------------------- #
# Differentiable backbone forward
# --------------------------------------------------------------------------- #

def right_to_left_pad(input_ids, attention_mask, pad_token_id):
    """Convert right-padded tensors to left-padded, so every row's last real token sits at index -1."""
    B, L = input_ids.shape
    out_ids = input_ids.new_full((B, L), pad_token_id)
    out_mask = attention_mask.new_zeros(B, L)
    for i in range(B):
        actual_len = int(attention_mask[i].sum().item())
        out_ids[i, L - actual_len:] = input_ids[i, :actual_len]
        out_mask[i, L - actual_len:] = 1
    return out_ids, out_mask


def embed_last_token(backbone, input_ids, attention_mask, encode_batch_size, pad_token_id):
    """
    One pooled vector per prompt: the hidden state at its last token.

    Caller controls backbone.train()/eval(). Chunked by encode_batch_size;
    with gradient checkpointing only layer inputs are stored, which keeps peak
    memory low across the full max_length sequence.

    Returns hidden [N, 1, H] (with gradient when called in train mode).
    """
    N = input_ids.size(0)
    lp_ids, lp_mask = right_to_left_pad(input_ids, attention_mask, pad_token_id)
    chunks = []
    for start in range(0, N, encode_batch_size):
        end = min(start + encode_batch_size, N)
        out = backbone.model(
            input_ids=lp_ids[start:end],
            attention_mask=lp_mask[start:end],
            use_cache=False,
        )
        chunks.append(out.last_hidden_state[:, -1:, :])  # [b, 1, H]
    return torch.cat(chunks, dim=0)


def embed_document_boundaries(backbone, input_ids, attention_mask, doc_end_offsets, encode_batch_size):
    """
    Differentiable version of encode_flat_document_boundaries
    (cache_embeddings.py): one hidden state per retrieved passage, at each
    passage's own last-token position.

    doc_end_offsets: [N, top_k_docs], 0-indexed position of each retained
    document's last token (from QwenActionDataset.encode_text_per_document).
    input_ids/attention_mask are QwenActionDataset's native right-padded rows.

    Returns hidden [N, top_k_docs, H].
    """
    N = input_ids.size(0)
    chunks = []
    for start in range(0, N, encode_batch_size):
        end = min(start + encode_batch_size, N)
        out = backbone.model(
            input_ids=input_ids[start:end],
            attention_mask=attention_mask[start:end],
            use_cache=False,
        )
        last_hidden = out.last_hidden_state  # [b, L, H]
        H = last_hidden.shape[-1]
        chunk_offsets = doc_end_offsets[start:end]
        idx = chunk_offsets.clamp(max=last_hidden.shape[1] - 1)
        keep = last_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))  # [b, K, H]
        chunks.append(keep)
    return torch.cat(chunks, dim=0)  # [N, K, H]


def embed_prompts(backbone, batch, device, extraction_source, encode_batch_size, pad_token_id):
    """Encode a batch of [B, C, seq_len] prompts. Returns hidden [B*C, K, H] with K=1 for last_token."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B, C, seq_len = input_ids.shape
    flat_ids = input_ids.view(B * C, seq_len)
    flat_mask = attention_mask.view(B * C, seq_len)
    if extraction_source == "document_boundaries":
        doc_end_offsets = batch["doc_end_offsets"].to(device).view(B * C, -1)
        return embed_document_boundaries(
            backbone=backbone,
            input_ids=flat_ids,
            attention_mask=flat_mask,
            doc_end_offsets=doc_end_offsets,
            encode_batch_size=encode_batch_size,
        )
    return embed_last_token(
        backbone=backbone,
        input_ids=flat_ids,
        attention_mask=flat_mask,
        encode_batch_size=encode_batch_size,
        pad_token_id=pad_token_id,
    )


# --------------------------------------------------------------------------- #
# Metric learning loss
# --------------------------------------------------------------------------- #

def metric_learning_loss(embeddings, f1s):
    """MSE between actual cosine distance and |F1_i - F1_j| for all retriever pairs."""
    B, C, _ = embeddings.shape
    h = F.normalize(embeddings.float(), dim=-1)
    sim = torch.bmm(h, h.transpose(1, 2))                          # [B, C, C]
    target_dist = (f1s.unsqueeze(2) - f1s.unsqueeze(1)).abs()     # [B, C, C]
    actual_dist = 1.0 - sim
    mask = torch.triu(torch.ones(C, C, device=h.device, dtype=torch.bool), diagonal=1)
    return F.mse_loss(actual_dist[:, mask], target_dist[:, mask])


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train_epoch(backbone, head, dataloader, optimizer, scheduler, device,
                encode_batch_size, pad_token_id,
                loss_mode, soft_target_temperature, max_grad_norm,
                metric_lambda, wandb_run, global_step, epoch, num_keep_tokens=1,
                extraction_source="last_token"):
    backbone.train()
    head.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch} train"):
        B, C, _ = batch["input_ids"].shape
        hidden = embed_prompts(
            backbone, batch, device, extraction_source, encode_batch_size, pad_token_id,
        )  # [B*C, K, H]
        if num_keep_tokens > 1:
            embeddings = hidden.view(B, C, num_keep_tokens, -1)  # [B, C, K, H]
            metric_emb = embeddings.mean(dim=2)                  # pool K for metric loss
        else:
            embeddings = hidden.view(B, C, -1)  # [B, C, H]
            metric_emb = embeddings

        logits = head(embeddings)           # [B, C]
        loss_routing = compute_training_loss(
            logits=logits,
            batch=batch,
            labels=batch["label"].to(device),
            device=device,
            loss_mode=loss_mode,
            soft_target_temperature=soft_target_temperature,
        )
        loss_metric = metric_learning_loss(metric_emb, batch["action_f1s"].to(device))
        loss = loss_routing + metric_lambda * loss_metric

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()),
                max_grad_norm,
            )
        optimizer.step()
        scheduler.step()

        step_loss = float(loss.item())
        total_loss += step_loss
        global_step += 1

        if wandb_run is not None:
            log_dict = {
                "train/loss": step_loss,
                "train/loss_routing": float(loss_routing.item()),
                "train/loss_metric": float(loss_metric.item()),
                "epoch": epoch,
            }
            wandb_run.log(log_dict, step=global_step)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(len(dataloader), 1), global_step


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(backbone, head, dataloader, device, action_names,
             encode_batch_size, pad_token_id, num_keep_tokens=1,
             extraction_source="last_token"):
    backbone.eval()
    head.eval()
    all_preds, all_labels, all_f1s = [], [], []

    for batch in tqdm(dataloader, desc="Eval"):
        B, C, _ = batch["input_ids"].shape
        hidden = embed_prompts(
            backbone, batch, device, extraction_source, encode_batch_size, pad_token_id,
        )  # [B*C, K, H]
        if num_keep_tokens > 1:
            embeddings = hidden.view(B, C, num_keep_tokens, -1)  # [B, C, K, H]
        else:
            embeddings = hidden[:, 0, :].view(B, C, -1)  # [B, C, H]
        logits = head(embeddings)
        preds = logits.argmax(dim=-1).cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].cpu().numpy().tolist())
        all_f1s.extend(batch["action_f1s"].cpu().numpy().tolist())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    f1s = np.array(all_f1s)

    selected_f1 = float(f1s[np.arange(len(preds)), preds].mean())
    oracle_f1 = float(f1s.max(axis=1).mean())
    best_fixed_f1 = float(max(f1s[:, i].mean() for i in range(f1s.shape[1])))

    cls = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }
    downstream = {
        "selected_f1": selected_f1,
        "oracle_f1": oracle_f1,
        "oracle_gap": oracle_f1 - selected_f1,
        "best_fixed_f1": best_fixed_f1,
        "improvement_vs_best_baseline": selected_f1 - best_fixed_f1,
    }
    return {"classification": cls, "downstream": downstream}


def print_eval(prefix, metrics):
    cls = metrics["classification"]
    ds = metrics["downstream"]
    print(
        f"{prefix} — Acc: {cls['accuracy']:.4f}  F1-macro: {cls['f1_macro']:.4f}  "
        f"Selected F1: {ds['selected_f1']:.4f}  Oracle: {ds['oracle_f1']:.4f}  "
        f"vs best-fixed: {ds['improvement_vs_best_baseline']:+.4f}"
    )


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #

def save_checkpoint(backbone, head, tokenizer, ckpt_dir, epoch, metrics):
    os.makedirs(ckpt_dir, exist_ok=True)
    backbone.save_pretrained(os.path.join(ckpt_dir, "backbone"))
    tokenizer.save_pretrained(os.path.join(ckpt_dir, "backbone"))
    head.save_pretrained(os.path.join(ckpt_dir, "head"))
    meta = {"epoch": epoch, "val_selected_f1": metrics["downstream"]["selected_f1"]}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved checkpoint (epoch {epoch}, val F1={meta['val_selected_f1']:.4f}) → {ckpt_dir}")


def load_checkpoint(ckpt_dir, action_names, device, torch_dtype,
                    cross_layers, cross_heads, cross_dropout, use_type_emb, cross_ffn_dim,
                    num_keep_tokens=1, late_interaction_pool="attn"):
    backbone = AutoModelForCausalLM.from_pretrained(
        os.path.join(ckpt_dir, "backbone"), trust_remote_code=True, torch_dtype=torch_dtype,
    ).to(device)
    backbone.config.use_cache = False
    if num_keep_tokens > 1:
        head = LateInteractionHeadModel.load_pretrained(
            os.path.join(ckpt_dir, "head"),
            action_names=action_names,
            device=device,
            n_layers=cross_layers,
            n_heads=cross_heads,
            dropout=cross_dropout,
            use_type_emb=use_type_emb,
            ffn_dim=cross_ffn_dim,
            num_keep_tokens=num_keep_tokens,
            pool=late_interaction_pool,
        )
    else:
        head = LateInterHeadModel.load_pretrained(
            os.path.join(ckpt_dir, "head"),
            action_names=action_names,
            device=device,
            n_layers=cross_layers,
            n_heads=cross_heads,
            dropout=cross_dropout,
            use_type_emb=use_type_emb,
            ffn_dim=cross_ffn_dim,
        )
    return backbone, head


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Joint training of backbone and comparison head")

    # Data
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str, default="auto")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--resplit_train_ratio", type=float, default=0.7)

    # Model
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--pretrained_backbone_path", type=str, default=None,
                        help="Warm-start backbone from this checkpoint directory.")
    parser.add_argument("--pretrained_head_path", type=str, default=None,
                        help="Warm-start head from this checkpoint directory (pass alongside backbone to resume).")
    parser.add_argument("--max_length", type=int, default=2048)

    # Representation fed to the head
    parser.add_argument("--extraction_source", type=str, default="last_token",
                        choices=["last_token", "document_boundaries"],
                        help="last_token (default): one pooled hidden state per retriever, at the last "
                             "prompt token. document_boundaries: one hidden state per retrieved passage, "
                             "at each passage's own last-token position (late-interaction head).")
    parser.add_argument("--late_interaction_pool", type=str, default="attn", choices=["attn", "mean"],
                        help="Pooling over the per-passage tokens in the late-interaction head "
                             "(document_boundaries only).")

    # Head architecture
    parser.add_argument("--cross_layers", type=int, default=2)
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--cross_ffn_dim", type=int, default=2048)
    parser.add_argument("--cross_dropout", type=float, default=0.1)
    parser.add_argument("--no_type_emb", action="store_true")

    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--encode_batch_size", type=int, default=5,
                        help="Sequences per backbone forward chunk (memory knob).")
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--loss_mode", type=str, default="soft_metric_ce",
                        choices=["ce", "soft_metric_ce", "expected_metric",
                                 "pointwise_bce", "pointwise_mse"])
    parser.add_argument("--soft_target_temperature", type=float, default=0.3)
    parser.add_argument("--backbone_dropout", type=float, default=0.0,
                        help="Attention dropout applied to the backbone (train mode only).")
    parser.add_argument("--metric_lambda", type=float, default=0.0,
                        help="Weight on the metric learning loss term (0 = disabled).")
    parser.add_argument("--seed", type=int, default=42)

    # Output
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for logs and metrics.")
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help="Base directory for model checkpoints.")
    parser.add_argument("--wandb_project", type=str,
                        default=os.getenv("WANDB_PROJECT", "ragcompass"))
    parser.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Optional name appended to the run directory, to keep runs apart.")

    args = parser.parse_args()
    # Hidden states kept per retriever: one per retrieved passage, or a single pooled vector.
    num_keep_tokens = args.top_k_docs if args.extraction_source == "document_boundaries" else 1
    set_seed(args.seed)
    use_type_emb = not args.no_type_emb

    # --- Paths ---
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    rtag = retriever_tag(action_names)
    dataset_tag = "__".join(sanitize_name(d) for d in dataset_names)
    run_tag = f"seed_{args.seed}-f1gap_{str(args.threshold).replace('.', '_')}"
    if args.run_name:
        run_tag = f"{run_tag}-{args.run_name}"

    model_dir = os.path.join(
        args.checkpoints_dir, METHOD_DIR, dataset_tag, rtag, run_tag, "checkpoints"
    )
    metrics_dir = os.path.join(
        args.runs_dir, METHOD_DIR, dataset_tag, rtag, run_tag, "metrics"
    )
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    install_file_logger(os.path.join(metrics_dir, "run.log"))
    print(f"Model dir   : {model_dir}")
    print(f"Metrics dir : {metrics_dir}")
    print(f"Actions     : {action_names}")

    # --- W&B ---
    wandb_run = None
    if not args.no_wandb and wandb is not None:
        try:
            run_name = args.wandb_run_name or f"e2e_{dataset_tag}"
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name=run_name, config=vars(args),
            )
        except Exception as e:
            print(f"W&B init failed: {e}")

    # --- Data ---
    print("Loading splits...")
    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=args.threshold,
        max_samples=args.max_samples,
        base_dir=args.results_base_dir,
        label_metric=args.label_metric,
        top_k_docs=args.top_k_docs,
    )
    train_data, val_data = resplit_split_data(
        splits["train"], splits["val"],
        train_ratio=args.resplit_train_ratio, seed=args.seed,
    )
    test_data = splits["test"]
    print(f"Train: {len(train_data['labels'])}  Val: {len(val_data['labels'])}  "
          f"Test: {len(test_data['labels'])}")

    # --- Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    backbone_path = args.pretrained_backbone_path or args.model_name
    print(f"Loading backbone from {backbone_path}...")
    # right padding for QwenActionDataset; right_to_left_pad converts where the last token is pooled
    tokenizer = AutoTokenizer.from_pretrained(
        backbone_path, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id

    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_path, trust_remote_code=True, torch_dtype=torch_dtype,
    ).to(device)
    backbone.config.use_cache = False
    # Recompute activations during backward instead of storing them — reduces
    # peak memory from ~32 GB (4 chunks × 5 seqs × 28 layers) to ~2-3 GB.
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.backbone_dropout > 0:
        # attention_dropout is stored per-layer as an instance attribute; update each one.
        n_updated = 0
        for module in backbone.modules():
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = args.backbone_dropout
                n_updated += 1
        print(f"Set attention_dropout={args.backbone_dropout} on {n_updated} modules.")

    multivec = num_keep_tokens > 1

    if args.pretrained_head_path:
        print(f"Loading head from {args.pretrained_head_path}...")
        head_cls = LateInteractionHeadModel if multivec else LateInterHeadModel
        extra = {"num_keep_tokens": num_keep_tokens, "pool": args.late_interaction_pool} if multivec else {}
        head = head_cls.load_pretrained(
            args.pretrained_head_path,
            action_names=action_names,
            device=device,
            n_layers=args.cross_layers,
            n_heads=args.cross_heads,
            dropout=args.cross_dropout,
            use_type_emb=use_type_emb,
            ffn_dim=args.cross_ffn_dim,
            **extra,
        )
    elif multivec:
        head = LateInteractionHeadModel(
            hidden_size=backbone.config.hidden_size,
            num_classes=len(action_names),
            action_names=action_names,
            num_keep_tokens=num_keep_tokens,
            n_layers=args.cross_layers,
            n_heads=args.cross_heads,
            dropout=args.cross_dropout,
            use_type_emb=use_type_emb,
            ffn_dim=args.cross_ffn_dim,
            pool=args.late_interaction_pool,
        ).to(device)
    else:
        head = LateInterHeadModel(
            hidden_size=backbone.config.hidden_size,
            num_classes=len(action_names),
            action_names=action_names,
            n_layers=args.cross_layers,
            n_heads=args.cross_heads,
            dropout=args.cross_dropout,
            use_type_emb=use_type_emb,
            ffn_dim=args.cross_ffn_dim,
        ).to(device)

    print(f"Backbone params : {sum(p.numel() for p in backbone.parameters()):,}")
    print(f"Head params     : {sum(p.numel() for p in head.parameters()):,}")

    # --- Datasets & loaders ---
    def make_dataset(split_data):
        return QwenActionDataset(
            split_data, tokenizer,
            max_length=args.max_length, top_k_docs=args.top_k_docs,
            routing_prompt=True,
            track_doc_boundaries=(args.extraction_source == "document_boundaries"),
        )

    train_loader = DataLoader(make_dataset(train_data), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make_dataset(val_data), batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(make_dataset(test_data), batch_size=args.eval_batch_size, shuffle=False)

    # --- Optimizer & scheduler ---
    n_steps = len(train_loader) * args.epochs
    optimizer = AdamW([
        {"params": backbone.parameters(), "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(n_steps * args.warmup_ratio),
        num_training_steps=n_steps,
    )

    # --- Training loop ---
    best_val_f1, best_epoch, global_step = -1.0, 0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        t0 = time.time()

        train_loss, global_step = train_epoch(
            backbone=backbone, head=head, dataloader=train_loader,
            optimizer=optimizer, scheduler=scheduler, device=device,
            encode_batch_size=args.encode_batch_size,
            pad_token_id=pad_token_id, loss_mode=args.loss_mode,
            soft_target_temperature=args.soft_target_temperature,
            max_grad_norm=args.max_grad_norm, metric_lambda=args.metric_lambda,
            wandb_run=wandb_run, global_step=global_step, epoch=epoch,
            num_keep_tokens=num_keep_tokens,
            extraction_source=args.extraction_source,
        )
        elapsed = (time.time() - t0) / 60
        print(f"Train loss: {train_loss:.4f}  ({elapsed:.1f} min)")

        val_metrics = evaluate(
            backbone=backbone, head=head, dataloader=val_loader, device=device,
            action_names=action_names,
            encode_batch_size=args.encode_batch_size, pad_token_id=pad_token_id,
            num_keep_tokens=num_keep_tokens,
            extraction_source=args.extraction_source,
        )
        print_eval("Val", val_metrics)
        _wandb_log_eval(wandb_run, "val", val_metrics, epoch, global_step)

        val_f1 = val_metrics["downstream"]["selected_f1"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_selected_f1": val_f1})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            save_checkpoint(backbone, head, tokenizer,
                            os.path.join(model_dir, "best"), epoch, val_metrics)
        save_checkpoint(backbone, head, tokenizer,
                        os.path.join(model_dir, "latest"), epoch, val_metrics)

    print(f"\nBest epoch: {best_epoch}  Best val F1: {best_val_f1:.4f}")

    # --- Test eval on best checkpoint ---
    print("\nLoading best checkpoint for test eval...")
    backbone_best, head_best = load_checkpoint(
        ckpt_dir=os.path.join(model_dir, "best"),
        action_names=action_names, device=device, torch_dtype=torch_dtype,
        cross_layers=args.cross_layers, cross_heads=args.cross_heads,
        cross_dropout=args.cross_dropout, use_type_emb=use_type_emb,
        cross_ffn_dim=args.cross_ffn_dim,
        num_keep_tokens=num_keep_tokens, late_interaction_pool=args.late_interaction_pool,
    )

    test_metrics = evaluate(
        backbone=backbone_best, head=head_best, dataloader=test_loader, device=device,
        action_names=action_names,
        encode_batch_size=args.encode_batch_size, pad_token_id=pad_token_id,
        num_keep_tokens=num_keep_tokens,
        extraction_source=args.extraction_source,
    )
    print_eval("Test", test_metrics)
    _wandb_log_eval(wandb_run, "test", test_metrics, best_epoch, global_step)

    results = {
        "best_epoch": best_epoch,
        "best_val_selected_f1": best_val_f1,
        "test": test_metrics,
        "history": history,
        "args": vars(args),
    }
    results_path = os.path.join(metrics_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
