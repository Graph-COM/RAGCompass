"""
Joint-prompt routing fine-tuning with final-position extraction.

All retrievers' docs appear in a single prompt. Instead of extracting a
hidden state at each retriever's section end (which suffers from causal
asymmetry — earlier retrievers can't see later ones), we extract the hidden
state at the very last token of the full prompt, where causal attention has
accumulated over all retrievers' content equally.

A single nn.Linear(hidden_size, num_retrievers) routing head maps this shared
final-position representation to per-retriever routing scores.

Default mode: freeze backbone (only 5k params to train), warm-start from the
contrastive FT checkpoint. Use --no-freeze_backbone for end-to-end fine-tuning.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import wandb
except ImportError:
    wandb = None

from router.shared_utils import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_INSTRUCTION,
    DEFAULT_RESULTS_BASE_DIR,
    DEFAULT_RUNS_DIR,
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    install_file_logger,
    resolve_run_dirs,
    set_seed,
)
from router.train_head import (
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)
from router.joint_prompt_scorer import (
    resplit_train_val_raw,
)

METHOD_DIR = "joint_final_pos_ft"


# ---------------------------------------------------------------------------
# Prompt building (same joint format as joint_prompt_scorer)
# ---------------------------------------------------------------------------

def build_joint_prompt_tokens(tokenizer, query, docs_per_action, action_names, top_k_docs, max_length):
    prefix = RERANKER_PREFIX + f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: {query}\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    num_actions = len(action_names)
    per_section_budget = max(16, (max_length - len(prefix_ids) - len(suffix_ids)) // num_actions)

    all_ids = list(prefix_ids)
    for i, (name, docs) in enumerate(zip(action_names, docs_per_action)):
        header_ids = tokenizer.encode(f"[Retriever {name}]\n", add_special_tokens=False)
        concat_text = "\n\n".join(f"[Document {j + 1}]\n{doc}" for j, doc in enumerate(docs[:top_k_docs]))
        doc_ids = tokenizer.encode(concat_text, add_special_tokens=False)
        doc_budget = per_section_budget - len(header_ids)
        if len(doc_ids) > doc_budget:
            doc_ids = doc_ids[:max(1, doc_budget)]
        all_ids.extend(header_ids + doc_ids)
        if i < num_actions - 1:
            all_ids.extend(tokenizer.encode("\n\n", add_special_tokens=False))

    all_ids.extend(suffix_ids)
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = [1] * len(all_ids)
    pad_len = max_length - len(all_ids)
    if pad_len > 0:
        all_ids = all_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return all_ids, attention_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JointFinalPosDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, top_k_docs, action_names,
                 shuffle_retrievers=False, seed=42):
        self.queries = data["queries"]
        self.docs_per_action = data["docs_per_action"]
        self.labels = data["labels"]
        self.f1_scores = data["f1_scores"]
        self.metric_scores = data["metric_scores"]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.top_k_docs = top_k_docs
        self.action_names = list(action_names)
        self.shuffle_retrievers = shuffle_retrievers
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        docs_per_action = self.docs_per_action[idx]
        label = self.labels[idx]
        f1s = list(self.f1_scores[idx])
        metrics = list(self.metric_scores[idx])

        C = len(self.action_names)
        order = list(range(C))
        if self.shuffle_retrievers:
            self.rng.shuffle(order)

        shuffled_docs = [docs_per_action[i] for i in order]
        shuffled_names = [self.action_names[i] for i in order]
        shuffled_f1s = [f1s[i] for i in order]
        shuffled_metrics = [metrics[i] for i in order]
        shuffled_label = order.index(label)

        ids, mask = build_joint_prompt_tokens(
            tokenizer=self.tokenizer,
            query=query,
            docs_per_action=shuffled_docs,
            action_names=shuffled_names,
            top_k_docs=self.top_k_docs,
            max_length=self.max_length,
        )

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(shuffled_label, dtype=torch.long),
            "action_f1s": torch.tensor(shuffled_f1s, dtype=torch.float),
            "action_metrics": torch.tensor(shuffled_metrics, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Routing head
# ---------------------------------------------------------------------------

class RoutingHead(nn.Module):
    """
    Linear projection from the shared final-position hidden state to per-retriever
    routing scores. Each of the num_classes output units learns a different
    direction in the 1024-dim space corresponding to "suitability of retriever j."
    """
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_size, num_classes)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_state):
        return self.linear(hidden_state)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_final_pos(backbone, routing_head, input_ids, attention_mask,
                    freeze_backbone, score_batch_size):
    """
    Single forward pass per batch. Extract hidden state at the last non-padding
    token (same position for all samples = last suffix token, after all docs).
    Apply routing head to get [B, C] scores.
    """
    B = input_ids.shape[0]
    all_scores = []

    for start in range(0, B, score_batch_size):
        end = min(start + score_batch_size, B)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        b = chunk_ids.shape[0]

        seq_lens = chunk_mask.sum(dim=1) - 1  # [b]
        batch_idx = torch.arange(b, device=chunk_ids.device)

        if freeze_backbone:
            with torch.inference_mode():
                outputs = backbone(chunk_ids, attention_mask=chunk_mask,
                                   output_hidden_states=True, use_cache=False)
                final_hidden = outputs.hidden_states[-1][batch_idx, seq_lens].detach().float()
        else:
            # Use a forward hook to capture the last hidden state without
            # output_hidden_states=True, which conflicts with gradient checkpointing.
            captured = {}

            def _hook(module, input, output):
                # output is (hidden_state,) or just hidden_state depending on the layer
                h = output[0] if isinstance(output, tuple) else output
                captured["h"] = h[batch_idx, seq_lens].float()

            # Hook onto the final layer norm (present in all decoder-only models)
            hook_handle = backbone.model.norm.register_forward_hook(_hook)
            try:
                backbone(chunk_ids, attention_mask=chunk_mask, use_cache=False)
            finally:
                hook_handle.remove()
            final_hidden = captured["h"]

        scores = routing_head(final_hidden)  # [b, C]
        all_scores.append(scores)

    return torch.cat(all_scores, dim=0)  # [B, C]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_routing_loss(scores, action_metrics, temperature, loss_mode):
    if loss_mode == "ce":
        return F.cross_entropy(scores, action_metrics.argmax(dim=-1))
    target_probs = torch.softmax(action_metrics.float() / temperature, dim=-1)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(backbone, routing_head, dataloader, device, action_names,
             freeze_backbone, score_batch_size):
    backbone.eval()
    routing_head.eval()
    all_preds, all_action_f1s = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            scores = score_final_pos(backbone, routing_head, input_ids, attention_mask,
                                     freeze_backbone=True, score_batch_size=score_batch_size)
            all_preds.extend(scores.argmax(dim=-1).cpu().tolist())
            all_action_f1s.extend(batch["action_f1s"].cpu().tolist())

    predictions = np.array(all_preds)
    action_f1s = np.array(all_action_f1s)
    row_idx = np.arange(len(predictions))
    selected_f1 = action_f1s[row_idx, predictions]
    oracle_f1 = action_f1s.max(axis=1)
    baseline_f1 = {name: float(action_f1s[:, i].mean()) for i, name in enumerate(action_names)}
    best_baseline = max(baseline_f1, key=baseline_f1.get)

    return {
        "selected_f1": float(selected_f1.mean()),
        "oracle_f1": float(oracle_f1.mean()),
        "oracle_gap": float(oracle_f1.mean() - selected_f1.mean()),
        "improvement_vs_best_baseline": float(selected_f1.mean() - baseline_f1[best_baseline]),
        "best_baseline": best_baseline,
        "best_baseline_f1": float(baseline_f1[best_baseline]),
        "per_retriever_baseline_f1": baseline_f1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(backbone, routing_head, dataloader, optimizer, scheduler,
                device, freeze_backbone, score_batch_size, temperature,
                loss_mode, grad_accum_steps, wandb_run, global_step):
    if freeze_backbone:
        backbone.eval()  # frozen — stays in eval mode throughout
    else:
        backbone.train()
    routing_head.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        action_metrics = batch["action_metrics"].to(device)

        scores = score_final_pos(backbone, routing_head, input_ids, attention_mask,
                                 freeze_backbone=freeze_backbone,
                                 score_batch_size=score_batch_size)
        loss = compute_routing_loss(scores, action_metrics, temperature, loss_mode) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(routing_head.parameters(), max_norm=1.0)
            if not freeze_backbone:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item() * grad_accum_steps}, step=global_step)

        total_loss += loss.item() * grad_accum_steps

    if len(dataloader) % grad_accum_steps:
        torch.nn.utils.clip_grad_norm_(routing_head.parameters(), max_norm=1.0)
        if not freeze_backbone:
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    return total_loss / len(dataloader), global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Joint final-position routing head fine-tuning")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="HuggingFace checkpoint dir to load backbone weights from.")
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=True,
                        help="Freeze backbone; only train the routing head (default: True).")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str, default="auto")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--top_k_docs", type=int, default=3)
    parser.add_argument("--shuffle_retrievers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resplit_train_val", action="store_true")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Higher LR is fine when only the routing head is trained.")
    parser.add_argument("--score_batch_size", type=int, default=4)
    parser.add_argument("--loss_mode", type=str, default="soft_metric_ce",
                        choices=["ce", "soft_metric_ce"])
    parser.add_argument("--soft_target_temperature", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for logs, metrics and plots.")
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help="Base directory for model checkpoints (used when --output_dir is not given).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Exact directory for this run's model checkpoint.")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    backbone_path = args.pretrained_model_path or args.model_name

    threshold_tag = "no_filtering" if not args.threshold else f"f1gap_{str(args.threshold).replace('.', '_')}"
    freeze_tag = "frozen" if args.freeze_backbone else "e2e"
    run_suffix = f"{threshold_tag}-{args.loss_mode}-T{args.soft_target_temperature}-top{args.top_k_docs}-{freeze_tag}"
    dataset_tag = "__".join(dataset_names)

    run_dirs = resolve_run_dirs(
        method_dir_name=METHOD_DIR,
        dataset_name=dataset_tag,
        seed=args.seed,
        retriever_names=action_names,
        run_name_suffix=run_suffix,
        runs_dir=args.runs_dir,
        checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
    )
    model_dir = run_dirs["model_dir"]
    log_dir = run_dirs["log_dir"]
    metrics_dir = run_dirs["metrics_dir"]

    for d in [model_dir, log_dir, metrics_dir]:
        os.makedirs(d, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    install_file_logger(log_file)

    print(f"Method: {METHOD_DIR}")
    print(f"Backbone: {backbone_path}  (frozen={args.freeze_backbone})")
    print(f"Datasets: {dataset_names}  Actions: {action_names}")
    print(f"Loss: {args.loss_mode}, T={args.soft_target_temperature}, threshold={args.threshold}")
    print(f"Checkpoint output: {model_dir}")

    print("\nLoading data splits...")
    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=args.threshold,
        max_samples=args.max_samples,
        base_dir=args.results_base_dir,
        label_metric=args.label_metric,
        top_k_docs=args.top_k_docs,
    )
    if args.resplit_train_val:
        splits["train"], splits["val"] = resplit_train_val_raw(
            splits["train"], splits["val"],
            train_ratio=args.train_ratio,
            seed=args.seed,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\nLoading backbone from {backbone_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(backbone_path, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModelForCausalLM.from_pretrained(
        backbone_path, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)

    if args.freeze_backbone:
        backbone.requires_grad_(False)
        print("Backbone frozen.")
    else:
        if hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()
        if hasattr(backbone.config, "use_cache"):
            backbone.config.use_cache = False

    hidden_size = backbone.config.hidden_size
    num_classes = len(action_names)
    routing_head = RoutingHead(hidden_size, num_classes).to(device)
    trainable_params = sum(p.numel() for p in routing_head.parameters())
    if not args.freeze_backbone:
        trainable_params += sum(p.numel() for p in backbone.parameters())
    print(f"Trainable parameters: {trainable_params:,}  "
          f"(routing head: {sum(p.numel() for p in routing_head.parameters()):,})")

    train_dataset = JointFinalPosDataset(
        splits["train"], tokenizer, args.max_length, args.top_k_docs, action_names,
        shuffle_retrievers=args.shuffle_retrievers, seed=args.seed,
    )
    val_dataset = JointFinalPosDataset(
        splits["val"], tokenizer, args.max_length, args.top_k_docs, action_names,
        shuffle_retrievers=False,
    )
    test_dataset = JointFinalPosDataset(
        splits["test"], tokenizer, args.max_length, args.top_k_docs, action_names,
        shuffle_retrievers=False,
    )
    print(f"\nSplit sizes — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    params_to_optimize = list(routing_head.parameters())
    if not args.freeze_backbone:
        params_to_optimize += list(backbone.parameters())

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wandb_run = None
    if wandb is not None:
        try:
            wandb_run = wandb.init(
                project=os.getenv("WANDB_PROJECT", "ragcompass"),
                entity=os.getenv("WANDB_ENTITY", None),
                name=run_dirs.get("run_name", run_suffix),
                config=vars(args),
                settings=wandb.Settings(init_timeout=300),
            )
        except Exception as e:
            print(f"W&B init failed: {e}")

    best_val_f1 = -float("inf")
    best_head_ckpt = os.path.join(model_dir, "best_routing_head.pt")
    best_backbone_ckpt = os.path.join(model_dir, "best_backbone.pt") if not args.freeze_backbone else None
    history = {"epochs": [], "train_loss": [], "val_selected_f1": [], "val_improvement": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_epoch(
            backbone=backbone,
            routing_head=routing_head,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            freeze_backbone=args.freeze_backbone,
            score_batch_size=args.score_batch_size,
            temperature=args.soft_target_temperature,
            loss_mode=args.loss_mode,
            grad_accum_steps=args.grad_accum_steps,
            wandb_run=wandb_run,
            global_step=global_step,
        )

        val_metrics = evaluate(backbone, routing_head, val_loader, device, action_names,
                               args.freeze_backbone, args.score_batch_size)
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_selected_f1"].append(val_metrics["selected_f1"])
        history["val_improvement"].append(val_metrics["improvement_vs_best_baseline"])

        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
            f"val_F1={val_metrics['selected_f1']:.4f} | "
            f"improvement={val_metrics['improvement_vs_best_baseline']:+.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "val/selected_f1": val_metrics["selected_f1"],
                "val/improvement_vs_best_baseline": val_metrics["improvement_vs_best_baseline"],
                "val/oracle_f1": val_metrics["oracle_f1"],
                "train/loss_epoch": train_loss,
                "epoch": epoch,
            }, step=global_step)

        if val_metrics["selected_f1"] > best_val_f1:
            best_val_f1 = val_metrics["selected_f1"]
            torch.save(routing_head.state_dict(), best_head_ckpt)
            if best_backbone_ckpt is not None:
                torch.save(backbone.state_dict(), best_backbone_ckpt)
            print(f"  → New best val F1: {best_val_f1:.4f} (epoch {epoch})")

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    routing_head.load_state_dict(torch.load(best_head_ckpt, map_location=device, weights_only=True))
    if best_backbone_ckpt and os.path.exists(best_backbone_ckpt):
        backbone.load_state_dict(torch.load(best_backbone_ckpt, map_location=device, weights_only=False))

    test_metrics = evaluate(backbone, routing_head, test_loader, device, action_names,
                            True, args.score_batch_size)

    print(f"\nTest selected F1:  {test_metrics['selected_f1']:.4f}")
    print(f"Test oracle F1:    {test_metrics['oracle_f1']:.4f}")
    print(f"Best baseline ({test_metrics['best_baseline']}): {test_metrics['best_baseline_f1']:.4f}")
    print(f"Improvement vs best baseline: {test_metrics['improvement_vs_best_baseline']:+.4f}")

    if wandb_run is not None:
        wandb_run.log({
            "test/selected_f1": test_metrics["selected_f1"],
            "test/oracle_f1": test_metrics["oracle_f1"],
            "test/improvement_vs_best_baseline": test_metrics["improvement_vs_best_baseline"],
        })
        wandb_run.finish()

    metrics_path = os.path.join(metrics_dir, "results.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "test": test_metrics,
            "best_val_f1": best_val_f1,
            "history": history,
            "args": vars(args),
        }, f, indent=2)

    print(f"\nRouting head: {best_head_ckpt}")
    print(f"Metrics:      {metrics_path}")


if __name__ == "__main__":
    main()
