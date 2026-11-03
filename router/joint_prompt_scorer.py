"""
Joint-prompt contrastive fine-tuning of the Qwen3-Reranker backbone.

All retrievers' docs appear in a single prompt. Yes-logits are extracted at
each retriever's section-end position, giving the backbone cross-retriever
context during both training and inference. Soft F1 targets replace hard
argmax labels to teach graded routing preferences.
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
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

METHOD_DIR = "joint_contrastive_ft"


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def resplit_train_val_raw(train_data, val_data, train_ratio=0.9, seed=42):
    """Merge train+val and resplit at train_ratio. Test set is never touched."""
    combined = {}
    for key in train_data:
        a, b = train_data[key], val_data[key]
        combined[key] = a + b  # both are plain lists

    N = len(combined["queries"])
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    n_train = int(N * train_ratio)
    train_idx = order[:n_train]
    val_idx = order[n_train:]

    def subset(data, idx):
        return {k: [data[k][i] for i in idx] for k in data}

    new_train = subset(combined, train_idx)
    new_val = subset(combined, val_idx)
    print(f"Resplit {N} samples at {train_ratio:.0%}/{1-train_ratio:.0%}: "
          f"train={len(new_train['queries'])}, val={len(new_val['queries'])}")
    return new_train, new_val


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_joint_prompt_tokens(tokenizer, query, docs_per_action, action_names, top_k_docs, max_length):
    """
    Single prompt with all retrievers' doc sections.
    Returns (input_ids, attention_mask, retriever_end_positions).
    retriever_end_positions[j] = token index of the last doc token for retriever j.
    With causal attention, each position attends to all preceding retrievers' docs.
    """
    prefix = RERANKER_PREFIX + f"<Instruct>: {DEFAULT_INSTRUCTION}\n<Query>: {query}\n\n"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False)
    num_actions = len(action_names)
    per_section_budget = max(16, (max_length - len(prefix_ids) - len(suffix_ids)) // num_actions)

    all_ids = list(prefix_ids)
    retriever_end_positions = []

    for i, (name, docs) in enumerate(zip(action_names, docs_per_action)):
        header_ids = tokenizer.encode(f"[Retriever {name}]\n", add_special_tokens=False)
        concat_text = "\n\n".join(f"[Document {j + 1}]\n{doc}" for j, doc in enumerate(docs[:top_k_docs]))
        doc_ids = tokenizer.encode(concat_text, add_special_tokens=False)
        doc_budget = per_section_budget - len(header_ids)
        if len(doc_ids) > doc_budget:
            doc_ids = doc_ids[:max(1, doc_budget)]
        all_ids.extend(header_ids + doc_ids)
        retriever_end_positions.append(len(all_ids) - 1)
        if i < num_actions - 1:
            all_ids.extend(tokenizer.encode("\n\n", add_special_tokens=False))

    all_ids.extend(suffix_ids)
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
        retriever_end_positions = [min(p, max_length - 1) for p in retriever_end_positions]

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_mask = [1] * len(all_ids)
    pad_len = max_length - len(all_ids)
    if pad_len > 0:
        all_ids = all_ids + [pad_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return all_ids, attention_mask, retriever_end_positions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JointContrastiveDataset(Dataset):
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

        ids, mask, end_positions = build_joint_prompt_tokens(
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
            "retriever_end_positions": torch.tensor(end_positions, dtype=torch.long),
            "label": torch.tensor(shuffled_label, dtype=torch.long),
            "action_f1s": torch.tensor(shuffled_f1s, dtype=torch.float),
            "action_metrics": torch.tensor(shuffled_metrics, dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# Scoring and loss
# ---------------------------------------------------------------------------

def score_joint_candidates(model, input_ids, attention_mask, retriever_end_positions,
                           token_true_id, score_batch_size):
    """
    Single forward pass. Extract yes-logit at each retriever's end position.
    Returns [B, C] scores.
    """
    B = input_ids.shape[0]
    C = retriever_end_positions.shape[1]
    all_scores = []

    for start in range(0, B, score_batch_size):
        end = min(start + score_batch_size, B)
        chunk_ids = input_ids[start:end]
        chunk_mask = attention_mask[start:end]
        chunk_pos = retriever_end_positions[start:end]  # [b, C]

        outputs = model(input_ids=chunk_ids, attention_mask=chunk_mask, use_cache=False)
        logits = outputs.logits  # [b, seq_len, vocab]
        b = chunk_ids.shape[0]
        batch_idx = torch.arange(b, device=logits.device).unsqueeze(1).expand(b, C)
        all_scores.append(logits[batch_idx, chunk_pos, token_true_id].float())
        del outputs

    return torch.cat(all_scores, dim=0)  # [B, C]


def compute_routing_loss(scores, action_metrics, temperature, loss_mode):
    if loss_mode == "ce":
        labels = action_metrics.argmax(dim=-1)
        return F.cross_entropy(scores, labels)
    target_probs = torch.softmax(action_metrics.float() / temperature, dim=-1)
    log_probs = torch.log_softmax(scores.float(), dim=-1)
    return F.kl_div(log_probs, target_probs, reduction="batchmean")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, action_names, token_true_id, score_batch_size):
    model.eval()
    all_preds, all_action_f1s = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            end_positions = batch["retriever_end_positions"].to(device)
            scores = score_joint_candidates(
                model, input_ids, attention_mask, end_positions, token_true_id, score_batch_size
            )
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

def train_epoch(model, dataloader, optimizer, scheduler, device, token_true_id,
                score_batch_size, temperature, loss_mode, grad_accum_steps, wandb_run, global_step):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        end_positions = batch["retriever_end_positions"].to(device)
        action_metrics = batch["action_metrics"].to(device)

        scores = score_joint_candidates(
            model, input_ids, attention_mask, end_positions, token_true_id, score_batch_size
        )
        loss = compute_routing_loss(scores, action_metrics, temperature, loss_mode) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item() * grad_accum_steps}, step=global_step)

        total_loss += loss.item() * grad_accum_steps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # flush any leftover gradient accumulation
    if len(dataloader) % grad_accum_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    return total_loss / len(dataloader), global_step


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Joint-prompt contrastive backbone fine-tuning for routing")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--pretrained_model_path", type=str, default=None,
                        help="Start from an existing checkpoint instead of the base model.")
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str, default="auto")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--top_k_docs", type=int, default=3)
    parser.add_argument("--shuffle_retrievers", action=argparse.BooleanOptionalAction, default=True,
                        help="Shuffle retriever order per sample during training to break positional bias.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--score_batch_size", type=int, default=1)
    parser.add_argument("--loss_mode", type=str, default="soft_metric_ce",
                        choices=["ce", "soft_metric_ce"])
    parser.add_argument("--soft_target_temperature", type=float, default=0.3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs_dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="Base directory for logs, metrics and plots.")
    parser.add_argument("--checkpoints_dir", type=str, default=DEFAULT_CHECKPOINTS_DIR,
                        help="Base directory for model checkpoints (used when --output_dir is not given).")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Exact directory for this run's model checkpoint.")
    parser.add_argument("--resplit_train_val", action="store_true",
                        help="Merge train+val and resplit at --train_ratio. Test set is unchanged.")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Fraction of combined train+val to use as training set when --resplit_train_val.")
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    model_path = args.pretrained_model_path or args.model_name

    threshold_tag = "no_filtering" if not args.threshold else f"f1gap_{str(args.threshold).replace('.', '_')}"
    run_suffix = f"{threshold_tag}-{args.loss_mode}-T{args.soft_target_temperature}-top{args.top_k_docs}"
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
    plots_dir = run_dirs["plots_dir"]

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    install_file_logger(log_file)

    print(f"Method: {METHOD_DIR}")
    print(f"Datasets: {dataset_names}")
    print(f"Actions: {action_names}")
    print(f"Model: {model_path}")
    print(f"Checkpoint output: {model_dir}")
    print(f"Loss: {args.loss_mode}, T={args.soft_target_temperature}, threshold={args.threshold}")
    print(f"Shuffle retrievers during training: {args.shuffle_retrievers}")

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

    print(f"\nLoading model architecture from {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_true_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    print(f"Yes-token id: {token_true_id}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, torch_dtype=torch_dtype
    ).to(device)

    # Load pretrained weights if a checkpoint directory or .bin file is given.
    if args.pretrained_model_path is not None:
        if os.path.isdir(args.pretrained_model_path):
            state_dict_path = os.path.join(args.pretrained_model_path, "best_model.bin")
        else:
            state_dict_path = args.pretrained_model_path
        print(f"Loading weights from {state_dict_path} ...")
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        del state_dict

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_dataset = JointContrastiveDataset(
        splits["train"], tokenizer, args.max_length, args.top_k_docs, action_names,
        shuffle_retrievers=args.shuffle_retrievers, seed=args.seed,
    )
    val_dataset = JointContrastiveDataset(
        splits["val"], tokenizer, args.max_length, args.top_k_docs, action_names,
        shuffle_retrievers=False,
    )
    test_dataset = JointContrastiveDataset(
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

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
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
    best_checkpoint = os.path.join(model_dir, "best_model.bin")
    history = {"epochs": [], "train_loss": [], "val_selected_f1": [], "val_improvement": []}
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            token_true_id=token_true_id,
            score_batch_size=args.score_batch_size,
            temperature=args.soft_target_temperature,
            loss_mode=args.loss_mode,
            grad_accum_steps=args.grad_accum_steps,
            wandb_run=wandb_run,
            global_step=global_step,
        )

        val_metrics = evaluate(
            model, val_loader, device, action_names, token_true_id, args.score_batch_size
        )
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
            torch.save(model.state_dict(), best_checkpoint)
            tokenizer.save_pretrained(model_dir)
            model.config.save_pretrained(model_dir)
            print(f"  → New best val F1: {best_val_f1:.4f} (epoch {epoch})")

    print(f"\nBest validation F1: {best_val_f1:.4f}")
    model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=False))
    test_metrics = evaluate(
        model, test_loader, device, action_names, token_true_id, args.score_batch_size
    )

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

    print(f"\nCheckpoint: {model_dir}")
    print(f"Metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
