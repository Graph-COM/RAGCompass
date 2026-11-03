import argparse
import json
import os
import time

import torch

# PyTorch SDPA can select a cuDNN attention backend with no valid execution plan
# on some GPU/cuDNN/torch combinations, failing with
#   "cuDNN Frontend error: [cudnn_frontend] Error: No valid execution plans built"
# This module owns the encode path that
# every backbone forward goes through -- the cache builder and anything importing
# encode_flat_* -- so the fix belongs here rather than being re-applied per
# script. Only the cuDNN backend is disabled; the other SDPA backends are fine.
if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from router.train_head import (
    DEFAULT_EMBEDDING_CACHE_BASE_DIR,
    DEFAULT_RESULTS_BASE_DIR,
    QwenActionDataset,
    default_embedding_cache_dir,
    discover_actions,
    load_action_splits,
    parse_dataset_names,
    set_seed,
)


def last_token_pool(last_hidden, attention_mask):
    lengths = attention_mask.sum(dim=1)
    idx = (lengths - 1).clamp(min=0)
    batch_size, _, hidden_size = last_hidden.size()
    idx_expanded = idx.view(batch_size, 1, 1).expand(batch_size, 1, hidden_size)
    return last_hidden.gather(1, idx_expanded).squeeze(1)


def encode_flat(base_model, input_ids, attention_mask, encode_batch_size, token_true_id=None):
    encoded_chunks = []
    yes_logit_chunks = []
    for start_idx in range(0, input_ids.size(0), encode_batch_size):
        end_idx = start_idx + encode_batch_size
        chunk_ids = input_ids[start_idx:end_idx]
        chunk_mask = attention_mask[start_idx:end_idx]
        outputs = base_model(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        encoded_chunks.append(last_token_pool(outputs.hidden_states[-1], chunk_mask))
        if token_true_id is not None:
            lengths = chunk_mask.sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            yes_logit_chunks.append(outputs.logits[torch.arange(chunk_ids.size(0), device=chunk_ids.device), last_idx, token_true_id])
        del outputs
    embeddings = torch.cat(encoded_chunks, dim=0)
    yes_logits = torch.cat(yes_logit_chunks, dim=0) if yes_logit_chunks else None
    return embeddings, yes_logits


def encode_flat_document_boundaries(base_model, input_ids, attention_mask, doc_end_offsets, encode_batch_size, token_true_id=None):
    """
    Single forward pass. Extract hidden states at each retained document's
    own last-token position (doc_end_offsets), one vector per document.
    Every position causally attends to the query and all preceding
    documents, so each vector is a contextual summary "as of that document".

    When token_true_id is given, also reads a per-document "yes" logit at
    each of the same K positions -- the model's own relevance judgment as of
    that document, from the already-computed outputs.logits (no extra
    forward pass). This makes a per-document version of that signal available
    to LateInteractionHeadModel (--use_yes_logit).

    Returns (embeddings [N, top_k_docs, H], yes_logits [N, top_k_docs] or None).
    """
    N = input_ids.size(0)
    emb_chunks = []
    yes_logit_chunks = []
    for start in range(0, N, encode_batch_size):
        end = min(start + encode_batch_size, N)
        chunk_ids = input_ids[start:end].to(base_model.device)
        chunk_mask = attention_mask[start:end].to(base_model.device)
        chunk_offsets = doc_end_offsets[start:end].to(base_model.device)  # [b, K]

        outputs = base_model(
            input_ids=chunk_ids,
            attention_mask=chunk_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]  # [b, L, H]
        H = last_hidden.shape[-1]
        idx = chunk_offsets.clamp(max=last_hidden.shape[1] - 1)  # [b, K]
        window = last_hidden.gather(1, idx.unsqueeze(-1).expand(-1, -1, H))  # [b, K, H]
        emb_chunks.append(window.cpu())

        if token_true_id is not None:
            yl = outputs.logits.gather(1, idx.unsqueeze(-1).expand(-1, -1, outputs.logits.shape[-1]))
            yl = yl[..., token_true_id].float()  # [b, K]
            yes_logit_chunks.append(yl.cpu())

        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    yes_logits = torch.cat(yes_logit_chunks, dim=0) if yes_logit_chunks else None
    return torch.cat(emb_chunks, dim=0), yes_logits  # [N, K, H], [N, K] or None


def cache_dtype_from_name(dtype_name):
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported cache dtype: {dtype_name}")


def build_split_cache(split_name, split_data, tokenizer, base_model, args, device, token_true_id=None):
    output_path = os.path.join(args.cache_dir, f"{split_name}.pt")
    if os.path.exists(output_path) and not args.overwrite:
        print(f"{split_name}: cache exists, skipping: {output_path}")
        return

    dataset = QwenActionDataset(
        split_data,
        tokenizer,
        max_length=args.max_length,
        top_k_docs=args.top_k_docs,
        routing_prompt=args.routing_prompt,
        track_doc_boundaries=(args.extraction_source == "document_boundaries"),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    cache_dtype = cache_dtype_from_name(args.cache_dtype)

    all_embeddings = []
    all_yes_logits = []
    all_labels = []
    all_action_f1s = []
    all_action_metrics = []

    started = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Caching {split_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size, num_actions, seq_len = input_ids.shape
            flat_input_ids = input_ids.view(batch_size * num_actions, seq_len)
            flat_attention_mask = attention_mask.view(batch_size * num_actions, seq_len)

            if args.extraction_source == "document_boundaries":
                flat_doc_end_offsets = batch["doc_end_offsets"].to(device).view(batch_size * num_actions, -1)
                encoded, yes_logits = encode_flat_document_boundaries(
                    base_model=base_model,
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,
                    doc_end_offsets=flat_doc_end_offsets,
                    encode_batch_size=args.encode_batch_size,
                    token_true_id=token_true_id if args.cache_per_document_yes_logits else None,
                )
                encoded = encoded.view(batch_size, num_actions, args.num_keep_tokens, -1)
                if args.num_keep_tokens == 1:
                    encoded = encoded.squeeze(2)
            else:
                encoded, yes_logits = encode_flat(
                    base_model=base_model,
                    input_ids=flat_input_ids,
                    attention_mask=flat_attention_mask,
                    encode_batch_size=args.encode_batch_size,
                    token_true_id=token_true_id,
                )
                encoded = encoded.view(batch_size, num_actions, -1)

            all_embeddings.append(encoded.to(cache_dtype).cpu())
            if yes_logits is not None:
                # yes_logits is [B*C] (scalar, last_token path) or
                # [B*C, K] (per-document, document_boundaries path) -- reshape
                # generically so both cases land as [B, C, ...] correctly.
                all_yes_logits.append(yes_logits.view(batch_size, num_actions, *yes_logits.shape[1:]).float().cpu())
            all_labels.append(batch["label"].cpu())
            all_action_f1s.append(batch["action_f1s"].cpu())
            all_action_metrics.append(batch["action_metrics"].cpu())

            del input_ids, attention_mask, flat_input_ids, flat_attention_mask, encoded, yes_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    payload = {
        "embeddings": torch.cat(all_embeddings, dim=0),
        "labels": torch.cat(all_labels, dim=0),
        "action_f1s": torch.cat(all_action_f1s, dim=0),
        "action_metrics": torch.cat(all_action_metrics, dim=0),
        "ids": split_data["ids"],
        "dataset_names": split_data["dataset_names"],
    }
    if all_yes_logits:
        payload["yes_logits"] = torch.cat(all_yes_logits, dim=0)
    torch.save(payload, output_path)
    elapsed = time.time() - started
    print(
        f"{split_name}: saved {len(dataset)} samples to {output_path} "
        f"in {elapsed / 60:.1f} min; embeddings shape={tuple(payload['embeddings'].shape)}"
        + (f"; yes_logits shape={tuple(payload['yes_logits'].shape)}" if all_yes_logits else "")
    )


def main():
    parser = argparse.ArgumentParser(description="Stage 2: cache the frozen backbone's per-retriever representations")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--pretrained_model_path", type=str, default=None)
    parser.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--retrievers", type=str, default="auto")
    parser.add_argument("--actions", type=str, default=None)
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--label_metric", type=str, default="f1", choices=["ndcg", "f1"])
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--cache_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--cache_base_dir", type=str, default=DEFAULT_EMBEDDING_CACHE_BASE_DIR)
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="Which splits to build. Use 'test' for an eval-only "
                             "cache, e.g. a retrieval-depth shift.")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--routing_prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the routing-focused system prompt and instruction (ROUTING_*) instead of the "
             "reranker's default relevance prompt.",
    )
    parser.add_argument(
        "--extraction_source",
        type=str,
        default="last_token",
        choices=["last_token", "document_boundaries"],
        help="last_token (default): one pooled hidden state per retriever, at the last prompt token; "
             "cache shape [N, C, H]. document_boundaries (used in the paper): one hidden state per "
             "retrieved passage, at each passage's own last-token position; cache shape "
             "[N, C, top_k_docs, H].",
    )
    parser.add_argument(
        "--cache_per_document_yes_logits",
        action="store_true",
        help="Only used with --extraction_source document_boundaries. Also cache a per-document "
             "'yes' logit (read from the same forward pass's outputs.logits at each document's "
             "end-token position, no extra compute) alongside the K embeddings, shape [N,C,K]. "
             "Off by default.",
    )
    args = parser.parse_args()
    # Hidden states kept per retriever: one per retrieved passage, or a single pooled vector.
    args.num_keep_tokens = args.top_k_docs if args.extraction_source == "document_boundaries" else 1
    if args.cache_per_document_yes_logits and args.extraction_source != "document_boundaries":
        parser.error("--cache_per_document_yes_logits is only supported with --extraction_source document_boundaries")

    set_seed(args.seed)
    dataset_names = parse_dataset_names(args.datasets, args.dataset)
    action_arg = args.actions if args.actions is not None else args.retrievers
    action_names = discover_actions(args.results_base_dir, dataset_names, action_arg)
    model_path = args.pretrained_model_path if args.pretrained_model_path else args.model_name

    if args.cache_dir is None:
        args.cache_dir = default_embedding_cache_dir(
            base_dir=args.cache_base_dir,
            dataset_names=dataset_names,
            action_names=action_names,
            label_metric=args.label_metric,
            top_k_docs=args.top_k_docs,
            max_length=args.max_length,
            model_name=model_path,
        )
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Datasets: {dataset_names}")
    print(f"Actions: {action_names}")
    print(f"Model path: {model_path}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Cache dtype: {args.cache_dtype}")

    print("Loading action datasets...")
    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=args.threshold,
        max_samples=args.max_samples,
        base_dir=args.results_base_dir,
        label_metric=args.label_metric,
        top_k_docs=args.top_k_docs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"Loading tokenizer/model on {device} with dtype={torch_dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_true_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    print(f"Yes-token id: {token_true_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    base_model.eval()
    if hasattr(base_model.config, "use_cache"):
        base_model.config.use_cache = False

    metadata = {
        "cache_format": "late_inter_qwen_embeddings_v1",
        "model_name": args.model_name,
        "model_path": model_path,
        "hidden_size": int(base_model.config.hidden_size),
        "datasets": dataset_names,
        "action_names": action_names,
        "results_base_dir": args.results_base_dir,
        "label_metric": args.label_metric,
        "threshold": args.threshold,
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "top_k_docs": args.top_k_docs,
        "num_keep_tokens": args.num_keep_tokens,
        "extraction_source": args.extraction_source,
        "cache_per_document_yes_logits": args.cache_per_document_yes_logits,
        "cache_dtype": args.cache_dtype,
        "split_sizes": {split_name: len(split_data["labels"]) for split_name, split_data in splits.items()},
    }
    with open(os.path.join(args.cache_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # A depth study only needs test.pt (dump_decisions.py reads
    # nothing else), and rebuilding train/val at a depth the model was not
    # trained on would be both wasteful and misleading.
    for split_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        build_split_cache(
            split_name=split_name,
            split_data=splits[split_name],
            tokenizer=tokenizer,
            base_model=base_model,
            args=args,
            device=device,
            token_true_id=token_true_id,
        )

    print("Embedding cache complete.")


if __name__ == "__main__":
    main()
