"""
Inject retrieval-relevance (nDCG) targets into an existing embedding cache.

Purpose: the method supervises routing with DOWNSTREAM ANSWER UTILITY
(per-retriever answer F1) rather than retrieval relevance, on the argument that
passages can be topically relevant yet fail to support an answer. This script
enables the controlled ablation of that choice: train the identical model against nDCG-vs-gold
instead of answer F1, and evaluate on downstream F1 either way.

Design points that keep the ablation clean:
  * nDCG is written to a NEW key, "action_ndcg" -- it does NOT overwrite
    "action_metrics". The F1-gap example filter runs off action_metrics, so
    both arms of the ablation see an IDENTICAL example set and identical
    evaluation; only the training target differs (--loss_target_key).
  * Rows are matched BY ID, never by positional assumption, and the script
    fails loudly if any cached id is missing a computed nDCG.
  * Writes to a copy, never in place (the caller supplies a fresh --cache_dir).

nDCG comes from load_action_splits(label_metric="ndcg"), which computes it via
src.metric.get_ndcg from the gold_doc_list already present in the artifacts, so
no re-retrieval and no GPU work is required.

Usage:
  python -m router.inject_ndcg_targets \
      --cache_dir <cache to augment, a copy> \
      --results_base_dir Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B
"""

import argparse
import json
import os

import torch

from router.train_head import (
    DEFAULT_RESULTS_BASE_DIR,
    discover_actions,
    load_action_splits,
    parse_dataset_names,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--datasets", type=str, default="nq,musique,hotpotqa,2wikimultihopqa")
    p.add_argument("--retrievers", type=str,
                   default="rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step")
    p.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    p.add_argument("--top_k_docs", type=int, default=5)
    p.add_argument("--splits", type=str, default="train,val,test")
    args = p.parse_args()

    dataset_names = parse_dataset_names(args.datasets, None)
    action_names = discover_actions(args.results_base_dir, dataset_names, args.retrievers)
    print(f"Datasets: {dataset_names}")
    print(f"Actions:  {action_names}")

    # threshold=0.0 -> no filtering, so every cached row has a counterpart here.
    print("\nComputing nDCG targets from gold_doc_list (no retrieval, no GPU)...")
    splits = load_action_splits(
        dataset_names=dataset_names,
        action_names=action_names,
        threshold=0.0,
        max_samples=None,
        base_dir=args.results_base_dir,
        label_metric="ndcg",
        top_k_docs=args.top_k_docs,
    )

    for split_name in args.splits.split(","):
        cache_path = os.path.join(args.cache_dir, f"{split_name}.pt")
        if not os.path.exists(cache_path):
            print(f"{split_name}: no cache file, skipping")
            continue
        if split_name not in splits:
            print(f"{split_name}: no computed split, skipping")
            continue

        ndcg_by_id = {
            str(_id): scores
            for _id, scores in zip(splits[split_name]["ids"], splits[split_name]["metric_scores"])
        }

        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        if "action_ndcg" in data:
            print(f"{split_name}: already has action_ndcg, leaving untouched")
            continue

        cached_ids = [str(i) for i in data["ids"]]
        missing = [i for i in cached_ids if i not in ndcg_by_id]
        if missing:
            raise ValueError(
                f"{split_name}: {len(missing)} cached ids have no computed nDCG "
                f"(first few: {missing[:5]}). Refusing to write a partially-aligned target."
            )

        aligned = torch.tensor([ndcg_by_id[i] for i in cached_ids], dtype=torch.float)
        if aligned.shape != data["action_metrics"].shape:
            raise ValueError(
                f"{split_name}: nDCG shape {tuple(aligned.shape)} != action_metrics shape "
                f"{tuple(data['action_metrics'].shape)}"
            )

        # Sanity signal: how different is the relevance target from the utility
        # target? If they were near-identical the ablation would be vacuous.
        f1 = data["action_metrics"].float()
        agree = (aligned.argmax(dim=1) == f1.argmax(dim=1)).float().mean().item()
        corr = torch.corrcoef(torch.stack([aligned.flatten(), f1.flatten()]))[0, 1].item()
        print(f"{split_name}: argmax agreement with utility target = {agree:.3f}, "
              f"elementwise corr = {corr:.3f}")

        data["action_ndcg"] = aligned
        torch.save(data, cache_path)
        print(f"{split_name}: added action_ndcg {tuple(aligned.shape)} -> {cache_path}")

    meta_path = os.path.join(args.cache_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        meta["has_action_ndcg"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print("\nmetadata.json updated (has_action_ndcg=True)")


if __name__ == "__main__":
    main()
