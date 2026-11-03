#!/bin/bash
# Shared settings, sourced by every script in this directory. Override any of
# them from the environment, e.g.  OUT=/big/disk/ragcompass bash scripts/01_train_scorer.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHON="${PYTHON:-python}"
# Per-retriever results: $RESULTS_DIR/<dataset>[_train|_val]/<retriever>.jsonl (see README, "Data layout").
export RESULTS_DIR="${RESULTS_DIR:-Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B}"
# Everything this repository writes (checkpoints, caches, metrics) goes under $OUT.
export OUT="${OUT:-outputs}"

export DATASETS="${DATASETS:-nq,musique,hotpotqa,2wikimultihopqa}"
export RETRIEVERS="${RETRIEVERS:-rag_1_step,bm25_1_step,splade_v3_1_step,colbertv2_1_step,hipporag2-0.5_1_step}"
export BACKBONE_NAME="${BACKBONE_NAME:-Qwen/Qwen3-Reranker-0.6B}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# Default locations of the reported system's three artifacts.
export SCORER_DIR="${SCORER_DIR:-$OUT/checkpoints/scorer_s42}"
export CACHE_DIR="${CACHE_DIR:-$OUT/cache/main}"
export DIAG_DIR="${DIAG_DIR:-$OUT/diagnostics}"
mkdir -p "$OUT" "$DIAG_DIR"
