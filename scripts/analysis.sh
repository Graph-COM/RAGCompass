#!/bin/bash
# Analyses of the reported system (frozen backbone + comparison head).
#
# Usage: bash scripts/analysis.sh [seed]          needs head run main_s<seed> from 03_train_head.sh
#
#   1. per-query routing decisions + per-dataset breakdown
#   2. recall@5 / nDCG@5 of the routed choice (also for RRF / MoR)
#   3. chosen-vs-oracle routing distribution
#   4. oracle-gap decomposition on the UNFILTERED test set
#
# Run separately (see the bottom of this file):
#   bash scripts/analysis.sh representation       what the scorer's representations retain
#   bash scripts/analysis.sh latency              per-query cost of a routing decision (GPU)
source "$(dirname "$0")/env.sh"

if [[ "${1:-}" == "representation" ]]; then
  # Single-vector cache of the scalar scorer (one pooled hidden state + yes-logit per retriever).
  SV_CACHE="$OUT/cache/single_vector"
  "$PYTHON" -u -m router.cache_embeddings \
    --model_name "$BACKBONE_NAME" --pretrained_model_path "$SCORER_DIR" \
    --datasets "$DATASETS" --retrievers "$RETRIEVERS" --results_base_dir "$RESULTS_DIR" \
    --label_metric f1 --threshold 0.05 --max_length 2048 --top_k_docs 5 \
    --batch_size 1 --encode_batch_size 16 --routing_prompt \
    --cache_dir "$SV_CACHE"
  "$PYTHON" -u -m analysis.representation_diagnostics --cache_dir "$SV_CACHE" \
    --output_json "$DIAG_DIR/representation_diagnostics.json"
  exit 0
fi

if [[ "${1:-}" == "latency" ]]; then
  "$PYTHON" -u -m analysis.measure_latency \
    --pretrained_model_path "$SCORER_DIR" --head_dir "$OUT/checkpoints/head_main_s42" \
    --datasets "$DATASETS" --retrievers "$RETRIEVERS" --results_base_dir "$RESULTS_DIR" \
    --n_queries "${N_QUERIES:-200}" --output_json "$DIAG_DIR/routing_latency.json"
  exit 0
fi

SEED="${1:-42}"
HEAD="${HEAD_DIR:-$OUT/checkpoints/head_main_s${SEED}}"
DEC="$DIAG_DIR/routing_decisions_s${SEED}.jsonl"

echo "### 1/4 per-query decisions + per-dataset breakdown"
"$PYTHON" -u -m analysis.dump_decisions --cache_dir "$CACHE_DIR" --head_dir "$HEAD" --output "$DEC"

echo "### 2/4 retrieval metrics of the routed choice"
"$PYTHON" -u -m analysis.retrieval_metrics --decisions "$DEC" --results_base_dir "$RESULTS_DIR" \
  --retrievers "$RETRIEVERS" --output_json "$DIAG_DIR/retrieval_metrics_s${SEED}.json"

echo "### 3/4 routing-decision distribution"
"$PYTHON" -u -m analysis.routing_distribution --input "$DEC" --output_dir "$DIAG_DIR/figures" \
  --output_json "$DIAG_DIR/routing_distribution_s${SEED}.json"

echo "### 4/4 oracle-gap decomposition (unfiltered test set)"
"$PYTHON" -u -m analysis.dump_decisions --cache_dir "$CACHE_DIR" --head_dir "$HEAD" \
  --threshold 0.0 --skip_gate --output "$DIAG_DIR/routing_decisions_s${SEED}_unfiltered.jsonl"
"$PYTHON" -u -m analysis.oracle_gap_decomposition \
  --input "$DIAG_DIR/routing_decisions_s${SEED}_unfiltered.jsonl" \
  --output_json "$DIAG_DIR/oracle_gap_decomposition_s${SEED}.json"
