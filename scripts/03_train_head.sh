#!/bin/bash
# Stage 3: train the comparison head (two-layer transformer across the R
# retrievers, no retriever-specific parameters) on the cached representations.
# With no extra arguments this is the reported system ("+ comparison head").
#
# Usage: bash scripts/03_train_head.sh <run_name> [extra args...]
#   bash scripts/03_train_head.sh main_s42 --seed 42
#
# Extra arguments override the defaults below; the ones the paper uses:
#   --holdout_dataset <d> [--holdout_dataset_test_on holdout|rest]   leave-one-dataset-out
#   --eval_dataset <d>                                              in-distribution control on one dataset
#   --holdout_action <r>                                            leave-one-retriever-out
#   --use_type_emb                                                  add a per-retriever identity embedding
#   --loss_mode pointwise_bce                                       no shared normalization across retrievers
#   --loss_target_key action_ndcg                                   supervise with nDCG instead of answer F1
#   --test_threshold 0.0                                            evaluate on unfiltered test queries
source "$(dirname "$0")/env.sh"
RUN="${1:?Usage: 03_train_head.sh <run_name> [extra args...]}"
shift

"$PYTHON" -u -m router.train_head \
  --freeze_base \
  --embedding_cache_dir "$CACHE_DIR" \
  --results_base_dir "$RESULTS_DIR" \
  --datasets "$DATASETS" \
  --retrievers "$RETRIEVERS" \
  --top_k_docs 5 \
  --threshold 0.05 \
  --model_type late_interaction \
  --late_interaction_pool mean \
  --no-use_type_emb \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --loss_mode soft_metric_ce \
  --loss_target_key action_metrics \
  --soft_target_temperature 0.1 \
  --max_length 2048 \
  --cross_layers 2 \
  --cross_heads 8 \
  --cross_dropout 0.0 \
  --seed 42 \
  --run_name "$RUN" \
  --runs_dir "$OUT/runs" \
  --output_dir "$OUT/checkpoints/head_${RUN}" \
  "$@"
