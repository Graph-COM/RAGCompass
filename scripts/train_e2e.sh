#!/bin/bash
# Joint training of backbone and identity-free head, starting from the base
# reranker (the alternative to the two-stage recipe of scripts 01-03).
#
# Usage: bash scripts/train_e2e.sh [seed] [loss_mode]
#   bash scripts/train_e2e.sh 42                    listwise control (soft_metric_ce)
#   bash scripts/train_e2e.sh 42 pointwise_bce      backbone-level objective ablation
source "$(dirname "$0")/env.sh"
SEED="${1:-42}"
LOSS_MODE="${2:-soft_metric_ce}"

"$PYTHON" -u -m router.train_e2e \
  --model_name "$BACKBONE_NAME" \
  --datasets "$DATASETS" \
  --retrievers "$RETRIEVERS" \
  --results_base_dir "$RESULTS_DIR" \
  --label_metric f1 \
  --threshold 0.05 \
  --top_k_docs 5 \
  --max_length 2048 \
  --resplit_train_ratio 0.7 \
  --epochs 3 \
  --batch_size 4 \
  --eval_batch_size 4 \
  --encode_batch_size 5 \
  --backbone_lr 2e-5 \
  --head_lr 1e-4 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.05 \
  --loss_mode "$LOSS_MODE" \
  --soft_target_temperature 0.3 \
  --seed "$SEED" \
  --run_name "e2e_${LOSS_MODE}_s${SEED}" \
  --no_type_emb \
  --no_wandb \
  --runs_dir "$OUT/runs" \
  --checkpoints_dir "$OUT/checkpoints"
