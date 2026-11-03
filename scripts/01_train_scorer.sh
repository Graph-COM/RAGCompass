#!/bin/bash
# Stage 1: fine-tune the reranker backbone as a scalar scorer, s_i = logit(yes | q, D_i),
# against the utility distribution with KL divergence. Evaluated on its own this is the
# "Scalar scorer, no head" row of the main results table.
#
# Usage: bash scripts/01_train_scorer.sh [seed] [extra args...]
#   bash scripts/01_train_scorer.sh 42
#   for s in 42 43 44; do bash scripts/01_train_scorer.sh $s; done
#
# DATASETS / RETRIEVERS / MODEL_DIR can be overridden to train a hold-out backbone
# (see scripts/holdout_fold.sh).
source "$(dirname "$0")/env.sh"
SEED="${1:-42}"
shift || true
MODEL_DIR="${MODEL_DIR:-$OUT/checkpoints/scorer_s${SEED}}"

"$PYTHON" -u -m router.train_scorer \
  --model_name "$BACKBONE_NAME" \
  --results_base_dir "$RESULTS_DIR" \
  --datasets "$DATASETS" \
  --retrievers "$RETRIEVERS" \
  --top_k 5 \
  --f1_gap_threshold 0.05 \
  --split_mode resplit_train_val \
  --resplit_train_ratio 0.7 \
  --epochs 6 \
  --max_grad_norm 1.0 \
  --lr_schedule cosine \
  --batch_size 8 \
  --eval_batch_size 8 \
  --score_batch_size 40 \
  --learning_rate 2e-5 \
  --max_length 2048 \
  --seed "$SEED" \
  --eval_train_final \
  --final_eval_checkpoint best \
  --routing_prompt \
  --runs_dir "$OUT/runs" \
  --output_dir "$MODEL_DIR" \
  "$@"
