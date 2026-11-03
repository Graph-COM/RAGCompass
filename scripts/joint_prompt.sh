#!/bin/bash
# Joint-prompt encoding ablation: all R retrievers' passages in ONE prompt and
# one forward pass, instead of one forward pass per retriever.
#
# Usage: bash scripts/joint_prompt.sh <variant>
#   yes_logit        end-to-end; a yes-logit read at the end of each retriever's section
#   final_frozen     frozen backbone; linear head on the final-token hidden state
#   final_e2e        end-to-end;     linear head on the final-token hidden state
# The two final-token variants warm-start from the Stage-1 scorer ($SCORER_DIR).
source "$(dirname "$0")/env.sh"
VARIANT="${1:?Usage: joint_prompt.sh yes_logit|final_frozen|final_e2e}"

COMMON=(
  --model_name "$BACKBONE_NAME"
  --results_base_dir "$RESULTS_DIR"
  --datasets "$DATASETS"
  --retrievers "$RETRIEVERS"
  --label_metric f1
  --threshold 0.05
  --max_length 4096
  --top_k_docs 3
  --shuffle_retrievers
  --resplit_train_val
  --loss_mode soft_metric_ce
  --soft_target_temperature 0.3
  --batch_size 1
  --grad_accum_steps 8
  --learning_rate 1e-5
  --score_batch_size 1
  --warmup_ratio 0.1
  --seed 42
  --runs_dir "$OUT/runs"
  --checkpoints_dir "$OUT/checkpoints"
)

case "$VARIANT" in
  yes_logit)
    "$PYTHON" -u -m router.joint_prompt_scorer "${COMMON[@]}" --train_ratio 0.9 --epochs 15 ;;
  final_frozen)
    "$PYTHON" -u -m router.joint_prompt_final_pos "${COMMON[@]}" --pretrained_model_path "$SCORER_DIR" \
      --freeze_backbone --train_ratio 0.8 --epochs 20 ;;
  final_e2e)
    "$PYTHON" -u -m router.joint_prompt_final_pos "${COMMON[@]}" --pretrained_model_path "$SCORER_DIR" \
      --no-freeze_backbone --train_ratio 0.8 --epochs 10 ;;
  *) echo "unknown variant: $VARIANT"; exit 1 ;;
esac
