#!/bin/bash
# Head-level controls and ablations. All share the main cache ($CACHE_DIR) and
# differ from the reported head by exactly the flags shown.
#
# Usage: bash scripts/ablations.sh <name> [dataset] [seed]
#   control            the reported head (no change)
#   pointwise          each retriever fit independently, no shared normalization
#   ndcg               supervised by nDCG against gold passages instead of answer F1
#   typeemb            adds a learned per-retriever identity embedding
#   unfiltered         evaluated on every test query, not only the routable ones
#   indist  <dataset>  trained on all four datasets, evaluated on <dataset> only
#                      (the in-distribution control for leave-one-dataset-out)
#   datasize <dataset> trained without <dataset>, evaluated on the other three
#                      (training-set-size ablation)
source "$(dirname "$0")/env.sh"
NAME="${1:?Usage: ablations.sh <name> [dataset] [seed]}"
case "$NAME" in
  indist|datasize) DS="${2:?$NAME needs a dataset}"; SEED="${3:-42}" ;;
  *) SEED="${2:-42}" ;;
esac

case "$NAME" in
  control)    ARGS=() ;;
  pointwise)  ARGS=(--loss_mode pointwise_bce) ;;
  ndcg)       ARGS=(--loss_target_key action_ndcg) ;;
  typeemb)    ARGS=(--use_type_emb) ;;
  unfiltered) ARGS=(--test_threshold 0.0) ;;
  indist)     ARGS=(--eval_dataset "$DS"); NAME="indist_${DS}" ;;
  datasize)   ARGS=(--holdout_dataset "$DS" --holdout_dataset_test_on rest); NAME="datasize_${DS}" ;;
  *) echo "unknown ablation: $NAME"; exit 1 ;;
esac

bash scripts/03_train_head.sh "${NAME}_s${SEED}" ${ARGS[@]+"${ARGS[@]}"} --seed "$SEED"
