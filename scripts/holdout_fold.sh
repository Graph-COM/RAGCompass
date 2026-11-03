#!/bin/bash
# One fold of a generalization protocol, end to end: the backbone AND the head
# are retrained without the held-out dataset / retriever, and the router is then
# evaluated on it.
#
#   unseen corpus     leave-one-dataset-out: the held-out dataset's queries and
#                     passages are never seen in training.
#   unseen mechanism  leave-one-retriever-out: the retriever is removed from the
#                     training pool and labels but kept in the test pool.
#
# Usage: bash scripts/holdout_fold.sh dataset   <name>
#        bash scripts/holdout_fold.sh retriever <name>
#   bash scripts/holdout_fold.sh dataset nq
#   bash scripts/holdout_fold.sh retriever hipporag2-0.5_1_step
#
# HEAD_SEEDS (default "42 43 44") sets the head seeds. TYPE_EMB=1 trains the
# identity-embedding head variant instead.
source "$(dirname "$0")/env.sh"
KIND="${1:?Usage: holdout_fold.sh dataset|retriever <name>}"
NAME="${2:?Usage: holdout_fold.sh dataset|retriever <name>}"
HEAD_SEEDS="${HEAD_SEEDS:-42 43 44}"
TAG="$(echo "$NAME" | tr '.-' '__')"

without() { echo "$1" | tr ',' '\n' | grep -vx "$2" | paste -sd, -; }

case "$KIND" in
  dataset)   TRAIN_DATASETS="$(without "$DATASETS" "$NAME")"; TRAIN_RETRIEVERS="$RETRIEVERS"
             PREFIX="lodo"; HOLDOUT_ARGS=(--holdout_dataset "$NAME" --holdout_dataset_test_on holdout) ;;
  retriever) TRAIN_DATASETS="$DATASETS"; TRAIN_RETRIEVERS="$(without "$RETRIEVERS" "$NAME")"
             PREFIX="loro"; HOLDOUT_ARGS=(--holdout_action "$NAME") ;;
  *) echo "first argument must be 'dataset' or 'retriever'"; exit 1 ;;
esac

FOLD_SCORER="$OUT/checkpoints/scorer_no_${TAG}"
FOLD_CACHE="$OUT/cache/no_${TAG}"
# 1. Backbone fine-tuned without the held-out dataset / retriever.
#    Skipped if the fold's cache already exists (e.g. for the TYPE_EMB=1 rerun).
if [[ ! -d "$FOLD_CACHE" ]]; then
  DATASETS="$TRAIN_DATASETS" RETRIEVERS="$TRAIN_RETRIEVERS" MODEL_DIR="$FOLD_SCORER" \
    bash scripts/01_train_scorer.sh 42
  # 2. Cache ALL datasets and retrievers with that backbone (the held-out one is needed at test time).
  SCORER_DIR="$FOLD_SCORER" CACHE_DIR="$FOLD_CACHE" SKIP_NDCG=1 bash scripts/02_cache_embeddings.sh
fi
export CACHE_DIR="$FOLD_CACHE"

# 3. Head trained without the held-out dataset / retriever, evaluated on it.
for SEED in $HEAD_SEEDS; do
  if [[ "${TYPE_EMB:-0}" == "1" ]]; then
    bash scripts/03_train_head.sh "${PREFIX}_${TAG}_typeemb_s${SEED}" "${HOLDOUT_ARGS[@]}" --use_type_emb --seed "$SEED"
  else
    bash scripts/03_train_head.sh "${PREFIX}_${TAG}_s${SEED}" "${HOLDOUT_ARGS[@]}" --seed "$SEED"
  fi
done
