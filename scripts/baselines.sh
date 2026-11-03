#!/bin/bash
# Baselines of the main results table and of the generalization comparison.
# All run on the same filtered population, splits and seeds as the router.
#
# Usage: bash scripts/baselines.sh <name> [args...]
#
#   rrf <dataset>                  reciprocal rank fusion (k=60) + reader answer, test split
#   mor <dataset>                  static-weight mixture (weights fit on validation utility)
#   fusion_scores                  test F1 of best fixed / oracle / RRF / MoR on the filtered set
#
#   ltrr [seed]                    LTRR-style ranker, in distribution
#   ltrr_lodo <dataset> [seed]     ... leave-one-dataset-out
#   ltrr_loro <retriever> [seed]   ... leave-one-retriever-out (shared per-retriever scorer);
#                                  add IDENTITY=1 for the retriever-identity variant
#   ltrr_shared [seed]             shared per-retriever scorer, no holdout (control for ltrr_loro)
#
#   r3ag_judge                     one-off LLM-judge pass for R3AG's retrieval-quality signal (needs vLLM)
#   r3ag [tag] [holdout_dataset] [holdout_retriever]     R3AG, seeds 42 43 44, then aggregate
#
#   router_retriever [seed] [holdout_dataset]            R.RETRIEVER nearest-centroid routing
#
#   majority_vote [seed]           modal answer across the five retrievers' answers
#   llm_router <judge_evidence|select_answers>           zero-shot Llama-3.1-8B-Instruct as router
#
# majority_vote and llm_router read the per-query decisions file written by
# scripts/analysis.sh (it defines the evaluated test population).
source "$(dirname "$0")/env.sh"
NAME="${1:?Usage: baselines.sh <name> [args...]}"
shift
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$OUT/artifacts}"
COMMON=(--datasets "$DATASETS" --retrievers "$RETRIEVERS" --results_base_dir "$RESULTS_DIR" --threshold 0.05 --top_k_docs 5)

case "$NAME" in
  rrf|mor)
    DATASET="${1:?$NAME needs a dataset}"
    export RAGCOMPASS_ARTIFACT_ROOT="$ARTIFACT_ROOT"
    "$PYTHON" -u -m "baselines.$NAME" --backend hf --datasets "$DATASET" --no_expand_groups \
      --artifact_root "$ARTIFACT_ROOT" --results_base_dir "$RESULTS_DIR" ;;
  fusion_scores)
    "$PYTHON" -u -m analysis.fixed_and_fusion_scores "$RESULTS_DIR" ;;

  ltrr)
    SEED="${1:-42}"
    "$PYTHON" -u -m baselines.ltrr "${COMMON[@]}" --label_metric f1 --seed "$SEED" \
      --output_json "$DIAG_DIR/ltrr_s${SEED}.json" ;;
  ltrr_lodo)
    HOLDOUT="${1:?ltrr_lodo needs a dataset}"; SEED="${2:-42}"
    "$PYTHON" -u -m baselines.ltrr "${COMMON[@]}" --label_metric f1 --seed "$SEED" \
      --holdout_dataset "$HOLDOUT" --output_json "$DIAG_DIR/ltrr_lodo_${HOLDOUT}_s${SEED}.json" ;;
  ltrr_shared|ltrr_loro)
    EXTRA=(); TAG="control"
    if [[ "$NAME" == "ltrr_loro" ]]; then
      HOLDOUT="${1:?ltrr_loro needs a retriever}"; shift
      EXTRA+=(--holdout_action "$HOLDOUT"); TAG="loro_$(echo "$HOLDOUT" | tr '.-' '__')"
    fi
    SEED="${1:-42}"; ARM="idfree"
    if [[ "${IDENTITY:-0}" == "1" ]]; then EXTRA+=(--use_retriever_identity); ARM="ident"; fi
    "$PYTHON" -u -m baselines.ltrr_shared_scorer "${COMMON[@]}" --label_metric f1 --seed "$SEED" \
      ${EXTRA[@]+"${EXTRA[@]}"} --output_json "$DIAG_DIR/ltrr_shared_${ARM}_${TAG}_s${SEED}.json" ;;

  r3ag_judge)
    "$PYTHON" -u -m baselines.r3ag_judge --datasets "$DATASETS" --retrievers "$RETRIEVERS" \
      --judge_model Qwen2.5-7B-Instruct --results_base_dir "$RESULTS_DIR" \
      --output_base_dir "$OUT/r3ag_quality_judge_cache" ;;
  r3ag)
    TAG="${1:-main}"; HOLDOUT_DATASET="${2:-}"; HOLDOUT_RETRIEVER="${3:-}"
    EXTRA=()
    [[ -n "$HOLDOUT_DATASET" ]] && EXTRA+=(--holdout_dataset "$HOLDOUT_DATASET")
    [[ -n "$HOLDOUT_RETRIEVER" ]] && EXTRA+=(--holdout_retriever "$HOLDOUT_RETRIEVER")
    JSONS=()
    for SEED in ${SEEDS:-42 43 44}; do
      J="$DIAG_DIR/r3ag_${TAG}_s${SEED}.json"; JSONS+=("$J")
      # No null (no-retrieval) action, so R3AG is scored on the same population as every other method.
      "$PYTHON" -u -m baselines.r3ag_train "${COMMON[@]}" \
        --no_retrieval_action none \
        --judge_cache_base_dir "$OUT/r3ag_quality_judge_cache" \
        --top_k_contrastive 2 --temperature 0.1 --beta 0.2 --gamma 0.2 --lambda_reg 0.1 \
        --num_trainable_layers 2 \
        --stage1_epochs 10 --stage1_batch_size 16 --stage1_lr 2e-6 \
        --stage2_epochs 10 --stage2_batch_size 64 --stage2_lr 1e-4 \
        --resplit_train_ratio 0.7 --seed "$SEED" \
        ${EXTRA[@]+"${EXTRA[@]}"} --output_json "$J"
    done
    "$PYTHON" -u -m baselines.r3ag_aggregate_seeds --input_jsons "${JSONS[@]}" \
      --output_json "$DIAG_DIR/r3ag_${TAG}_aggregate.json" ;;

  router_retriever)
    SEED="${1:-42}"; HOLDOUT_DATASET="${2:-}"; TAG="main"; EXTRA=()
    if [[ -n "$HOLDOUT_DATASET" ]]; then EXTRA+=(--holdout_dataset "$HOLDOUT_DATASET"); TAG="lodo_${HOLDOUT_DATASET}"; fi
    "$PYTHON" -u -m baselines.router_retriever "${COMMON[@]}" --resplit_train_ratio 0.7 --seed "$SEED" \
      --embedding_cache_path "$OUT/cache/router_retriever_query_embeddings.pt" \
      ${EXTRA[@]+"${EXTRA[@]}"} --output_json "$DIAG_DIR/router_retriever_${TAG}_s${SEED}.json" ;;

  majority_vote)
    SEED="${1:-42}"
    "$PYTHON" -u -m baselines.majority_vote --decisions "$DIAG_DIR/routing_decisions_s${SEED}.jsonl" \
      --results_base_dir "$RESULTS_DIR" --retrievers "$RETRIEVERS" \
      --output_json "$DIAG_DIR/majority_vote_s${SEED}.json" ;;
  llm_router)
    MODE="${1:?llm_router needs judge_evidence or select_answers}"; shift
    export RAGCOMPASS_ARTIFACT_ROOT="$ARTIFACT_ROOT"
    "$PYTHON" -u -m baselines.llm_router --mode "$MODE" \
      --decisions "$DIAG_DIR/routing_decisions_s42.jsonl" \
      --results_base_dir "$RESULTS_DIR" --retrievers "$RETRIEVERS" \
      --llm Llama-3.1-8B-Instruct --top_k 5 --seed 42 \
      --output_json "$DIAG_DIR/llm_router_${MODE}.json" \
      --output_jsonl "$DIAG_DIR/llm_router_${MODE}_per_query.jsonl" "$@" ;;
  *) echo "unknown baseline: $NAME"; exit 1 ;;
esac
