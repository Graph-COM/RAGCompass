#!/bin/bash
# Unseen reader: the router and its evidence are unchanged; a different LLM
# (Qwen2.5-7B-Instruct) answers from each retriever's cached top-5 passages, and
# the router's existing choices are re-scored under the new reader's utilities.
#
# Usage: bash scripts/unseen_reader.sh relabel <dataset>     one run per dataset (GPU)
#        bash scripts/unseen_reader.sh eval [seed]           after all four datasets are relabeled
source "$(dirname "$0")/env.sh"
STEP="${1:?Usage: unseen_reader.sh relabel <dataset> | eval [seed]}"
READER_TAG="${READER_TAG:-Qwen2.5-7B-Instruct}"
READER_ID="${READER_ID:-Qwen/Qwen2.5-7B-Instruct}"
READER_DIR="${READER_DIR:-${READER_TAG}_Qwen3-Embedding-0.6B}"

case "$STEP" in
  relabel)
    DATASET="${2:?relabel needs a dataset}"
    "$PYTHON" -u -m analysis.relabel_with_reader \
      --decisions "$DIAG_DIR/routing_decisions_s42.jsonl" \
      --dataset "$DATASET" --retrievers "$RETRIEVERS" \
      --source_base_dir "$RESULTS_DIR" --output_dir "$READER_DIR" \
      --reader_tag "$READER_TAG" --model_id "$READER_ID" --top_k 5 ;;
  eval)
    SEED="${2:-42}"
    "$PYTHON" -u -m analysis.second_reader_eval \
      --decisions "$DIAG_DIR/routing_decisions_s${SEED}.jsonl" \
      --original_dir "$RESULTS_DIR" --reader_dir "$READER_DIR" --reader_label "$READER_TAG" \
      --retrievers "$RETRIEVERS" --output_json "$DIAG_DIR/second_reader_eval_s${SEED}.json" ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
