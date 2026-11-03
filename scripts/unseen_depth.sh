#!/bin/bash
# Unseen retrieval depth: the router is trained at k=5 and evaluated at a
# shallower k with no retraining. Same reader, same prompt, same retrieved
# ranking; the answers are regenerated from the top-k passages only, because
# utility at depth k is not recoverable by slicing a k=5 answer.
#
# Usage: bash scripts/unseen_depth.sh relabel <dataset> [k]   one run per dataset (GPU), default k=3
#        bash scripts/unseen_depth.sh eval [k] [seed]         test-split cache at depth k + routed F1
source "$(dirname "$0")/env.sh"
STEP="${1:?Usage: unseen_depth.sh relabel <dataset> [k] | eval [k] [seed]}"

case "$STEP" in
  relabel)
    DATASET="${2:?relabel needs a dataset}"; K="${3:-3}"
    "$PYTHON" -u -m analysis.relabel_with_reader \
      --decisions "$DIAG_DIR/routing_decisions_s42.jsonl" \
      --dataset "$DATASET" --retrievers "$RETRIEVERS" \
      --source_base_dir "$RESULTS_DIR" --output_dir "${RESULTS_DIR}__k${K}" \
      --reader_tag Llama-3.1-8B-Instruct --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
      --top_k "$K" ;;
  eval)
    K="${2:-3}"; SEED="${3:-42}"
    DEPTH_CACHE="$OUT/cache/depth_k${K}"
    # Test split only, encoded with the same backbone as the main cache.
    RESULTS_DIR="${RESULTS_DIR}__k${K}" CACHE_DIR="$DEPTH_CACHE" TOP_K="$K" SKIP_NDCG=1 \
      bash scripts/02_cache_embeddings.sh --splits test
    # The relabeled results hold exactly the reported test queries, so no utility-gap
    # filter is re-applied (--threshold 0.0). The head's only depth-dependent parameter is a position embedding indexed
    # 0..k-1, so at k<5 it uses the first k learned positions.
    "$PYTHON" -u -m analysis.dump_decisions --cache_dir "$DEPTH_CACHE" \
      --head_dir "${HEAD_DIR:-$OUT/checkpoints/head_main_s${SEED}}" --threshold 0.0 --skip_gate \
      --output "$DIAG_DIR/routing_decisions_k${K}_s${SEED}.jsonl" ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
