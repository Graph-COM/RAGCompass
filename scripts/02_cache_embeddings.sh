#!/bin/bash
# Stage 2: freeze the fine-tuned backbone and cache, for every query and every
# retriever, one hidden state per retrieved passage (shape [N, R, 5, H]). Also
# adds nDCG-against-gold-passages targets to the cache, used only by the
# utility-vs-relevance ablation.
#
# Usage: bash scripts/02_cache_embeddings.sh [extra args...]
#   SCORER_DIR (backbone checkpoint) and CACHE_DIR (output) come from env.sh.
source "$(dirname "$0")/env.sh"
TOP_K="${TOP_K:-5}"

"$PYTHON" -u -m router.cache_embeddings \
  --model_name "$BACKBONE_NAME" \
  --pretrained_model_path "$SCORER_DIR" \
  --datasets "$DATASETS" \
  --retrievers "$RETRIEVERS" \
  --results_base_dir "$RESULTS_DIR" \
  --label_metric f1 \
  --threshold 0.0 \
  --max_length 2048 \
  --top_k_docs "$TOP_K" \
  --batch_size 1 \
  --encode_batch_size 16 \
  --routing_prompt \
  --extraction_source document_boundaries \
  --cache_dir "$CACHE_DIR" \
  "$@"

if [[ "${SKIP_NDCG:-0}" != "1" ]]; then
  "$PYTHON" -u -m router.inject_ndcg_targets \
    --cache_dir "$CACHE_DIR" \
    --datasets "$DATASETS" \
    --retrievers "$RETRIEVERS" \
    --results_base_dir "$RESULTS_DIR"
fi
