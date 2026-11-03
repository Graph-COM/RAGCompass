#!/bin/bash
# Stage 0 (optional): build the per-retriever results from the raw QA datasets.
# For every query of every split, each retriever returns its top-5 passages and
# the reader (Llama-3.1-8B-Instruct) answers from them; the answer F1 is the
# utility the router is trained on.
#
# Expects dataset/<name>.json and dataset/<name>_corpus.json for
# <name> in {<dataset>, <dataset>_train, <dataset>_val}.
#
# Usage: bash scripts/00_build_data.sh <dataset>     (nq | musique | hotpotqa | 2wikimultihopqa)
#
# HippoRAG2 builds a knowledge-graph index with vLLM; if vLLM lives in a
# separate environment, point KG_PYTHON at that interpreter.
source "$(dirname "$0")/env.sh"
DATASET="${1:?Usage: 00_build_data.sh <dataset>}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$OUT/artifacts}"
export RAGCOMPASS_ARTIFACT_ROOT="$ARTIFACT_ROOT"

"$PYTHON" -u preprocess_expanded_datasets.py \
  --datasets "$DATASET" \
  --artifact_root "$ARTIFACT_ROOT" \
  --results_base_dir "$RESULTS_DIR" \
  --stages artifacts,direct \
  --retrievers bm25_1_step,rag_1_step,splade_v3_1_step,colbertv2_1_step

"$PYTHON" -u preprocess_expanded_datasets.py \
  --datasets "$DATASET" \
  --artifact_root "$ARTIFACT_ROOT" \
  --results_base_dir "$RESULTS_DIR" \
  --stages artifacts,direct \
  --retrievers hipporag2-0.5_1_step \
  --kg_backend vllm \
  --kg_python "${KG_PYTHON:-$PYTHON}"
