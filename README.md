# RAGCompass: Generalizable Retriever Routing via Reranking

Code for [*RAGCompass: Generalizable Retriever Routing via Reranking*](https://arxiv.org/abs/2611.05596) (Valentin Six, Mufei Li, Dongqi Fu, Yinglong Xia, Hong Li, Pan Li). Data: [Graph-COM/RAGCompass](https://huggingface.co/datasets/Graph-COM/RAGCompass) on the Hugging Face Hub.

Given a query and the passage sets returned by several retrievers, RAGCompass picks the retriever whose passages will produce the best answer. A reranker backbone scores each `(query, passage set)` pair and a small comparison head ranks the candidates. Neither has retriever-specific parameters, so the router applies to retrievers and corpora unseen in training.

## Repository layout

```
router/      the method
  train_scorer.py        Stage 1: fine-tune the reranker as a scalar scorer
  cache_embeddings.py    Stage 2: cache per-passage hidden states of the frozen backbone
  train_head.py          Stage 3: train the comparison head; hold-out protocols
  train_e2e.py           joint training of backbone and head
  joint_prompt_*.py      joint-prompt encoding ablation
baselines/   RRF, MoR, LTRR-style ranker, R3AG, R.RETRIEVER, majority vote, LLM-as-router
analysis/    per-query decisions, retrieval metrics, oracle gap, latency, diagnostics
scripts/     one shell script per experiment, with the paper's hyperparameters
src/, main_hf.py, kg_*.py, preprocess_expanded_datasets.py
             retrieval + reader pipeline producing per-retriever results (Stage 0)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download Graph-COM/RAGCompass --repo-type dataset --local-dir . --include "*.json*"
```

Python 3.10, with versions pinned in `requirements.txt`. The download (~1 GB) creates `dataset/` and `Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B/`. Backbone training, embedding caching and the reader need a GPU. HippoRAG2 indexing and the R3AG judge need [vLLM](https://github.com/vllm-project/vllm) 0.10.2 in a separate environment (point the scripts to it with `KG_PYTHON` / `PYTHON`).

Models: backbone `Qwen/Qwen3-Reranker-0.6B`, reader `meta-llama/Llama-3.1-8B-Instruct` (gated: accept the license and log in), dense retriever `Qwen/Qwen3-Embedding-0.6B`.

**Quick check** (CPU, ~3 min): `bash scripts/baselines.sh ltrr 42` should report `Combined TEST: 3735 examples`, `best_fixed=0.2782`, `oracle=0.7508`.

## Data

Per-retriever results cover 4 datasets × 3 splits × 5 retrievers, with 5,000 queries per split:

```
Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B/        # $RESULTS_DIR
  <dataset>[_train|_val]/<retriever>.jsonl          # no suffix = test
dataset/
  <dataset>[_train|_val]_corpus.json                # list of {"title", "text"}
```

`<dataset>` ∈ {`nq`, `musique`, `hotpotqa`, `2wikimultihopqa`}. `<retriever>` ∈ {`rag_1_step` (dense), `bm25_1_step`, `splade_v3_1_step`, `colbertv2_1_step`, `hipporag2-0.5_1_step`}. Each row is one query with these fields: `id`, `question`, `answers`, `top_k_indices`/`top_k_scores` (top-5 passages), `pred_ans` and `f1` (the reader's answer and the utility used for training), `recall`, `ndcg`, `gold_doc_list`, `latency`, `raw_response`.

To rebuild from scratch (optional), run `bash scripts/00_build_data.sh <dataset>`. Stage 0, RRF and MoR also need the question files `dataset/<dataset>[_train|_val].json`, which are not included. They use the [HippoRAG 2 format](https://huggingface.co/datasets/osunlp/HippoRAG_2).

## Reproducing the results

The scripts read `scripts/env.sh`. Set `RESULTS_DIR` (input) and `OUT` (output, default `outputs/`) there. Runs write to `$OUT/runs/`, checkpoints go to `$OUT/checkpoints/`, and baselines and analyses go to `$OUT/diagnostics/`. Trained methods report mean ± std over seeds 42, 43, 44. The filtered test set has 3,735 queries, where the best fixed retriever (SPLADE-v3) scores 0.2782 and the oracle 0.7508.

### Main result

```bash
# Scalar scorer, no head  (0.3412 ± 0.0059)
for s in 42 43 44; do bash scripts/01_train_scorer.sh $s; done

# + comparison head       (0.3552 ± 0.0026), on the frozen seed-42 backbone
bash scripts/02_cache_embeddings.sh
for s in 42 43 44; do bash scripts/03_train_head.sh main_s$s --seed $s; done
```

### Baselines

```bash
for d in nq musique hotpotqa 2wikimultihopqa; do
  bash scripts/baselines.sh rrf $d; bash scripts/baselines.sh mor $d
done
bash scripts/baselines.sh fusion_scores            # RRF 0.3002, MoR 0.3203

for s in 42 43 44; do bash scripts/baselines.sh ltrr $s; done               # LTRR-style ranker
bash scripts/baselines.sh r3ag_judge && bash scripts/baselines.sh r3ag      # R3AG
for s in 42 43 44; do bash scripts/baselines.sh router_retriever $s; done   # R.RETRIEVER

bash scripts/analysis.sh 42                         # needed by the next two
bash scripts/baselines.sh majority_vote
bash scripts/baselines.sh llm_router judge_evidence
bash scripts/baselines.sh llm_router select_answers
```

### Generalization

```bash
# Leave-one-dataset-out and leave-one-retriever-out
for d in nq musique hotpotqa 2wikimultihopqa; do
  bash scripts/holdout_fold.sh dataset $d
  for s in 42 43 44; do bash scripts/ablations.sh indist $d $s; done        # in-distribution control
done
for r in rag_1_step bm25_1_step splade_v3_1_step colbertv2_1_step hipporag2-0.5_1_step; do
  bash scripts/holdout_fold.sh retriever $r
  TYPE_EMB=1 HEAD_SEEDS=42 bash scripts/holdout_fold.sh retriever $r         # identity-embedding variant
done

# Query-only routers under the same protocols
bash scripts/baselines.sh ltrr_lodo nq 42
bash scripts/baselines.sh ltrr_loro bm25_1_step 42            # IDENTITY=1 for the identity arm
bash scripts/baselines.sh r3ag lodo_nq nq
bash scripts/baselines.sh r3ag loro_bm25 "" bm25_1_step
bash scripts/baselines.sh router_retriever 42 nq

# Unseen reader (Qwen2.5-7B-Instruct) and retrieval depth (k=3), no retraining
for d in nq musique hotpotqa 2wikimultihopqa; do
  bash scripts/unseen_reader.sh relabel $d
  bash scripts/unseen_depth.sh relabel $d 3
done
bash scripts/unseen_reader.sh eval
bash scripts/unseen_depth.sh eval 3
```

### Ablations and analyses

| Result | Command |
|---|---|
| Utility vs. relevance supervision | `bash scripts/ablations.sh ndcg <seed>` |
| Shared normalization, frozen head | `bash scripts/ablations.sh pointwise <seed>` |
| Shared normalization, end to end | `bash scripts/train_e2e.sh <seed> pointwise_bce` vs. `bash scripts/train_e2e.sh <seed>` |
| Joint training of backbone and head | `bash scripts/train_e2e.sh <seed>` |
| Identity embedding in the head | `bash scripts/ablations.sh typeemb <seed>` |
| Gain over unfiltered queries | `bash scripts/ablations.sh unfiltered <seed>` |
| Training-set size | `bash scripts/ablations.sh datasize <dataset> <seed>` |
| Joint-prompt encoding | `bash scripts/joint_prompt.sh <yes_logit\|final_frozen\|final_e2e>` |
| Per-dataset breakdown, retrieval metrics, routing distribution, oracle gap | `bash scripts/analysis.sh <seed>` |
| Scorer representations | `bash scripts/analysis.sh representation` |
| Routing cost | `bash scripts/analysis.sh latency` |

For the head architecture sweep, pass `router/train_head.py` flags (e.g. `--cross_layers`, `--loss_mode`) after the run name to `scripts/03_train_head.sh`.

## License

MIT (see `LICENSE`). Baselines and the retrieval pipeline reimplement published methods (HippoRAG 2, RRF, MoR, LTRR, R3AG, RouterRetriever) and may reuse portions of their authors' code and prompts. Those portions keep their original licenses, and we thank the authors. The data on the Hub derives from Natural Questions, HotpotQA, MuSiQue and 2WikiMultihopQA and keeps their licenses.

## Citation

```bibtex
@misc{six2026ragcompass,
  title         = {RAGCompass: Generalizable Retriever Routing via Reranking},
  author        = {Six, Valentin and Li, Mufei and Fu, Dongqi and Xia, Yinglong and Li, Hong and Li, Pan},
  year          = {2026},
  eprint        = {2611.05596},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2611.05596}
}
```
