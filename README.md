# RAGCompass: Generalizable Retriever Routing via Reranking

Code for the paper [*RAGCompass: Generalizable Retriever Routing via Reranking*](https://arxiv.org/abs/2611.05596) by Valentin Six, Mufei Li, Dongqi Fu, Yinglong Xia, Hong Li and Pan Li. The data is on the Hugging Face Hub at [Graph-COM/RAGCompass](https://huggingface.co/datasets/Graph-COM/RAGCompass).

Given a query and the passage sets returned by several retrievers, RAGCompass picks the retriever whose passages will produce the best answer. It reads the retrieved passages themselves: a reranker backbone scores each `(query, passage set)` pair, and a small comparison head ranks the candidates. Neither component has retriever-specific parameters, so the router stays defined for a retriever or corpus it was not trained on.

## Repository layout

```
router/      the method
  train_scorer.py        Stage 1: fine-tune the reranker as a scalar scorer (KL to the utility distribution)
  cache_embeddings.py    Stage 2: cache per-passage hidden states of the frozen backbone
  train_head.py          Stage 3: train the comparison head on the cache; hold-out protocols
  train_e2e.py           joint training of backbone and head
  joint_prompt_*.py      joint-prompt encoding ablation
baselines/   RRF, MoR, LTRR-style ranker, R3AG, R.RETRIEVER, majority vote, LLM-as-router
analysis/    per-query decisions, retrieval metrics, oracle gap, second reader, latency, diagnostics
scripts/     one shell script per experiment, with the hyperparameters used in the paper
src/, main_hf.py, kg_index.py, kg_emb.py, preprocess_expanded_datasets.py
             retrieval + reader pipeline that produces the per-retriever results (Stage 0)

Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B/    per-retriever results used in the paper (downloaded, see below)
dataset/                                       passage corpora of every split (downloaded, see below)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The experiments ran with Python 3.10; `requirements.txt` pins the library versions of that environment (PyTorch 2.11.0, Transformers 4.57.1, scikit-learn 1.7.2). Backbone fine-tuning, embedding caching and the reader need a GPU; head training runs on a small GPU, and the LTRR-style baseline and most analyses run on CPU. HippoRAG2 indexing and the R3AG judge use [vLLM](https://github.com/vllm-project/vllm) 0.10.2, which requires a different PyTorch build and therefore a separate environment (`pip install vllm==0.10.2`); the scripts that need it accept a `KG_PYTHON` / `PYTHON` override.

The backbone is `Qwen/Qwen3-Reranker-0.6B`, the reader is `meta-llama/Llama-3.1-8B-Instruct` (a gated model: accept its license on the Hugging Face Hub and log in before running anything that calls the reader), and the dense retriever uses `Qwen/Qwen3-Embedding-0.6B`. Models are downloaded from the Hub on first use.

### Data

The per-retriever results and the passage corpora (about 1 GB) are hosted on the Hugging Face Hub. From the root of this repository, with the environment above active (it provides the `hf` command):

```bash
hf download Graph-COM/RAGCompass --repo-type dataset --local-dir . --include "*.json*"
```

This creates `dataset/` and `Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B/` next to the code. Everything below needs them.

### Quick check (CPU, about 3 minutes)

```bash
bash scripts/baselines.sh ltrr 42
```

This reads the downloaded data, rebuilds the filtered test set, and trains the LTRR-style baseline. It should report `Combined TEST: 3735 examples`, `best_fixed=0.2782` and `oracle=0.7508`, the reference numbers every method in the paper is compared against.

## Data layout

All experiments read *per-retriever results*: for every query, the passages one retriever returned and the reader's answer from them. The results used in the paper are on the Hub (4 datasets x 3 splits x 5 retrievers, 5,000 queries per split), together with the passage corpora, so Stages 1-3, the trained baselines and the analyses run without rebuilding anything.

```
Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B/        # $RESULTS_DIR
  <dataset>_train/<retriever>.jsonl
  <dataset>_val/<retriever>.jsonl
  <dataset>/<retriever>.jsonl                       # test split
dataset/
  <dataset>[_train|_val]_corpus.json                # list of {"title", "text"}  (on the Hub)
  <dataset>[_train|_val].json                       # questions and gold answers (only for Stage 0, RRF and MoR)
```

with `<dataset>` in `nq`, `musique`, `hotpotqa`, `2wikimultihopqa` and `<retriever>` in `rag_1_step` (dense), `bm25_1_step`, `splade_v3_1_step`, `colbertv2_1_step`, `hipporag2-0.5_1_step`. Each JSONL row holds one query:

| field | meaning |
|---|---|
| `id`, `question`, `answers` | query id, text, gold answers |
| `top_k_indices`, `top_k_scores` | the retriever's top-5 passages (indices into the split's corpus) and scores |
| `pred_ans`, `f1` | the reader's answer from those passages and its F1: the utility the router is trained on |
| `recall`, `ndcg`, `gold_doc_list` | retrieval quality against gold passages |
| `latency` | seconds for retrieval plus generation |

Each row also keeps the reader's full response (`raw_response`).

Rebuilding the results from scratch is optional. It runs every retriever and the reader over each split (Stage 0), and needs the question files `dataset/<dataset>[_train|_val].json` next to the corpora, in the JSON format read by `src/data.py` (the one used by the [HippoRAG 2 release](https://huggingface.co/datasets/osunlp/HippoRAG_2): a list of samples with `question`, `answer` and their supporting passages); they are not included because the results already carry every query, its gold answers and its gold passages. RRF and MoR re-run the reader on fused passages and read the same question files.

```bash
bash scripts/00_build_data.sh nq        # also: musique, hotpotqa, 2wikimultihopqa
```

## Reproducing the results

Every script reads `scripts/env.sh`; set `RESULTS_DIR` (input) and `OUT` (everything written, default `outputs/`) to relocate data. Each run writes its logs and metrics under `$OUT/runs/<method>/.../<run>/` (`metrics/summary.json` for the scorer and the head, `metrics/results.json` for joint training and the joint-prompt ablation) and its checkpoint under `$OUT/checkpoints/`; baselines and analyses write JSON to `$OUT/diagnostics/`. Trained methods are reported as mean and standard deviation over seeds 42, 43, 44.

### Main result

```bash
# Scalar scorer, no head  (0.3412 +- 0.0059)
for s in 42 43 44; do bash scripts/01_train_scorer.sh $s; done

# + comparison head       (0.3552 +- 0.0026), on the frozen seed-42 backbone
bash scripts/02_cache_embeddings.sh
for s in 42 43 44; do bash scripts/03_train_head.sh main_s$s --seed $s; done
```

The filtered test set has 3,735 queries (top-two utility gap of at least 0.05); on it the best fixed retriever (SPLADE-v3) scores 0.2782 and the oracle 0.7508. Several scripts check these two numbers before writing any output, as a guard that the evaluated population is the reported one.

### Baselines

```bash
for d in nq musique hotpotqa 2wikimultihopqa; do
  bash scripts/baselines.sh rrf $d; bash scripts/baselines.sh mor $d
done
bash scripts/baselines.sh fusion_scores            # best fixed 0.2782, RRF 0.3002, MoR 0.3203, oracle 0.7508

for s in 42 43 44; do bash scripts/baselines.sh ltrr $s; done               # LTRR-style ranker
bash scripts/baselines.sh r3ag_judge && bash scripts/baselines.sh r3ag      # R3AG
for s in 42 43 44; do bash scripts/baselines.sh router_retriever $s; done   # R.RETRIEVER

bash scripts/analysis.sh 42                         # writes the per-query decisions the next two need
bash scripts/baselines.sh majority_vote
bash scripts/baselines.sh llm_router judge_evidence
bash scripts/baselines.sh llm_router select_answers
```

### Generalization

```bash
# Unseen corpus (leave-one-dataset-out) and unseen mechanism (leave-one-retriever-out).
# Each fold retrains the backbone and the head without the held-out dataset / retriever.
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
bash scripts/baselines.sh ltrr_loro bm25_1_step 42            # IDENTITY=1 for the retriever-identity arm
bash scripts/baselines.sh r3ag lodo_nq nq
bash scripts/baselines.sh r3ag loro_bm25 "" bm25_1_step
bash scripts/baselines.sh router_retriever 42 nq

# Unseen reader (Qwen2.5-7B-Instruct) and unseen retrieval depth (k=3); no retraining
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
| Training-set-size ablation | `bash scripts/ablations.sh datasize <dataset> <seed>` |
| Joint-prompt encoding | `bash scripts/joint_prompt.sh <variant>` with `yes_logit`, `final_frozen`, `final_e2e` |
| Per-dataset breakdown, recall@5 / nDCG@5, routing distribution, oracle-gap decomposition | `bash scripts/analysis.sh <seed>` |
| What the scorer's representations retain | `bash scripts/analysis.sh representation` |
| Cost of a routing decision | `bash scripts/analysis.sh latency` |

The head architecture sweep varies `--late_interaction_pool`, `--cross_layers`, `--cross_dropout`, `--use_type_emb`, `--loss_mode`, `--soft_target_temperature` and `--use_yes_logit` of `router/train_head.py`; pass them after the run name to `scripts/03_train_head.sh`.

## Notes

- Experiment tracking with Weights & Biases is off by default (`WANDB_MODE=disabled` in `scripts/env.sh`).
- `RAGCOMPASS_ARTIFACT_ROOT` (set by the scripts to `$OUT/artifacts`) is where retriever indexes, corpus embeddings and the HippoRAG2 knowledge graph are stored.
- Gradient-boosting and random-forest fits in the LTRR-style baseline differ across scikit-learn versions; use the pinned version to reproduce them.

## License and acknowledgements

The code is released under the MIT License (see `LICENSE`). The retrieval pipeline and the baselines (HippoRAG 2, RRF, MoR, the LTRR-style ranker, R3AG, RouterRetriever) are reimplementations of published methods and may reuse literal portions of their authors' released code and prompts; those portions remain under their original licenses, and we thank the authors for releasing them. The data on the Hub is derived from Natural Questions, HotpotQA, MuSiQue and 2WikiMultihopQA and is redistributed under the licenses of those datasets.

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
