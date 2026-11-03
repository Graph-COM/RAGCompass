"""
LLM-as-router baselines: "why not just ask an LLM which retriever to trust?"

There are two versions of this question:

  --mode judge_evidence   Show the reader LLM the query and all R evidence sets,
                          ask which set it would rather answer from. This is the
                          zero-shot version of exactly our task, using a model
                          ~13x larger than our 0.6B router.
  --mode select_answers   Let the reader answer from each evidence set (already
                          done -- the answers are cached), then ask the LLM to
                          pick the best answer. This is the "generate first,
                          choose later" strategy, whose extra cost is R full
                          generations per query.

Both are evaluated on the SAME filtered test set as the router, and scored the
same way: the choice selects one retriever, and that retriever's already-scored
answer F1 is the outcome. So the numbers are directly comparable to the main results table.

Presentation order is shuffled per query with a fixed seed and mapped back, so
a positional preference in the LLM cannot masquerade as a routing preference --
the raw position histogram is reported so that bias is visible if present.

Usage:
    python -m baselines.llm_router --mode judge_evidence \
        --decisions outputs/diagnostics/routing_decisions_s42.jsonl \
        --output_json outputs/diagnostics/llm_router_judge_evidence.json
"""

import argparse
import json
import os
import random
import string
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_hf import load_model  # noqa: E402
from src.artifacts import apply_cache_environment  # noqa: E402
from src.data import load_corpus  # noqa: E402
from src.llm_hf import query_llm_hf  # noqa: E402

DEFAULT_RESULTS_BASE_DIR = "Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B"
DEFAULT_RETRIEVERS = [
    "rag_1_step",
    "bm25_1_step",
    "splade_v3_1_step",
    "colbertv2_1_step",
    "hipporag2-0.5_1_step",
]
CHOICE_LETTERS = string.ascii_uppercase


def item_key(item, idx):
    for key in ("id", "_id", "qid", "question_id"):
        if key in item:
            return str(item[key])
    return f"row_{idx}"


def as_answer_text(pred_ans):
    if pred_ans is None:
        return ""
    if isinstance(pred_ans, str):
        return pred_ans
    if isinstance(pred_ans, (list, tuple)):
        return " ".join(as_answer_text(part) for part in pred_ans)
    return str(pred_ans)


def load_rows(base_dir, dataset, retriever):
    path = os.path.join(base_dir, dataset, f"{retriever}.jsonl")
    rows = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rows[item_key(item, idx)] = item
    return rows


def build_evidence_prompt(question, evidence_blocks):
    options = "\n\n".join(
        f"Option {CHOICE_LETTERS[i]}:\n{block}" for i, block in enumerate(evidence_blocks)
    )
    letters = ", ".join(CHOICE_LETTERS[: len(evidence_blocks)])
    return [
        {
            "role": "system",
            "content": (
                "You compare sets of retrieved passages. Given a question and several "
                "candidate passage sets, identify which single set is most likely to "
                "let you answer the question correctly. Reply with one letter only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n{options}\n\n"
                f"Which option's passages best support answering the question? "
                f"Reply with exactly one letter ({letters}) and nothing else."
            ),
        },
    ]


def build_answer_prompt(question, answers):
    options = "\n".join(
        f"{CHOICE_LETTERS[i]}. {answer or '(no answer)'}" for i, answer in enumerate(answers)
    )
    letters = ", ".join(CHOICE_LETTERS[: len(answers)])
    return [
        {
            "role": "system",
            "content": (
                "You judge candidate answers. Given a question and several candidate "
                "answers produced from different evidence, identify the one most likely "
                "to be correct. Reply with one letter only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nCandidate answers:\n{options}\n\n"
                f"Which candidate answer is most likely correct? "
                f"Reply with exactly one letter ({letters}) and nothing else."
            ),
        },
    ]


def parse_choice(response, n_options):
    """First valid option letter in the response; None if the model refused to pick."""
    for ch in (response or "").strip().upper():
        if ch in CHOICE_LETTERS[:n_options]:
            return CHOICE_LETTERS.index(ch)
    return None


def main():
    parser = argparse.ArgumentParser(description="LLM-as-router baselines")
    parser.add_argument("--mode", choices=["judge_evidence", "select_answers"], required=True)
    parser.add_argument("--decisions", type=str, required=True,
                        help="per_query_routing_decisions.jsonl -- defines the example set and "
                             "carries each retriever's answer F1 for scoring.")
    parser.add_argument("--results_base_dir", type=str, default=DEFAULT_RESULTS_BASE_DIR)
    parser.add_argument("--retrievers", type=str, default=",".join(DEFAULT_RETRIEVERS))
    parser.add_argument("--llm", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None, help="Smoke-test cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default=None)
    args = parser.parse_args()

    apply_cache_environment(artifact_root=args.artifact_root)
    retrievers = [r.strip() for r in args.retrievers.split(",") if r.strip()]

    decisions = []
    with open(args.decisions) as f:
        for line in f:
            line = line.strip()
            if line:
                decisions.append(json.loads(line))
    if args.max_samples:
        decisions = decisions[: args.max_samples]
    datasets = sorted({row["dataset"] for row in decisions})
    print(f"{len(decisions)} queries across {datasets}; mode={args.mode}")

    print("Loading per-retriever result rows...")
    rows_by = {
        (dataset, retriever): load_rows(args.results_base_dir, dataset, retriever)
        for dataset in datasets
        for retriever in retrievers
    }
    corpora = {}
    if args.mode == "judge_evidence":
        for dataset in datasets:
            print(f"Loading corpus for {dataset}...")
            corpora[dataset] = load_corpus(dataset)

    tokenizer, model = load_model(args.llm, args.model_id)
    rng = random.Random(args.seed)

    selected_f1, chosen_positions, chosen_retrievers = [], Counter(), Counter()
    unparsed = 0
    skipped = 0
    per_query = []
    start = time.time()

    for row in tqdm(decisions, desc=args.mode):
        dataset, qid = row["dataset"], str(row["id"])
        items = [rows_by[(dataset, r)].get(qid) for r in retrievers]
        if any(item is None for item in items):
            skipped += 1
            continue
        question = items[0]["question"]

        # Shuffle which retriever sits at which option letter, so a positional
        # preference cannot be read as a routing preference.
        order = list(range(len(retrievers)))
        rng.shuffle(order)

        if args.mode == "judge_evidence":
            corpus = corpora[dataset]
            blocks = []
            for pos in order:
                indices = items[pos].get("top_k_indices", [])[: args.top_k]
                blocks.append("\n\n".join(str(corpus[i]) for i in indices))
            prompt = build_evidence_prompt(question, blocks)
        else:
            answers = [as_answer_text(items[pos].get("pred_ans")) for pos in order]
            prompt = build_answer_prompt(question, answers)

        response = query_llm_hf(prompt, tokenizer, model, args.max_new_tokens, temperature=0)
        picked = parse_choice(response, len(order))
        if picked is None:
            unparsed += 1
            # Unparseable output must not be silently scored as a lucky guess:
            # fall back to the first presented option, which is arbitrary by
            # construction thanks to the shuffle.
            picked = 0
        chosen_positions[picked] += 1
        retriever_idx = order[picked]
        retriever_name = retrievers[retriever_idx]
        chosen_retrievers[retriever_name] += 1
        f1 = row["all_f1s"][retriever_name]
        selected_f1.append(f1)

        per_query.append({
            "id": qid,
            "dataset": dataset,
            "chosen_retriever": retriever_name,
            "chosen_position": picked,
            "f1": f1,
            "raw_response": response,
        })

    elapsed = time.time() - start
    n = len(selected_f1)
    per_retriever_f1 = {r: np.mean([d["all_f1s"][r] for d in decisions]) for r in retrievers}
    best_fixed_name = max(per_retriever_f1, key=per_retriever_f1.get)

    results = {
        "mode": args.mode,
        "llm": args.model_id or args.llm,
        "n": n,
        "n_skipped": skipped,
        "n_unparsed": unparsed,
        "selected_f1": float(np.mean(selected_f1)) if n else None,
        "best_fixed_f1": float(per_retriever_f1[best_fixed_name]),
        "best_fixed_retriever": best_fixed_name,
        "oracle_f1": float(np.mean([d["oracle_f1"] for d in decisions])),
        "our_router_f1": float(np.mean([d["chosen_f1"] for d in decisions])),
        "choice_distribution_by_retriever": dict(chosen_retrievers),
        "choice_distribution_by_position": {str(k): v for k, v in sorted(chosen_positions.items())},
        "seconds_total": elapsed,
        "seconds_per_query": elapsed / n if n else None,
    }
    results["vs_best_fixed"] = (results["selected_f1"] - results["best_fixed_f1"]) if n else None
    results["vs_our_router"] = (results["selected_f1"] - results["our_router_f1"]) if n else None

    print(f"\n=== {args.mode} ({results['llm']}) ===")
    print(f"  LLM router selected F1  {results['selected_f1']:.4f}")
    print(f"  best fixed retriever    {results['best_fixed_f1']:.4f}  ({best_fixed_name})")
    print(f"  our 0.6B router         {results['our_router_f1']:.4f}")
    print(f"  oracle                  {results['oracle_f1']:.4f}")
    print(f"  vs best fixed {results['vs_best_fixed']:+.4f} | vs our router {results['vs_our_router']:+.4f}")
    print(f"  unparsed responses: {unparsed}/{n}")
    print(f"  choice by position (bias check): {results['choice_distribution_by_position']}")
    print(f"  choice by retriever: {results['choice_distribution_by_retriever']}")
    print(f"  {results['seconds_per_query']:.2f} s/query")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.output_json}")
    if args.output_jsonl:
        with open(args.output_jsonl, "w") as f:
            for record in per_query:
                f.write(json.dumps(record) + "\n")
        print(f"Saved per-query -> {args.output_jsonl}")


if __name__ == "__main__":
    main()
