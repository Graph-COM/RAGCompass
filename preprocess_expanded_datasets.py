#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.artifacts import (
    ARTIFACT_ROOT_ENV,
    apply_cache_environment,
    build_kg_metadata,
    embedding_file,
    kg_embedding_dir,
    kg_index_dir,
    kg_metadata_file,
    load_json_if_exists,
    model_cache_dir,
)
from src.data import load_corpus, load_qa


DATASET_GROUPS = {
    "nq": ["nq_train", "nq_val", "nq"],
    "musique": ["musique_train", "musique_val", "musique"],
    "hotpotqa": ["hotpotqa_train", "hotpotqa_val", "hotpotqa"],
    "2wikimultihopqa": ["2wikimultihopqa_train", "2wikimultihopqa_val", "2wikimultihopqa"],
}

DIRECT_RETRIEVERS = {
    "bm25_1_step": ("bm25", 1),
    "rag_1_step": ("rag", 1),
    "rag_2_step": ("rag", 2),
    "hipporag2-0.5_1_step": ("hipporag2-0.5", 1),
    "splade_v3_1_step": ("splade_v3", 1),
    "colbertv2_1_step": ("colbertv2", 1),
}
DEFAULT_DIRECT_OUTPUTS = tuple(DIRECT_RETRIEVERS)
ACTION_OUTPUTS = ("rrf_1_step", "no_retrieval_0_step")
RRF_SOURCES = ("rag_1_step", "hipporag2-0.5_1_step", "bm25_1_step", "splade_v3_1_step", "colbertv2_1_step")
REQUIRED_KEYS = {
    "id",
    "question",
    "top_k_indices",
    "top_k_scores",
    "f1",
    "ndcg",
    "latency",
    "pred_ans",
}


def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def sanitize_for_path(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "command"


def init_run_log_dir(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = os.getenv("SLURM_JOB_ID", "").strip()
    suffix = f"_{job_id}" if job_id else ""
    run_log_dir = Path(args.log_dir) / f"run_{timestamp}{suffix}"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    args._run_log_dir = run_log_dir
    args._command_counter = 0
    log(f"Run logs: {run_log_dir}")
    return run_log_dir


def command_log_path(args, dataset, label):
    args._command_counter += 1
    filename = (
        f"{args._command_counter:04d}_"
        f"{sanitize_for_path(dataset)}_"
        f"{sanitize_for_path(label)}.log"
    )
    return args._run_log_dir / filename


def write_command_record(run_log_dir, record, current=True):
    if run_log_dir is None:
        return
    run_log_dir = Path(run_log_dir)
    if current:
        path = run_log_dir / "current_command.json"
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    history_path = run_log_dir / "command_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_datasets(raw_value):
    datasets = []
    seen = set()
    for item in [part.strip() for part in raw_value.split(",") if part.strip()]:
        expanded = DATASET_GROUPS.get(item, [item])
        for dataset in expanded:
            if dataset not in seen:
                datasets.append(dataset)
                seen.add(dataset)
    return datasets


def parse_stages(raw_value):
    if raw_value.strip().lower() == "all":
        return {"artifacts", "direct", "actions"}
    stages = {item.strip().lower() for item in raw_value.split(",") if item.strip()}
    invalid = stages - {"artifacts", "direct", "actions"}
    if invalid:
        raise ValueError(f"Unsupported stage(s): {sorted(invalid)}")
    return stages


def parse_direct_outputs(raw_value):
    if raw_value is None or raw_value.strip().lower() == "default":
        return list(DEFAULT_DIRECT_OUTPUTS)
    aliases = {
        "bm25": "bm25_1_step",
        "rag": "rag_1_step",
        "rag_1": "rag_1_step",
        "rag_2": "rag_2_step",
        "hipporag2-0.5": "hipporag2-0.5_1_step",
        "hipporag2": "hipporag2-0.5_1_step",
        "splade_v3": "splade_v3_1_step",
        "colbertv2": "colbertv2_1_step",
    }
    outputs = []
    for item in [part.strip() for part in raw_value.split(",") if part.strip()]:
        output_name = aliases.get(item, item)
        if output_name not in DIRECT_RETRIEVERS:
            raise ValueError(f"Unsupported retriever/output: {item}")
        outputs.append(output_name)
    return outputs


def result_path(results_base_dir, dataset, output_name):
    return Path(results_base_dir) / dataset / f"{output_name}.jsonl"


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def load_current_samples(dataset):
    samples = load_qa(dataset)
    id_to_question = {}
    question_to_ids = {}
    for sample in samples:
        sample_id = sample["id"]
        question = sample["question"]
        if sample_id in id_to_question:
            raise ValueError(f"Duplicate current sample id in {dataset}: {sample_id}")
        id_to_question[sample_id] = question
        question_to_ids.setdefault(question, []).append(sample_id)
    return samples, id_to_question, question_to_ids


def backup_path(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"{path}.bak.{timestamp}")


def repair_jsonl(path, dataset, id_to_question, question_to_ids, dry_run=False):
    path = Path(path)
    if not path.exists():
        return {"exists": False, "done": 0, "ids": set(), "changed": False, "dropped": 0, "remapped": 0}

    rows = []
    changed = False
    dropped = 0
    remapped = 0
    seen = set()
    malformed = False

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                changed = True
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed = True
                changed = True
                log(f"{path}: quarantining malformed trailing content starting at line {line_no}")
                break

            missing = sorted(REQUIRED_KEYS - row.keys())
            if missing:
                raise ValueError(f"{path}:{line_no} missing required keys: {missing}")

            row_id = row.get("id")
            question = row.get("question")
            if row_id in id_to_question:
                if question != id_to_question[row_id]:
                    raise ValueError(
                        f"{path}:{line_no} id {row_id!r} has a question that does not match current {dataset}"
                    )
            else:
                matches = question_to_ids.get(question, [])
                if len(matches) != 1:
                    raise ValueError(
                        f"{path}:{line_no} stale id {row_id!r} cannot be safely mapped by exact question"
                    )
                mapped_id = matches[0]
                if mapped_id in seen:
                    dropped += 1
                    changed = True
                    continue
                row["id"] = mapped_id
                remapped += 1
                changed = True
                row_id = mapped_id

            if row_id in seen:
                raise ValueError(f"{path}:{line_no} duplicate current id after repair: {row_id!r}")
            seen.add(row_id)
            rows.append(row)

    if changed and not dry_run:
        backup = backup_path(path)
        shutil.copy2(path, backup)
        atomic_write_jsonl(path, rows)
        log(
            f"Repaired {path}: backup={backup}, kept={len(rows)}, "
            f"remapped={remapped}, dropped={dropped}, malformed={malformed}"
        )
    elif changed:
        log(
            f"Would repair {path}: kept={len(rows)}, remapped={remapped}, "
            f"dropped={dropped}, malformed={malformed}"
        )

    return {
        "exists": True,
        "done": len(seen),
        "ids": seen,
        "changed": changed,
        "dropped": dropped,
        "remapped": remapped,
    }


def output_status(path, current_ids, projected_ids=None):
    if projected_ids is not None:
        ids = set(projected_ids)
        done = len(ids & current_ids)
        return done, len(current_ids - ids)

    path = Path(path)
    if not path.exists():
        return 0, len(current_ids)
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["id"])
    done = len(ids & current_ids)
    return done, len(current_ids - ids)


def write_dataset_status(args, dataset, current_ids, output_names, phase):
    status = {
        "dataset": dataset,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
        "expected_samples": len(current_ids),
        "outputs": {},
    }
    for output_name in output_names:
        done, pending = output_status(
            result_path(args.results_base_dir, dataset, output_name),
            current_ids,
        )
        status["outputs"][output_name] = {
            "done": done,
            "pending": pending,
            "path": str(result_path(args.results_base_dir, dataset, output_name)),
        }

    path = args._run_log_dir / f"status_{sanitize_for_path(dataset)}.json"
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    log(f"{dataset}: wrote status snapshot ({phase}) to {path}")


def run_command(command, dry_run=False, env=None, log_path=None):
    log(f"Command: {' '.join(command)}")
    run_log_dir = log_path.parent if log_path is not None else None
    started_record = {
        "timestamp": datetime.now().isoformat(),
        "status": "dry_run" if dry_run else "running",
        "command": command,
        "log_path": str(log_path) if log_path is not None else None,
    }
    write_command_record(run_log_dir, started_record)
    if dry_run:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as f:
                f.write("DRY RUN command:\n")
                f.write(" ".join(command) + "\n")
            log(f"Dry-run command log: {log_path}")
        return

    if log_path is None:
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env)
        if completed.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = datetime.now()
    log(f"Streaming command output to: {log_path}")
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"Started: {start_time.isoformat()}\n")
        f.write(f"Command: {' '.join(command)}\n\n")
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        return_code = process.wait()
        end_time = datetime.now()
        f.write(f"\nFinished: {end_time.isoformat()}\n")
        f.write(f"Return code: {return_code}\n")

    if return_code != 0:
        log(f"Command failed with exit code {return_code}; see {log_path}")
        failed_record = {
            "timestamp": datetime.now().isoformat(),
            "status": "failed",
            "return_code": return_code,
            "command": command,
            "log_path": str(log_path),
        }
        write_command_record(run_log_dir, failed_record)
        raise RuntimeError(
            f"Command failed with exit code {return_code}: {' '.join(command)}. "
            f"Log: {log_path}"
        )

    completed_record = {
        "timestamp": datetime.now().isoformat(),
        "status": "completed",
        "return_code": return_code,
        "command": command,
        "log_path": str(log_path),
    }
    write_command_record(run_log_dir, completed_record)
    log(f"Command finished successfully; log={log_path}")


def build_env(args):
    env = os.environ.copy()
    if args.artifact_root:
        env[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(env, args.artifact_root, override=True)
    if args.kg_backend:
        env["RAGCOMPASS_KG_INDEX_BACKEND"] = args.kg_backend
    return env


def expected_kg_metadata(args, dataset):
    return build_kg_metadata(args.llm, dataset, args.text_encoder, load_corpus(dataset))


def verify_kg_artifacts(args, dataset):
    index_dir = kg_index_dir(args.llm, dataset, artifact_root=args.artifact_root)
    emb_dir = kg_embedding_dir(args.llm, dataset, args.text_encoder, artifact_root=args.artifact_root)
    required_files = [
        index_dir / "triples.json",
        index_dir / "triple2doc_id.json",
        index_dir / "entities.json",
        index_dir / "synonym_edges.tsv",
        emb_dir / "triple_emb.parquet",
        emb_dir / "entity_emb.parquet",
        embedding_file(args.text_encoder, dataset, artifact_root=args.artifact_root),
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    metadata_path = kg_metadata_file(args.llm, dataset, artifact_root=args.artifact_root)
    metadata = load_json_if_exists(metadata_path)
    if metadata is None:
        return False, "missing metadata", missing

    expected = expected_kg_metadata(args, dataset)
    for key in ("llm", "dataset", "text_encoder", "corpus_doc_count", "corpus_sha256"):
        if metadata.get(key) != expected.get(key):
            return False, f"metadata mismatch for {key}", missing
    if missing:
        return False, "missing files", missing
    return True, "ok", []


def ensure_rag_artifacts(args, dataset, dry_run=False):
    path = embedding_file(args.text_encoder, dataset, artifact_root=args.artifact_root)
    if path.exists():
        log(f"{dataset}: corpus embeddings present: {path}")
    else:
        log(f"{dataset}: corpus embeddings missing; main_hf.py will generate: {path}")
    if dry_run:
        return


def ensure_kg_artifacts(args, dataset, env):
    ok, reason, missing = verify_kg_artifacts(args, dataset)
    if ok:
        log(f"{dataset}: verified HippoRAG artifacts under {kg_index_dir(args.llm, dataset, args.artifact_root)}")
        return

    log(f"{dataset}: HippoRAG artifacts not verified ({reason}).")
    if missing:
        for path in missing:
            log(f"{dataset}: missing artifact: {path}")

    core_files = [
        kg_index_dir(args.llm, dataset, args.artifact_root) / "triples.json",
        kg_index_dir(args.llm, dataset, args.artifact_root) / "triple2doc_id.json",
        kg_index_dir(args.llm, dataset, args.artifact_root) / "entities.json",
    ]
    metadata_missing_or_mismatch = reason != "missing files"
    if metadata_missing_or_mismatch or any(not path.exists() for path in core_files):
        kg_python = args.kg_python or sys.executable
        command = [
            kg_python,
            "kg_index.py",
            "--dataset",
            dataset,
            "--llm",
            args.llm,
            "--text_encoder",
            args.text_encoder,
            "--force_rebuild",
        ]
        if args.artifact_root:
            command.extend(["--artifact_root", args.artifact_root])
        run_command(
            command,
            args.dry_run,
            env,
            log_path=command_log_path(args, dataset, "kg_index"),
        )

    command = [
        sys.executable,
        "kg_emb.py",
        "--dataset",
        dataset,
        "--llm",
        args.llm,
        "--text_encoder",
        args.text_encoder,
    ]
    if args.emb_port is not None:
        command.extend(["--emb_port", str(args.emb_port)])
    if args.artifact_root:
        command.extend(["--artifact_root", args.artifact_root])
    run_command(
        command,
        args.dry_run,
        env,
        log_path=command_log_path(args, dataset, "kg_emb"),
    )


def run_direct_retriever(args, dataset, output_name, current_ids, env, projected_ids=None):
    retriever, max_steps = DIRECT_RETRIEVERS[output_name]
    path = result_path(args.results_base_dir, dataset, output_name)
    done, pending = output_status(path, current_ids, projected_ids=projected_ids)
    log(f"{dataset}/{output_name}: done={done}, pending={pending}, output={path}")
    if pending == 0:
        return

    if retriever == "rag":
        ensure_rag_artifacts(args, dataset, args.dry_run)
    elif retriever.startswith("hipporag2"):
        ensure_kg_artifacts(args, dataset, env)

    command = [
        sys.executable,
        "main_hf.py",
        "--dataset",
        dataset,
        "--retriever",
        retriever,
        "--max_steps",
        str(max_steps),
        "--llm",
        args.llm,
        "--text_encoder",
        args.text_encoder,
        "--top_k",
        str(args.top_k),
        "--n_proc",
        str(args.n_proc),
        "--results_base_dir",
        args.results_base_dir,
    ]
    if args.model_id:
        command.extend(["--model_id", args.model_id])
    if args.emb_port is not None and retriever in {"rag", "hipporag2-0.5"}:
        command.extend(["--emb_port", str(args.emb_port)])
    if args.artifact_root:
        command.extend(["--artifact_root", args.artifact_root])
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    run_command(
        command,
        args.dry_run,
        env,
        log_path=command_log_path(args, dataset, output_name),
    )


def run_actions(args, dataset, current_ids, env, projected_ids_by_output=None):
    projected_ids_by_output = projected_ids_by_output or {}
    for source in RRF_SOURCES:
        done, pending = output_status(
            result_path(args.results_base_dir, dataset, source),
            current_ids,
            projected_ids=projected_ids_by_output.get(source),
        )
        if pending:
            log(f"{dataset}: skipping actions because {source} still has {pending} pending samples")
            return

    action_pending = []
    for output_name in ACTION_OUTPUTS:
        done, pending = output_status(
            result_path(args.results_base_dir, dataset, output_name),
            current_ids,
            projected_ids=projected_ids_by_output.get(output_name),
        )
        action_pending.append(pending)
        log(f"{dataset}/{output_name}: done={done}, pending={pending}")
    if not any(action_pending):
        return

    command = [
        sys.executable,
        "-m",
        "baselines.rrf",
        "--backend",
        args.action_backend,
        "--llm",
        args.llm,
        "--datasets",
        dataset,
        "--results_base_dir",
        args.results_base_dir,
        "--top_k",
        str(args.top_k),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
    ]
    if args.model_id:
        command.extend(["--model_id", args.model_id])
    if args.llm_port is not None:
        command.extend(["--llm_port", str(args.llm_port)])
    if args.model_path:
        command.extend(["--model_path", args.model_path])
    if args.artifact_root:
        command.extend(["--artifact_root", args.artifact_root])
    if args.max_samples is not None:
        command.extend(["--max_samples", str(args.max_samples)])
    run_command(
        command,
        args.dry_run,
        env,
        log_path=command_log_path(args, dataset, "actions_rrf_no_retrieval"),
    )


def process_dataset(args, dataset, direct_outputs, stages, env):
    samples, id_to_question, question_to_ids = load_current_samples(dataset)
    current_ids = set(id_to_question)
    log(f"========== {dataset}: current samples={len(samples)} ==========")
    all_outputs = list(direct_outputs) + list(ACTION_OUTPUTS)
    write_dataset_status(args, dataset, current_ids, all_outputs, phase="start")

    repair_states = {}
    for output_name in all_outputs:
        repair_states[output_name] = repair_jsonl(
            result_path(args.results_base_dir, dataset, output_name),
            dataset,
            id_to_question,
            question_to_ids,
            dry_run=args.dry_run,
        )

    if "artifacts" in stages:
        if any(DIRECT_RETRIEVERS[name][0] == "rag" for name in direct_outputs):
            ensure_rag_artifacts(args, dataset, args.dry_run)
        if any(DIRECT_RETRIEVERS[name][0].startswith("hipporag2") for name in direct_outputs):
            ensure_kg_artifacts(args, dataset, env)

    if "direct" in stages:
        for output_name in direct_outputs:
            projected_ids = repair_states.get(output_name, {}).get("ids") if args.dry_run else None
            run_direct_retriever(args, dataset, output_name, current_ids, env, projected_ids=projected_ids)

    if "actions" in stages:
        projected = {
            name: state.get("ids")
            for name, state in repair_states.items()
            if args.dry_run and state.get("exists")
        }
        run_actions(args, dataset, current_ids, env, projected_ids_by_output=projected)

    write_dataset_status(args, dataset, current_ids, all_outputs, phase="end")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Resume expanded 5k dataset preprocessing.")
    parser.add_argument("--datasets", default="nq,musique,hotpotqa,2wikimultihopqa")
    parser.add_argument("--artifact_root", default=None)
    parser.add_argument("--results_base_dir", default="Llama-3.1-8B-Instruct_Qwen3-Embedding-0.6B")
    parser.add_argument("--llm", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--text_encoder", default="Qwen3-Embedding-0.6B")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--n_proc", type=int, default=1)
    parser.add_argument("--emb_port", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--log_dir", default="logs/preprocess_expanded")
    parser.add_argument("--stages", default="all", help="Comma-separated: artifacts,direct,actions or all.")
    parser.add_argument("--retrievers", default="default", help="Comma-separated direct retrievers or output names.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--action_backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument(
        "--kg_backend",
        choices=["vllm", "hf"],
        default=None,
        help=(
            "Backend used by kg_index.py when KG indexes must be built. "
            "Defaults to kg_index.py behavior, usually vllm."
        ),
    )
    parser.add_argument(
        "--kg_python",
        default=None,
        help=(
            "Optional Python executable used only for kg_index.py. This allows vLLM "
            "KG extraction to run in a dedicated environment while retrieval and KG "
            "embedding continue in the current environment."
        ),
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Log dataset failures and continue with later datasets instead of stopping the run.",
    )
    parser.add_argument("--llm_port", type=int, default=None)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser


def main():
    args = build_arg_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")
    if args.n_proc <= 0:
        raise ValueError("--n_proc must be positive")
    if args.action_backend == "vllm" and args.llm_port is None:
        raise ValueError("--llm_port is required when --action_backend vllm")
    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(os.environ, args.artifact_root, override=True)
    init_run_log_dir(args)

    datasets = resolve_datasets(args.datasets)
    stages = parse_stages(args.stages)
    direct_outputs = parse_direct_outputs(args.retrievers)
    env = build_env(args)

    log(f"Datasets: {datasets}")
    log(f"Stages: {sorted(stages)}")
    log(f"Direct outputs: {direct_outputs}")
    log(f"Artifact root: {args.artifact_root or os.getenv(ARTIFACT_ROOT_ENV, '<repo-relative>')}")
    if args.artifact_root:
        log(f"HF/model cache root: {model_cache_dir('huggingface', args.artifact_root)}")
        log(f"SentenceTransformers cache root: {model_cache_dir('sentence_transformers', args.artifact_root)}")
        log(f"FastEmbed cache root: {model_cache_dir('fastembed', args.artifact_root)}")
        log(f"Retriever cache root: {Path(args.artifact_root) / 'retriever_cache'}")
    log(f"Results base dir: {args.results_base_dir}")
    log(f"KG backend: {args.kg_backend or os.getenv('RAGCOMPASS_KG_INDEX_BACKEND', '<kg_index.py default>')}")
    log(f"KG Python: {args.kg_python or sys.executable}")
    log(f"Continue on error: {args.continue_on_error}")
    log(f"Dry run: {args.dry_run}")

    failures = []
    for dataset in datasets:
        try:
            process_dataset(args, dataset, direct_outputs, stages, env)
        except Exception as exc:
            failure = {
                "dataset": dataset,
                "timestamp": datetime.now().isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            failure_path = args._run_log_dir / "failures.jsonl"
            with failure_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(failure, ensure_ascii=False) + "\n")
            log(f"{dataset}: FAILED ({type(exc).__name__}: {exc})")
            log(f"{dataset}: failure recorded in {failure_path}")
            if not args.continue_on_error:
                raise

    if failures:
        log(f"Expanded preprocessing finished with {len(failures)} dataset failure(s).")
        if not args.continue_on_error:
            raise RuntimeError(f"Preprocessing failed for {len(failures)} dataset(s).")
    else:
        log("Expanded preprocessing orchestration complete.")


if __name__ == "__main__":
    main()
