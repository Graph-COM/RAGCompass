import hashlib
import json
import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "RAGCOMPASS_ARTIFACT_ROOT"
ARTIFACT_REPO_ENV = "RAGCOMPASS_ARTIFACT_REPO_ID"
KG_METADATA_FILE = "metadata.json"


def resolve_artifact_root(artifact_root=None):
    root = artifact_root or os.getenv(ARTIFACT_ROOT_ENV, "").strip()
    if not root:
        return Path(".")
    return Path(root).expanduser()


def artifact_path(*parts, artifact_root=None):
    return resolve_artifact_root(artifact_root).joinpath(*parts)


def embedding_dir(text_encoder, dataset, artifact_root=None):
    return artifact_path(text_encoder, dataset, artifact_root=artifact_root)


def embedding_file(text_encoder, dataset, filename="chunk_emb.parquet", artifact_root=None):
    return embedding_dir(text_encoder, dataset, artifact_root=artifact_root) / filename


def kg_index_dir(llm, dataset, artifact_root=None):
    return artifact_path(f"{llm}_kg_index", dataset, artifact_root=artifact_root)


def kg_embedding_dir(llm, dataset, text_encoder, artifact_root=None):
    return kg_index_dir(llm, dataset, artifact_root=artifact_root) / text_encoder


def kg_metadata_file(llm, dataset, artifact_root=None):
    return kg_index_dir(llm, dataset, artifact_root=artifact_root) / KG_METADATA_FILE


def retriever_cache_dir(retriever_name, artifact_root=None):
    return artifact_path("retriever_cache", retriever_name, artifact_root=artifact_root)


def model_cache_dir(cache_name, artifact_root=None):
    return artifact_path("model_cache", cache_name, artifact_root=artifact_root)


def apply_cache_environment(env=None, artifact_root=None, override=False):
    env = env if env is not None else os.environ
    root = resolve_artifact_root(artifact_root)
    if str(root) == "." and not os.getenv(ARTIFACT_ROOT_ENV, "").strip():
        return env

    cache_paths = {
        "HF_HOME": model_cache_dir("huggingface", artifact_root=root),
        "HF_HUB_CACHE": model_cache_dir("huggingface/hub", artifact_root=root),
        "TRANSFORMERS_CACHE": model_cache_dir("huggingface/transformers", artifact_root=root),
        "SENTENCE_TRANSFORMERS_HOME": model_cache_dir("sentence_transformers", artifact_root=root),
        "TORCH_HOME": model_cache_dir("torch", artifact_root=root),
        "XDG_CACHE_HOME": model_cache_dir("xdg", artifact_root=root),
        "FASTEMBED_CACHE_PATH": model_cache_dir("fastembed", artifact_root=root),
        "RAGCOMPASS_FASTEMBED_CACHE_DIR": model_cache_dir("fastembed", artifact_root=root),
    }
    for name, path in cache_paths.items():
        if override or not env.get(name):
            env[name] = str(path)
    for path in cache_paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    return env


def path_for_log(path):
    return str(Path(path))


def artifact_repo_path(path, artifact_root=None):
    path = Path(path).expanduser()
    root = resolve_artifact_root(artifact_root).expanduser()
    try:
        return path.resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return path.as_posix().lstrip("/")


def artifact_repo_id():
    """Optional Hugging Face dataset repo that mirrors the artifact root; unset means local files only."""
    return os.getenv(ARTIFACT_REPO_ENV, "").strip() or None


def upload_artifact(path):
    repo_id = artifact_repo_id()
    if repo_id is None:
        return
    from huggingface_hub import upload_file

    try:
        upload_file(
            path_or_fileobj=path,
            path_in_repo=artifact_repo_path(path),
            repo_id=repo_id,
            repo_type="dataset",
        )
    except Exception as e:
        print(
            f"Warning: could not upload {path} to Hugging Face "
            f"({type(e).__name__}: {e}). Continuing with local file only."
        )


def download_artifact(path):
    repo_id = artifact_repo_id()
    if repo_id is None:
        raise FileNotFoundError(
            f"{path} not found. Build it locally (kg_index.py / kg_emb.py), "
            f"or set {ARTIFACT_REPO_ENV} to a dataset repo that mirrors the artifact root."
        )
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=artifact_repo_path(path),
        local_dir=str(resolve_artifact_root()),
        local_dir_use_symlinks=False,
    )


def atomic_replace_path(path, suffix=".tmp"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f"{path.name}.{os.getpid()}{suffix}")


def atomic_write_json(path, payload, indent=2):
    path = Path(path)
    tmp_path = atomic_replace_path(path)
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
    os.replace(tmp_path, path)


def corpus_signature(corpus_iterable):
    hasher = hashlib.sha256()
    count = 0
    for doc in corpus_iterable:
        encoded = str(doc).encode("utf-8")
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
        count += 1
    return {
        "corpus_doc_count": count,
        "corpus_sha256": hasher.hexdigest(),
    }


def build_kg_metadata(llm, dataset, text_encoder, corpus_iterable):
    metadata = corpus_signature(corpus_iterable)
    metadata.update({
        "llm": llm,
        "dataset": dataset,
        "text_encoder": text_encoder,
    })
    return metadata


def load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
