import gc
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from ..artifacts import retriever_cache_dir
from .base import build_doc_title_to_index, resolve_excluded_indices


def sanitize_name(name):
    return name.replace("/", "_").replace("-", "_")


def int_env(name, default, min_value=1):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    value = int(raw_value)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def cache_base_dir(default_dir, retriever_name):
    explicit_dir = os.getenv("RAGCOMPASS_RETRIEVER_CACHE_DIR", "").strip()
    if explicit_dir:
        return os.path.join(explicit_dir, retriever_name)
    if default_dir != ".retriever_cache/colbertv2":
        return default_dir
    return str(retriever_cache_dir(retriever_name))


class ColBERTv2Retriever:
    def __init__(
        self,
        corpus,
        top_k,
        dataset,
        model_name="colbert-ir/colbertv2.0",
        index_base_dir=".retriever_cache/colbertv2",
        batch_size=8,
    ):
        try:
            from pylate import indexes, models, retrieve
        except ImportError as exc:
            raise ImportError(
                "ColBERTv2Retriever requires pylate and a Voyager-backed index. "
                "Install the project requirements or run `pip install pylate voyager`. "
                f"Original import error: {type(exc).__name__}: {exc}"
            ) from exc

        self.corpus = corpus
        self.top_k = top_k
        self.doc_title_to_index = build_doc_title_to_index(corpus)
        self.batch_size = int_env("RAGCOMPASS_COLBERTV2_BATCH_SIZE", batch_size)
        self.index_add_batch_size = int_env("RAGCOMPASS_COLBERTV2_INDEX_ADD_BATCH_SIZE", 64)
        self.index_m = int_env("RAGCOMPASS_COLBERTV2_VOYAGER_M", 16)
        self.index_ef_construction = int_env(
            "RAGCOMPASS_COLBERTV2_VOYAGER_EF_CONSTRUCTION",
            100,
        )
        self.index_ef_search = int_env("RAGCOMPASS_COLBERTV2_VOYAGER_EF_SEARCH", 100)
        self.model_device = os.getenv("RAGCOMPASS_COLBERTV2_DEVICE", "").strip() or None

        print(
            f"Loading ColBERTv2 model {model_name} on "
            f"{self.model_device or 'default device'} with batch_size={self.batch_size}",
            flush=True,
        )
        model_kwargs = {
            "model_name_or_path": model_name,
        }
        if self.model_device is not None:
            model_kwargs["device"] = self.model_device
        self.model = models.ColBERT(**model_kwargs)

        resolved_index_base_dir = cache_base_dir(index_base_dir, "colbertv2")
        safe_dataset = sanitize_name(dataset)
        index_folder = os.path.join(resolved_index_base_dir, safe_dataset, f"pid_{os.getpid()}")
        os.makedirs(index_folder, exist_ok=True)

        self.index = indexes.Voyager(
            index_folder=index_folder,
            index_name="index",
            override=True,
            M=self.index_m,
            ef_construction=self.index_ef_construction,
            ef_search=self.index_ef_search,
        )

        print(
            f"Indexing {len(corpus)} documents for dataset={dataset} into {index_folder} "
            f"with Voyager M={self.index_m}, ef_construction={self.index_ef_construction}, "
            f"ef_search={self.index_ef_search}, index_add_batch_size={self.index_add_batch_size}",
            flush=True,
        )
        for start in tqdm(
            range(0, len(corpus), self.batch_size),
            desc="Building ColBERTv2 index",
        ):
            end = min(start + self.batch_size, len(corpus))
            print(
                f"Encoding ColBERTv2 documents {start}:{end} for dataset={dataset}",
                flush=True,
            )
            document_ids = [str(idx) for idx in range(start, end)]
            document_embeddings = self.model.encode(
                sentences=corpus[start:end],
                batch_size=self.batch_size,
                is_query=False,
                show_progress_bar=False,
            )
            print(
                f"Adding ColBERTv2 documents {start}:{end} to Voyager index",
                flush=True,
            )
            maybe_index = self.index.add_documents(
                documents_ids=document_ids,
                documents_embeddings=document_embeddings,
                batch_size=self.index_add_batch_size,
            )
            if maybe_index is not None:
                self.index = maybe_index
            del document_embeddings
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.retriever = retrieve.ColBERT(index=self.index)
        self.reset()

    def reset(self):
        self.chunk_scores = defaultdict(float)

    def __call__(self, query, excluded_ids=None):
        excluded_indices = resolve_excluded_indices(
            self.doc_title_to_index,
            excluded_ids,
        )
        query_embeddings = self.model.encode(
            [query],
            is_query=True,
            batch_size=1,
            show_progress_bar=False,
        )
        scores = self.retriever.retrieve(
            queries_embeddings=query_embeddings,
            k=self.top_k + len(excluded_indices),
        )[0]

        latest_scores = {}
        for item in scores:
            idx = int(item["id"])
            score = float(item["score"])
            if idx in excluded_indices:
                continue
            self.chunk_scores[idx] = max(self.chunk_scores[idx], score)
            latest_scores[idx] = score

        top_k_indices = sorted(
            self.chunk_scores,
            key=lambda chunk_idx: self.chunk_scores[chunk_idx],
            reverse=True,
        )[:self.top_k]
        top_k_docs = [self.corpus[idx] for idx in top_k_indices]
        top_k_scores = [
            float(latest_scores.get(idx, self.chunk_scores[idx]))
            for idx in top_k_indices
        ]

        return top_k_indices, top_k_docs, top_k_scores
