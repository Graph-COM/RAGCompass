import os
import gc
import inspect
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ..artifacts import model_cache_dir
from .base import (
    build_doc_title_to_index,
    resolve_excluded_indices,
    select_top_k_with_exclusions,
)


def int_env(name, default, min_value=1):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default

    value = int(raw_value)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def optional_int_env(name):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return None

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


class SPLADEv3Retriever:
    def __init__(
        self,
        corpus,
        top_k,
        model_name="naver/splade-v3",
        batch_size=8,
    ):
        print("SPLADEv3Retriever: importing sentence_transformers.SparseEncoder", flush=True)
        try:
            from sentence_transformers import SparseEncoder
        except ImportError as exc:
            raise ImportError(
                "SPLADEv3Retriever requires sentence-transformers with SparseEncoder "
                "support. Install the project requirements or run "
                "`pip install -U sentence-transformers`."
            ) from exc
        print("SPLADEv3Retriever: imported SparseEncoder successfully", flush=True)

        self.corpus = corpus
        self.top_k = top_k
        self.batch_size = int_env("RAGCOMPASS_SPLADE_BATCH_SIZE", batch_size)
        self.max_active_dims = optional_int_env("RAGCOMPASS_SPLADE_MAX_ACTIVE_DIMS")
        self.doc_title_to_index = build_doc_title_to_index(corpus)
        self.doc_sparse_embedding_batches = []

        print(
            f"Loading SPLADEv3 model {model_name} with batch_size={self.batch_size}, "
            f"max_active_dims={self.max_active_dims or 'default'}",
            flush=True,
        )
        print(f"SPLADEv3Retriever: constructing SparseEncoder({model_name})", flush=True)
        model_kwargs = {}
        cache_folder = os.getenv("SENTENCE_TRANSFORMERS_HOME", "").strip()
        if not cache_folder:
            cache_folder = str(model_cache_dir("sentence_transformers"))
        signature = inspect.signature(SparseEncoder)
        if "cache_folder" in signature.parameters:
            model_kwargs["cache_folder"] = cache_folder
        print(f"SPLADEv3Retriever: sentence-transformers cache dir={cache_folder}", flush=True)
        self.model = SparseEncoder(model_name, **model_kwargs)
        print("SPLADEv3Retriever: SparseEncoder constructed successfully", flush=True)
        if hasattr(self.model, "similarity_fn_name"):
            self.model.similarity_fn_name = "dot"

        encode_kwargs = {
            "batch_size": self.batch_size,
            "show_progress_bar": True,
        }
        if self.max_active_dims is not None:
            encode_kwargs["max_active_dims"] = self.max_active_dims

        print(
            f"Encoding {len(corpus)} SPLADEv3 documents in batches of {self.batch_size}",
            flush=True,
        )
        for start in tqdm(
            range(0, len(corpus), self.batch_size),
            desc="Building SPLADEv3 document batches",
        ):
            end = min(start + self.batch_size, len(corpus))
            batch_embeddings = self.model.encode_document(corpus[start:end], **encode_kwargs)
            if hasattr(batch_embeddings, "cpu"):
                batch_embeddings = batch_embeddings.cpu()
            self.doc_sparse_embedding_batches.append(batch_embeddings)
            gc.collect()

        self.reset()

    def reset(self):
        self.chunk_scores = defaultdict(float)

    def __call__(self, query, excluded_ids=None):
        encode_kwargs = {
            "batch_size": 1,
            "show_progress_bar": False,
        }
        if self.max_active_dims is not None:
            encode_kwargs["max_active_dims"] = self.max_active_dims

        query_embedding = self.model.encode_query(query, **encode_kwargs)
        if hasattr(query_embedding, "cpu"):
            query_embedding = query_embedding.cpu()

        scores = np.empty(len(self.corpus), dtype=np.float32)
        cursor = 0
        for batch_embeddings in self.doc_sparse_embedding_batches:
            similarities = self.model.similarity(query_embedding, batch_embeddings)
            if hasattr(similarities, "detach"):
                batch_scores = similarities.detach().cpu().numpy()
            else:
                batch_scores = np.asarray(similarities)
            batch_scores = np.asarray(batch_scores, dtype=np.float32).reshape(-1)
            next_cursor = cursor + len(batch_scores)
            scores[cursor:next_cursor] = batch_scores
            cursor = next_cursor

        excluded_indices = resolve_excluded_indices(
            self.doc_title_to_index,
            excluded_ids,
        )
        top_k_indices = np.argsort(scores)[-self.top_k:][::-1]

        for idx in top_k_indices:
            idx = int(idx)
            if idx in excluded_indices:
                continue
            self.chunk_scores[idx] = max(self.chunk_scores[idx], float(scores[idx]))

        return select_top_k_with_exclusions(
            self.corpus,
            self.chunk_scores,
            scores,
            self.top_k,
            excluded_indices,
        )
