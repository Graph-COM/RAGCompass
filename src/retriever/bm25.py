import inspect
import numpy as np
import os

from collections import defaultdict

from fastembed import SparseTextEmbedding

from ..artifacts import model_cache_dir
from .base import (
    build_doc_title_to_index,
    resolve_excluded_indices,
    select_top_k_with_exclusions,
)


def sparse_dot_product(query_embedding, doc_embedding):
    query_map = {
        int(index): float(value)
        for index, value in zip(query_embedding.indices, query_embedding.values)
    }

    score = 0.0
    for index, value in zip(doc_embedding.indices, doc_embedding.values):
        score += query_map.get(int(index), 0.0) * float(value)

    return score


class BM25Retriever:
    def __init__(self, corpus, top_k, model_name="Qdrant/bm25"):
        self.corpus = corpus
        self.top_k = top_k
        model_kwargs = {"model_name": model_name}
        cache_dir = os.getenv("RAGCOMPASS_FASTEMBED_CACHE_DIR", "").strip()
        if not cache_dir:
            cache_dir = str(model_cache_dir("fastembed"))
        signature = inspect.signature(SparseTextEmbedding)
        if "cache_dir" in signature.parameters:
            model_kwargs["cache_dir"] = cache_dir
        print(f"BM25Retriever: FastEmbed cache dir={cache_dir}", flush=True)
        self.model = SparseTextEmbedding(**model_kwargs)
        self.doc_sparse_embeddings = list(self.model.embed(corpus))
        self.doc_title_to_index = build_doc_title_to_index(corpus)
        self.reset()

    def reset(self):
        self.chunk_scores = defaultdict(float)

    def __call__(self, query, excluded_ids=None):
        query_embedding = next(self.model.embed([query]))
        scores = np.array([
            sparse_dot_product(query_embedding, doc_embedding)
            for doc_embedding in self.doc_sparse_embeddings
        ], dtype=np.float32)

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
