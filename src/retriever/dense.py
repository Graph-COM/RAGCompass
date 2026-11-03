import numpy as np

from .base import BaseRetriever
from ..emb.infer import emb_query

class DenseRetriever(BaseRetriever):
    def __call__(self, query, excluded_ids=None):
        query_emb = emb_query(
            self.text_encoder,
            self.emb_client, 
            query
        )
        similarities = np.dot(self.corpus_emb, query_emb)
        top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        return self.retrieve(top_k_indices, similarities, excluded_ids=excluded_ids)
