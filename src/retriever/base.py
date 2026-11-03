from collections import defaultdict

from ..emb.infer import init_emb_client


def get_doc_title(doc):
    return doc.split('\n', 1)[0]


def build_doc_title_to_index(corpus):
    return {
        get_doc_title(doc): idx
        for idx, doc in enumerate(corpus)
    }


def resolve_excluded_indices(doc_title_to_index, excluded_ids):
    if not excluded_ids:
        return set()

    return {
        doc_title_to_index[doc_id]
        for doc_id in excluded_ids
        if doc_id in doc_title_to_index
    }


def select_top_k_with_exclusions(corpus, chunk_scores, scores, top_k, excluded_indices):
    ranked_indices = [
        idx for idx in sorted(
            chunk_scores, key=lambda chunk_idx: chunk_scores[chunk_idx], reverse=True
        )
        if idx not in excluded_indices
    ]
    top_k_indices = ranked_indices[:top_k]

    if len(top_k_indices) < top_k:
        for idx in scores.argsort()[::-1]:
            idx = int(idx)
            if idx in excluded_indices or idx in top_k_indices:
                continue
            top_k_indices.append(idx)
            if len(top_k_indices) == top_k:
                break

    top_k_docs = [corpus[idx] for idx in top_k_indices]
    top_k_scores = [float(scores[idx]) for idx in top_k_indices]
    return top_k_indices, top_k_docs, top_k_scores


class BaseRetriever:
    def __init__(self, text_encoder, emb_port, corpus_emb, top_k, corpus):
        self.text_encoder = text_encoder
        self.top_k = top_k
        self.corpus = corpus
        self.emb_client = init_emb_client(emb_port, text_encoder)
        self.corpus_emb = corpus_emb
        self.doc_title_to_index = build_doc_title_to_index(corpus)

    def reset(self):
        self.chunk_scores = defaultdict(float)
    
    def retrieve(self, top_k_indices, scores, excluded_ids=None):
        excluded_indices = resolve_excluded_indices(
            self.doc_title_to_index,
            excluded_ids,
        )
        for idx in top_k_indices:
            idx = int(idx)
            if idx in excluded_indices:
                continue
            self.chunk_scores[idx] = max(
                self.chunk_scores[idx], float(scores[idx]))

        return select_top_k_with_exclusions(
            self.corpus,
            self.chunk_scores,
            scores,
            self.top_k,
            excluded_indices,
        )
