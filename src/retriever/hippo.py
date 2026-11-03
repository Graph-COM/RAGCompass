import csv
import difflib
import igraph as ig
import json
import numpy as np
import os
import re

from collections import defaultdict

from .base import BaseRetriever
from ..artifacts import download_artifact, kg_index_dir
from ..emb.infer import emb_query2triple, emb_triples, emb_query
from ..kg_index import load_triples
from ..llm import init_tokenizer_client, query_llm
from ..llm_hf import query_llm_hf

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def parse_filter(response, candidate_facts):
    try:
        match = re.search(r"\[\[ ## fact_after_filter ## \]\](.*?)\[\[ ## completed ## \]\]", response, re.DOTALL)
        if match:
            json_string = match.group(1).strip()
            # Parse the JSON string and extract the 'fact' list
            data = json.loads(json_string)
            fact_list = data.get("fact", [])
        else:
            fact_list = []
    except json.JSONDecodeError:
        fact_list = []

    filter_facts = []
    for fact in fact_list:
        closest_matched_fact = difflib.get_close_matches(
            str(fact), [str(i) for i in candidate_facts], n=1, cutoff=0.0)[0]
        filter_facts.append(eval(closest_matched_fact))
        
    return filter_facts

class HippoRAG2Retriever(BaseRetriever):
    def __init__(
        self,
        llm,
        dataset,
        prompt_manager,
        text_encoder,
        emb_port,
        corpus_emb,
        top_k,
        corpus,
        damping,
        llm_port=None,
        model=None,
        tokenizer=None,
        adaptive_llm=None,
        top_k_link=5
    ):
        super().__init__(
            text_encoder=text_encoder,
            emb_port=emb_port,
            corpus_emb=corpus_emb,
            top_k=top_k,
            corpus=corpus
        )
        
        self.prompt_manager = prompt_manager
        
        self.llm = llm
        
        self.llm_port = llm_port
        if llm_port is not None:
            llm_tokenizer, llm_client = init_tokenizer_client(llm, llm_port)
            self.llm_tokenizer = llm_tokenizer
            self.llm_client = llm_client
        else:
            self.model = model
            self.tokenizer = tokenizer
        self.adaptive_llm = adaptive_llm
        
        self.triple_list = load_triples(llm, dataset)
        self.triple_emb = emb_triples(
            text_encoder,
            self.triple_list,
            emb_port,
            llm,
            dataset,
            upload=False
        )
        self.top_k_link = top_k_link
        self.damping = damping
        print(f'Using damping: {self.damping}')
        
        # Tracking latest facts for logging
        self.latest_candidate_facts = []
        self.latest_filter_facts = []
        
        self.create_g(index_dir=str(kg_index_dir(llm, dataset)))
    
    def reset(self):
        super().reset()
        self.latest_candidate_facts = []
        self.latest_filter_facts = []
    
    def create_g(self, index_dir):
        entities_file = os.path.join(index_dir, "entities.json")

        if not os.path.exists(entities_file):
            download_artifact(entities_file)   

        with open(entities_file, 'r', encoding='utf-8') as f:
            entities = json.load(f)
        
        self.entity2vertex_id = {entity: i for i, entity in enumerate(entities)}
        
        triple_to_docs_file = os.path.join(index_dir, "triple2doc_id.json")
        if not os.path.exists(triple_to_docs_file):
            download_artifact(triple_to_docs_file)
        
        with open(triple_to_docs_file, 'r', encoding='utf-8') as f:
            id_triple_to_docs = json.load(f)
        
        n2n_stats = defaultdict(float)
        self.n2d = dict()
        for triple_id, triple_i in enumerate(self.triple_list):
            u_i, _, v_i = triple_i
            
            u_id = self.entity2vertex_id[u_i]
            v_id = self.entity2vertex_id[v_i]
            
            n2n_stats[(u_id, v_id)] = n2n_stats[(u_id, v_id)] + len(id_triple_to_docs[str(triple_id)])
            n2n_stats[(v_id, u_id)] = n2n_stats[(v_id, u_id)] + len(id_triple_to_docs[str(triple_id)])
            
            self.n2d[u_id] = self.n2d.get(u_id, set()).union(set(id_triple_to_docs[str(triple_id)]))
            self.n2d[v_id] = self.n2d.get(v_id, set()).union(set(id_triple_to_docs[str(triple_id)]))

        synonym_edges_file = os.path.join(index_dir, "synonym_edges.tsv")
        
        if not os.path.exists(synonym_edges_file):
            download_artifact(synonym_edges_file)
        
        with open(synonym_edges_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                u_i, v_i, w_i = row
                u_id = self.entity2vertex_id[u_i]
                v_id = self.entity2vertex_id[v_i]
                
                n2n_stats[(u_id, v_id)] = float(w_i)
                n2n_stats[(v_id, u_id)] = float(w_i)
        
        self.graph = ig.Graph(directed=False)
        
        num_entities = len(entities)
        num_chunks = len(self.corpus)
        
        # Add entity and passage nodes
        self.graph.add_vertices(num_entities + num_chunks)
        
        # Set attributes for entity nodes
        self.graph.vs[0:num_entities]['type'] = 'entity'
        self.graph.vs[0:num_entities]['name'] = entities
        
        # Set attributes for passage nodes
        chunk_vertex_ids = range(num_entities, num_entities + num_chunks)
        self.graph.vs[chunk_vertex_ids]['type'] = 'chunk'
        self.graph.vs[chunk_vertex_ids]['name'] = [f'chunk_{i}' for i in range(num_chunks)]
        self.chunk_id2vertex_id = {i: i + num_entities for i in range(num_chunks)}
        
        # Add weighted edges.
        # Entity-Entity edges
        entity_entity_edges = list(n2n_stats.keys())
        entity_entity_weights = list(n2n_stats.values())
        
        self.graph.add_edges(entity_entity_edges)
        self.graph.es[self.graph.ecount() - len(entity_entity_edges):]['weight'] = entity_entity_weights
        
        # Entity-Chunk edges
        entity_chunk_edges = []
        for entity_id, chunk_ids in self.n2d.items():
            for chunk_id in chunk_ids:
                chunk_vertex_id = self.chunk_id2vertex_id[int(chunk_id)]
                entity_chunk_edges.append((entity_id, chunk_vertex_id))
        
        self.graph.add_edges(entity_chunk_edges, attributes={'weight': [1.0] * len(entity_chunk_edges)})
    
    def triple_linking(self, query):
        query2triple_emb = emb_query2triple(
            self.text_encoder,
            self.emb_client, 
            query
        )
        
        query_fact_scores = np.dot(self.triple_emb, query2triple_emb.T)
        query_fact_scores = min_max_normalize(query_fact_scores).flatten()
        candidate_fact_indices = np.argsort(
            query_fact_scores)[-self.top_k_link:][::-1].tolist()
        candidate_facts = [
            self.triple_list[i] for i in candidate_fact_indices
        ]
        candidate_scores = [query_fact_scores[i] for i in candidate_fact_indices]
        fact_score_map = {tuple(fact): float(score) for fact, score in zip(candidate_facts, candidate_scores)}

        # Triple filtering
        filter_prompt = self.prompt_manager.render_filter_triple(
            query, candidate_facts)
        
        if self.llm_port is None:
            response = query_llm_hf(
                filter_prompt,
                self.tokenizer,
                self.model,
                max_new_tokens=512
            )
        else:
            response = query_llm(
                filter_prompt, 
                self.llm_tokenizer,
                self.llm_client,
                self.adaptive_llm if self.adaptive_llm is not None else self.llm,
                max_new_tokens=512,
                base_timeout=60
            )
        filter_facts = parse_filter(response, candidate_facts)
        filter_scores = [fact_score_map.get(tuple(fact), 0) for fact in filter_facts]
        
        return candidate_facts, filter_facts, filter_scores
    
    def _get_top_k_weights(self, entity_weights):
        # Get indices of all phrases with non-zero weights
        non_zero_indices = np.where(entity_weights > 0)[0]

        if len(non_zero_indices) <= self.top_k_link:
            return entity_weights

        # Get the scores of these phrases
        non_zero_scores = entity_weights[non_zero_indices]

        # Get indices of top-k scores within the non_zero_scores array
        indices_of_top_k_in_non_zero = np.argpartition(non_zero_scores, -self.top_k_link)[-self.top_k_link:]

        # Map these back to the original entity_weights indices
        top_k_original_indices = non_zero_indices[indices_of_top_k_in_non_zero]

        # Create a new entity_weights array, keeping only top-k scores
        new_entity_weights = np.zeros_like(entity_weights)
        new_entity_weights[top_k_original_indices] = entity_weights[top_k_original_indices]

        return new_entity_weights
    
    def graph_score(
        self,
        filter_triples,
        filter_scores, 
        doc_scores,
        passage_node_weight=0.05, 
    ):
        num_vertices = len(self.graph.vs)
        entity_weights = np.zeros(num_vertices)
        number_of_occurs = np.zeros(num_vertices)

        for i, triple in enumerate(filter_triples):
            score = filter_scores[i]
            h, _, t = triple
            
            for phrase in [h, t]:
                phrase = phrase.lower()
                phrase_id = self.entity2vertex_id[phrase]
                    
                weighted_score = score
                weighted_score /= len(self.n2d[phrase_id])
            
                entity_weights[phrase_id] += weighted_score
                number_of_occurs[phrase_id] += 1
        
        # Avoid division by zero
        non_zero_occurs = number_of_occurs > 0
        entity_weights[non_zero_occurs] /= number_of_occurs[non_zero_occurs]

        # Keep only top-k phrase weights
        entity_weights = self._get_top_k_weights(entity_weights)
        
        # Get doc scores from dense retrieval
        doc_scores = min_max_normalize(doc_scores)
        
        passage_weights = np.zeros(num_vertices)
        num_chunks = len(self.corpus)
        num_entities = num_vertices - num_chunks
        passage_weights[num_entities:] = doc_scores * passage_node_weight

        vertex_weights = entity_weights + passage_weights

        # Run Personalized PageRank
        pagerank_scores = self.graph.personalized_pagerank(
            damping=self.damping,
            directed=False,
            weights='weight',
            reset=vertex_weights,
            implementation='prpack'
        )

        # pagerank_scores is for all vertices, extract scores for chunk vertices
        doc_scores = np.array(pagerank_scores[num_entities:])
        top_k_indices = np.argsort(doc_scores)[-self.top_k:][::-1]

        return top_k_indices, doc_scores
    
    def __call__(self, query, excluded_ids=None):
        candidate_triples, filter_triples, filter_scores = self.triple_linking(query)
        # Store latest facts for logging
        self.latest_candidate_facts = candidate_triples
        self.latest_filter_facts = filter_triples

        query_emb = emb_query(
            self.text_encoder,
            self.emb_client, 
            query
        )
        scores = np.dot(self.corpus_emb, query_emb)
        
        if len(filter_triples) == 0:
            top_k_indices = np.argsort(scores)[-self.top_k:][::-1]
        else:
            top_k_indices, scores = self.graph_score(
                filter_triples, filter_scores, doc_scores=scores)
        
        return self.retrieve(top_k_indices, scores, excluded_ids=excluded_ids)
