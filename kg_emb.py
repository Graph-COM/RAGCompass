import csv
import os
import torch

from tqdm import tqdm

from src.data import load_corpus
from src.artifacts import (
    ARTIFACT_ROOT_ENV,
    apply_cache_environment,
    upload_artifact,
    atomic_replace_path,
    atomic_write_json,
    build_kg_metadata,
    embedding_dir,
    kg_index_dir,
    kg_metadata_file,
)
from src.emb.infer import emb_corpus, emb_triples
from src.kg_index import load_triples, load_entities

def knn(
    entity_list,
    emb,
    query_batch_size=1000, 
    key_batch_size=10000, 
    k=100,
    threshold=0.8,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    emb_tensor = torch.as_tensor(emb, dtype=torch.float32, device=device)
    num_entities = emb_tensor.size(0)

    out_path = kg_index_dir(args.llm, args.dataset) / "synonym_edges.tsv"
    num_edges = 0
    tmp_out_path = atomic_replace_path(out_path, suffix=".tsv.tmp")
    with open(tmp_out_path, "w", newline="") as fout:
        writer = csv.writer(fout, delimiter='\t')
        # Iterate over query batches
        for q_start in tqdm(range(0, num_entities, query_batch_size), desc="Building kNN graph"):
            q_end = min(q_start + query_batch_size, num_entities)
            query_batch = emb_tensor[q_start:q_end]  # (Q, D)

            # Maintain running top-k neighbors for the current query batch
            best_scores = torch.full((query_batch.size(0), k), -1.0, device=device)
            best_indices = torch.full((query_batch.size(0), k), -1, dtype=torch.long, device=device)

            # Compare against the keys in manageable chunks
            for k_start in range(0, num_entities, key_batch_size):
                k_end = min(k_start + key_batch_size, num_entities)
                key_batch = emb_tensor[k_start:k_end]  # (K, D)

                # Cosine similarity via dot product (embeddings are L2-normalised)
                sim = torch.matmul(query_batch, key_batch.T)  # (Q, K)

                # Mask self-similarity when the query window overlaps the key window
                if q_start <= k_end and k_start <= q_end:
                    overlap_start = max(q_start, k_start)
                    overlap_end = min(q_end, k_end)
                    rel_q = torch.arange(overlap_start, overlap_end, device=device) - q_start
                    rel_k = torch.arange(overlap_start, overlap_end, device=device) - k_start
                    sim[rel_q, rel_k] = -1.0  # make self-similarity ineligible

                # Top-k within this key chunk
                chunk_scores, chunk_indices = torch.topk(sim, min(k, sim.size(1)), dim=1)
                chunk_indices += k_start  # shift to absolute indices

                # Merge with running best
                merged_scores = torch.cat([best_scores, chunk_scores], dim=1)
                merged_indices = torch.cat([best_indices, chunk_indices], dim=1)
                best_scores, pos = torch.topk(merged_scores, k, dim=1)
                best_indices = torch.gather(merged_indices, 1, pos)

            # Write edges that satisfy the similarity threshold
            for row in range(best_scores.size(0)):
                query_id = q_start + row
                for score, neigh_id in zip(best_scores[row].tolist(), best_indices[row].tolist()):
                    if neigh_id == -1 or score < threshold:
                        continue
                    writer.writerow([entity_list[query_id], entity_list[neigh_id], score])
                    num_edges += 1

    os.replace(tmp_out_path, out_path)
    upload_artifact(str(out_path))

    print(f"Wrote {num_edges} kNN edges to {out_path}")

def main(args):
    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(os.environ, args.artifact_root, override=True)

    corpus = load_corpus(args.dataset)
    
    # Emb corpus.
    emb_corpus(
        args.text_encoder,
        corpus,
        args.emb_port,
        emb_dir=str(embedding_dir(args.text_encoder, args.dataset)),
        emb_file='chunk_emb.parquet',
    )
    
    # Emb triples.
    triple_list = load_triples(args.llm, args.dataset)
    emb_triples(
        args.text_encoder,
        triple_list,
        args.emb_port,
        args.llm,
        args.dataset,
        upload=True
    )
    
    # Emb entities.
    entity_list = load_entities(args.llm, args.dataset)
    entity_emb = emb_corpus(
        args.text_encoder,
        entity_list,
        args.emb_port,
        emb_dir=str(kg_index_dir(args.llm, args.dataset) / args.text_encoder),
        emb_file='entity_emb.parquet',
        upload=True
    )
    
    # Add weighted synonymy edges.
    knn(entity_list, entity_emb)

    metadata = build_kg_metadata(args.llm, args.dataset, args.text_encoder, corpus)
    metadata["triple_count"] = len(triple_list)
    metadata["entity_count"] = len(entity_list)
    metadata["has_synonym_edges"] = True
    atomic_write_json(kg_metadata_file(args.llm, args.dataset), metadata)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        choices=['nq', 'nq_train', 'nq_val',
                                 'musique', 'musique_train', 'musique_val',
                                 'hotpotqa', 'hotpotqa_train', 'hotpotqa_val',
                                 '2wikimultihopqa', '2wikimultihopqa_train', '2wikimultihopqa_val'])
    parser.add_argument("--llm", type=str, required=True,
                        choices=["Llama-3.1-8B-Instruct"])
    parser.add_argument('--text_encoder', type=str, default='Qwen3-Embedding-0.6B')
    parser.add_argument("--emb_port", type=int,
                        help="Optional port for the local embedding API server. If omitted, embeddings run in-process.")
    parser.add_argument("--artifact_root", type=str, default=None)
    args = parser.parse_args()
    
    main(args)
