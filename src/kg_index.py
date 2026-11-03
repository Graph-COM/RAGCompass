import json
import os

from .artifacts import download_artifact, kg_index_dir

def load_triples(llm, dataset):
    index_dir = str(kg_index_dir(llm, dataset))
    
    triple_file = os.path.join(index_dir, "triples.json")

    if not os.path.exists(triple_file):
        download_artifact(triple_file)

    with open(triple_file, 'r', encoding='utf-8') as f:
        unique_triples = json.load(f)
    
    print(f"Loaded {len(unique_triples)} unique triples from {triple_file}")
    
    return unique_triples

def load_entities(llm, dataset):
    index_dir = str(kg_index_dir(llm, dataset))
    
    entity_file = os.path.join(index_dir, "entities.json")

    if not os.path.exists(entity_file):
        download_artifact(entity_file)   
    
    with open(entity_file, 'r', encoding='utf-8') as f:
        unique_entities = json.load(f)
    
    print(f"Loaded {len(unique_entities)} unique entities from {entity_file}")
    
    return unique_entities
