import numpy as np
import os
import pandas as pd
import time
import torch
import torch.nn.functional as F

from huggingface_hub import upload_file, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.artifacts import artifact_repo_path, atomic_replace_path, kg_embedding_dir, resolve_artifact_root

model_map = {
    "Qwen3-Embedding-0.6B": "Qwen/Qwen3-Embedding-0.6B",
}

_LOCAL_EMBEDDING_CLIENTS = {}


def resolve_local_emb_device():
    requested_device = os.getenv("RAGCOMPASS_EMB_DEVICE", "cpu").strip()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def get_embedding_repo_id():
    """Optional Hugging Face dataset repo holding precomputed embeddings; None means local files only."""
    return os.getenv("RAGCOMPASS_EMBEDDING_REPO_ID", "").strip() or None


def last_token_pool(last_hidden_states, attention_mask):
    left_padding = bool(torch.all(attention_mask[:, -1] == 1))
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
    return last_hidden_states[batch_indices, sequence_lengths]


class LocalEmbeddingClient:
    def __init__(self, text_encoder):
        if text_encoder not in model_map:
            raise ValueError(f"Unsupported text encoder for local embedding: {text_encoder}")

        self.text_encoder = text_encoder
        self.model_id = model_map[text_encoder]
        self.device = resolve_local_emb_device()
        self.max_length = int(os.getenv("RAGCOMPASS_EMB_MAX_LENGTH", "10000"))

        torch_dtype = torch.float32
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch_dtype = torch.bfloat16

        print(
            f"Loading local embedding model {self.model_id} on {self.device}. "
            "Set RAGCOMPASS_EMB_DEVICE to override."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text_list):
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            if hasattr(outputs, "last_hidden_state"):
                last_hidden_states = outputs.last_hidden_state
            else:
                last_hidden_states = outputs[0]
            embeddings = last_token_pool(last_hidden_states, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.float().cpu()


def init_local_emb_client(text_encoder):
    device = resolve_local_emb_device()
    client_key = (text_encoder, device)
    if client_key not in _LOCAL_EMBEDDING_CLIENTS:
        _LOCAL_EMBEDDING_CLIENTS[client_key] = LocalEmbeddingClient(text_encoder)
    return _LOCAL_EMBEDDING_CLIENTS[client_key]


def init_emb_client(port, text_encoder=None):
    if port is None:
        if text_encoder is None:
            raise ValueError("text_encoder is required when emb_port is not provided.")
        return init_local_emb_client(text_encoder)

    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="token-abc123")
    return client


def maybe_upload_file(path):
    repo_id = get_embedding_repo_id()
    if repo_id is None:
        return
    try:
        print(f"Uploading {path} to Hugging Face repo {repo_id}...")
        upload_file(
            path_or_fileobj=path,
            path_in_repo=artifact_repo_path(path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Upload complete.")
    except Exception as e:
        print(
            f"Warning: could not upload {path} to Hugging Face "
            f"({type(e).__name__}: {e}). Continuing with local file only."
        )

def get_emb(
    client,
    text_encoder,
    text_list
):
    tries = 0
    while tries < 5:
        tries += 1
        try:                
            if hasattr(client, "embed"):
                return client.embed(text_list)

            response = client.embeddings.create(
                model=model_map[text_encoder],
                input=text_list,
                timeout=60,
            )
            batch_emb = [response.data[i].embedding 
                         for i in range(len(response.data))]
            
            return torch.tensor(batch_emb)
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        raise ValueError

def emb_corpus(
    text_encoder,
    corpus,
    port,
    emb_dir,
    emb_file,
    batch_size=8,
    upload=True,
    fallback_emb_files=None,
):
    os.makedirs(emb_dir, exist_ok=True)
    emb_file = os.path.join(emb_dir, emb_file)
    candidate_files = [emb_file]
    if fallback_emb_files:
        candidate_files.extend(
            candidate for candidate in fallback_emb_files
            if candidate not in candidate_files
        )

    repo_id = get_embedding_repo_id()
    local_dir = str(resolve_artifact_root())

    for candidate_file in candidate_files:
        if os.path.exists(candidate_file):
            print(f"Loading embeddings from local file {candidate_file}")
            try:
                df = pd.read_parquet(candidate_file)
                corpus_emb = np.stack(df['embedding'].to_list())
                if corpus_emb.shape[0] != len(corpus):
                    raise ValueError(
                        f"embedding rows={corpus_emb.shape[0]} but corpus docs={len(corpus)}"
                    )
                return corpus_emb
            except Exception as e:
                print(
                    f"Warning: could not read local embedding file {candidate_file} "
                    f"({type(e).__name__}: {e}). Trying the next candidate."
                )

    for candidate_file in candidate_files if repo_id is not None else []:
        print(
            f"Embedding file {candidate_file} not found locally. "
            f"Trying Hugging Face repo {repo_id}..."
        )
        try:
            repo_filename = artifact_repo_path(candidate_file)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=repo_filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            print(f"Successfully downloaded {repo_filename} from {repo_id}.")
            try:
                df = pd.read_parquet(downloaded_path)
                corpus_emb = np.stack(df['embedding'].to_list())
                if corpus_emb.shape[0] != len(corpus):
                    raise ValueError(
                        f"embedding rows={corpus_emb.shape[0]} but corpus docs={len(corpus)}"
                    )
                return corpus_emb
            except Exception as e:
                print(
                    f"Warning: downloaded embedding file {downloaded_path} could not be read "
                    f"({type(e).__name__}: {e}). Trying the next candidate."
                )
        except (EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as e:
            print(
                f"Could not download {candidate_file} from Hugging Face Hub "
                f"({type(e).__name__}: {e})."
            )

    print(
        f"No cached embeddings found for {emb_file} "
        f"(checked {len(candidate_files)} path(s)). Generating embeddings locally..."
    )
    client = init_emb_client(port, text_encoder)

    results = []
    for i in tqdm(range(0, len(corpus), batch_size)):
        batch_emb = get_emb(client, text_encoder, corpus[i:i + batch_size])
        results.append(batch_emb)

    if len(results) > 1:
        results = torch.cat(results, dim=0)
    else:
        results = results[0]
    results = F.normalize(results, p=2, dim=1)
    corpus_emb = results.float().cpu().numpy()

    df = pd.DataFrame({
        "embedding": list(corpus_emb)
    })
    tmp_emb_file = atomic_replace_path(emb_file, suffix=".parquet.tmp")
    df.to_parquet(tmp_emb_file, index=False)
    os.replace(tmp_emb_file, emb_file)
    print(f"Saved embeddings to {emb_file}")

    if upload:
        maybe_upload_file(emb_file)

    return corpus_emb

def emb_triples(
    text_encoder,
    triple_list,
    emb_port,
    llm,
    dataset,
    upload=False
):
    return emb_corpus(
        text_encoder,
        [str(triple) for triple in triple_list],
        emb_port,
        emb_dir=str(kg_embedding_dir(llm, dataset, text_encoder)),
        emb_file='triple_emb.parquet',
        upload=upload
    )

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def emb_query(
    text_encoder,
    client,
    q
):
    task_query2doc = 'Given a web search query, retrieve relevant passages that answer the query.'
    q = get_detailed_instruct(task_query2doc, q)
    q_emb = get_emb(client, text_encoder, [q])
    q_emb = F.normalize(q_emb, p=2, dim=1)
    
    return q_emb.numpy()[0]

def emb_query2triple(
    text_encoder,
    client,
    q
):
    task_query2triple = 'Given a web search query, retrieve relevant triplet facts that match this query.'
    q = get_detailed_instruct(task_query2triple, q)
    q_emb = get_emb(client, text_encoder, [q])
    q_emb = F.normalize(q_emb, p=2, dim=1)
    
    return q_emb.numpy()[0]
