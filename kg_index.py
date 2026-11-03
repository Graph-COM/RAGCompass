import json
import os
import re
import torch
import gc

from tqdm import tqdm

from src.data import iter_corpus
from src.artifacts import (
    ARTIFACT_ROOT_ENV,
    apply_cache_environment,
    upload_artifact,
    atomic_write_json,
    build_kg_metadata,
    kg_index_dir,
    kg_metadata_file,
)
from src.kg_index import load_triples, load_entities
from src.llm import model_map, PROMPT_JSON_TEMPLATE
from src.llm_hf import query_llm_hf
from src.prompt import PromptTemplateManager


KG_CHECKPOINT_FILE = "kg_index_checkpoint.json"


def env_bool(name, default):
    raw_value = os.getenv(name, "").strip().lower()
    if not raw_value:
        return default
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw_value!r}")


def env_float(name, default):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    return float(raw_value)


def env_int(name, default, min_value=1):
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    value = int(raw_value)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def get_kg_index_backend():
    return os.getenv("RAGCOMPASS_KG_INDEX_BACKEND", "vllm").strip().lower()


def normalize_doc_for_kg(doc):
    max_doc_chars = env_int("RAGCOMPASS_KG_INDEX_MAX_DOC_CHARS", 12000, min_value=0)
    if max_doc_chars == 0 or len(doc) <= max_doc_chars:
        return doc
    return doc[:max_doc_chars]



def import_vllm_modules():
    from vllm import SamplingParams, LLM

    try:
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError:
        StructuredOutputsParams = None

    try:
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError:
        GuidedDecodingParams = None

    return SamplingParams, LLM, StructuredOutputsParams, GuidedDecodingParams


def build_sampling_params(json_template, max_tokens):
    SamplingParams, _, StructuredOutputsParams, GuidedDecodingParams = import_vllm_modules()
    schema = PROMPT_JSON_TEMPLATE[json_template]
    kwargs = {
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    if StructuredOutputsParams is not None:
        kwargs["structured_outputs"] = StructuredOutputsParams(json=schema)
    elif GuidedDecodingParams is not None:
        kwargs["guided_decoding"] = GuidedDecodingParams(json=schema)
    else:
        raise ImportError(
            "Could not import StructuredOutputsParams or GuidedDecodingParams "
            "from vllm.sampling_params. Please install a compatible vLLM version."
        )

    return SamplingParams(**kwargs)


def vllm_batch_infer(client, tokenizer, msg_list, json_template, max_tokens):
    prompts = [tokenizer.apply_chat_template(
        conversation=msg, tokenize=False, add_generation_prompt=True) for msg in msg_list]
    sampling_params = build_sampling_params(json_template, max_tokens)
    vllm_output = client.generate(prompts=prompts,
                                  sampling_params=sampling_params)
    
    all_responses = [completion.outputs[0].text for completion in vllm_output]
    all_prompt_tokens = [len(completion.prompt_token_ids) for completion in vllm_output]
    all_completion_tokens = [len(completion.outputs[0].token_ids) for completion in vllm_output]

    metadata = {
        "prompt_tokens": sum(all_prompt_tokens),
        "completion_tokens": sum(all_completion_tokens),
        "num_request": len(msg_list)
    }
    return all_responses, metadata


def hf_batch_infer(model, tokenizer, msg_list, max_tokens):
    all_responses = []
    all_prompt_tokens = 0
    all_completion_tokens = 0

    for msg in msg_list:
        inputs = tokenizer.apply_chat_template(
            msg,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        all_prompt_tokens += int(inputs["input_ids"].shape[-1])
        response = query_llm_hf(msg, tokenizer, model, max_new_tokens=max_tokens, temperature=0)
        completion_ids = tokenizer.encode(response, add_special_tokens=False)
        all_completion_tokens += len(completion_ids)
        all_responses.append(response)

    metadata = {
        "prompt_tokens": all_prompt_tokens,
        "completion_tokens": all_completion_tokens,
        "num_request": len(msg_list),
    }
    return all_responses, metadata

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def post_process_triple(all_responses):
    triple_dict = dict()
    for idx, response in enumerate(all_responses):
        try:
            extract_results = json.loads(response)["triples"]
        except Exception as e:
            extract_results = []
        
        unique_triples = set()
        valid_triples = []
        
        for triple in extract_results:
            if len(triple) != 3:
                continue
            
            triple = text_processing(triple)
            if any(len(item) == 0 for item in triple):
                continue
            
            if tuple(triple) not in unique_triples:
                unique_triples.add(tuple(triple))
                valid_triples.append(triple)
        
        triple_dict[idx] = valid_triples
    
    return triple_dict

def update_unique_triples_and_mapping(triple_to_id, unique_triples, triple_to_docs, doc_idx, triple_list):
    for triple in triple_list:
        triple_key = tuple(triple)
        triple_id = triple_to_id.get(triple_key)
        if triple_id is None:
            triple_id = len(unique_triples)
            triple_to_id[triple_key] = triple_id
            unique_triples.append(list(triple))
            triple_to_docs[triple_id] = []
        triple_to_docs[triple_id].append(doc_idx)


def get_checkpoint_path(args):
    return kg_index_dir(args.llm, args.dataset) / KG_CHECKPOINT_FILE


def build_triple_to_id(unique_triples):
    return {
        tuple(triple): triple_id
        for triple_id, triple in enumerate(unique_triples)
    }


def load_checkpoint(args):
    checkpoint_path = get_checkpoint_path(args)
    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    expected = {
        "llm": args.llm,
        "dataset": args.dataset,
        "text_encoder": args.text_encoder,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"KG checkpoint {checkpoint_path} has {key}={checkpoint.get(key)!r}, "
                f"expected {value!r}. Remove it or use the matching run configuration."
            )

    checkpoint["triple_to_docs"] = {
        int(triple_id): doc_ids
        for triple_id, doc_ids in checkpoint.get("triple_to_docs", {}).items()
    }
    print(
        f"Resuming KG extraction from checkpoint {checkpoint_path}: "
        f"next_doc_idx={checkpoint.get('next_doc_idx', 0)}, "
        f"unique_triples={len(checkpoint.get('unique_triples', []))}",
        flush=True,
    )
    return checkpoint


def save_checkpoint(args, unique_triples, triple_to_docs, next_doc_idx, token_totals):
    checkpoint_path = get_checkpoint_path(args)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm": args.llm,
        "dataset": args.dataset,
        "text_encoder": args.text_encoder,
        "next_doc_idx": next_doc_idx,
        "unique_triples": unique_triples,
        "triple_to_docs": triple_to_docs,
        **token_totals,
    }
    atomic_write_json(checkpoint_path, payload)
    print(
        f"Saved KG checkpoint to {checkpoint_path}: "
        f"next_doc_idx={next_doc_idx}, unique_triples={len(unique_triples)}",
        flush=True,
    )


def remove_checkpoint(args):
    checkpoint_path = get_checkpoint_path(args)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Removed completed KG checkpoint {checkpoint_path}", flush=True)


def extract_relations(args, corpus_iter):
    prompt_manager = PromptTemplateManager()
    backend = get_kg_index_backend()
    enforce_eager = env_bool("RAGCOMPASS_KG_INDEX_VLLM_ENFORCE_EAGER", True)
    gpu_memory_utilization = env_float("RAGCOMPASS_KG_INDEX_GPU_MEMORY_UTILIZATION", 0.6)
    batch_size = env_int("RAGCOMPASS_KG_INDEX_BATCH_SIZE", 32)
    checkpoint_interval_docs = env_int("RAGCOMPASS_KG_INDEX_CHECKPOINT_INTERVAL_DOCS", 256)
    max_doc_chars = env_int("RAGCOMPASS_KG_INDEX_MAX_DOC_CHARS", 12000, min_value=0)

    print(
        "Starting KG extraction with "
        f"backend={backend}, "
        f"enforce_eager={enforce_eager}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, "
        f"tensor_parallel_size={torch.cuda.device_count()}, "
        f"batch_size={batch_size}, "
        f"checkpoint_interval_docs={checkpoint_interval_docs}, "
        f"max_doc_chars={max_doc_chars}",
        flush=True,
    )

    if backend == "vllm":
        # FlashInfer's sampler may JIT-compile kernels and require a system
        # ninja/CUDA toolchain. Greedy KG extraction does not benefit from it.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        print(
            "vLLM FlashInfer sampler enabled="
            f"{os.environ['VLLM_USE_FLASHINFER_SAMPLER'] != '0'}",
            flush=True,
        )
        _, LLM, _, _ = import_vllm_modules()
        client = LLM(
            model=model_map[args.llm],
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
            dtype='auto',
            enable_prefix_caching=True,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=4096,
            quantization=None,
            load_format='auto',
            trust_remote_code=True
        )
        tokenizer = client.get_tokenizer()
        batch_infer = lambda msg_list, json_template, max_tokens: vllm_batch_infer(
            client, tokenizer, msg_list, json_template, max_tokens
        )
    elif backend == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = os.getenv("RAGCOMPASS_LLM_DEVICE", "").strip() or (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        torch_dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_available() else torch.float32
        print(
            f"Loading HF model {model_map[args.llm]} for KG extraction on {device}",
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_map[args.llm], trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        elif device != "cpu":
            model_kwargs["device_map"] = {"": device}
        client = AutoModelForCausalLM.from_pretrained(
            model_map[args.llm],
            **model_kwargs,
        )
        batch_size = 1
        print("HF backend forces RAGCOMPASS_KG_INDEX_BATCH_SIZE=1 for stability.", flush=True)
        batch_infer = lambda msg_list, json_template, max_tokens: hf_batch_infer(
            client, tokenizer, msg_list, max_tokens
        )
    else:
        raise ValueError(
            f"Unsupported RAGCOMPASS_KG_INDEX_BACKEND={backend!r}. Expected 'vllm' or 'hf'."
        )

    checkpoint = load_checkpoint(args)
    if checkpoint is None:
        unique_triples = []
        triple_to_docs = {}
        total_ner_prompt_tokens = 0
        total_ner_completion_tokens = 0
        total_triple_prompt_tokens = 0
        total_triple_completion_tokens = 0
        num_docs = 0
    else:
        unique_triples = checkpoint.get("unique_triples", [])
        triple_to_docs = checkpoint.get("triple_to_docs", {})
        total_ner_prompt_tokens = checkpoint.get("total_ner_prompt_tokens", 0)
        total_ner_completion_tokens = checkpoint.get("total_ner_completion_tokens", 0)
        total_triple_prompt_tokens = checkpoint.get("total_triple_prompt_tokens", 0)
        total_triple_completion_tokens = checkpoint.get("total_triple_completion_tokens", 0)
        num_docs = checkpoint.get("next_doc_idx", 0)

    triple_to_id = build_triple_to_id(unique_triples)
    last_checkpoint_doc = num_docs
    progress = tqdm(desc="Extracting KG triples", initial=num_docs)
    doc_batch = []

    def current_token_totals():
        return {
            "total_ner_prompt_tokens": total_ner_prompt_tokens,
            "total_ner_completion_tokens": total_ner_completion_tokens,
            "total_triple_prompt_tokens": total_triple_prompt_tokens,
            "total_triple_completion_tokens": total_triple_completion_tokens,
        }

    def maybe_save_progress(force=False):
        nonlocal last_checkpoint_doc
        if force or num_docs - last_checkpoint_doc >= checkpoint_interval_docs:
            save_checkpoint(
                args,
                unique_triples,
                triple_to_docs,
                num_docs,
                current_token_totals(),
            )
            last_checkpoint_doc = num_docs

    for doc_idx, doc in enumerate(corpus_iter):
        if doc_idx < num_docs:
            continue

        doc_batch.append(normalize_doc_for_kg(doc))
        if len(doc_batch) < batch_size:
            continue

        start = num_docs
        num_docs += len(doc_batch)

        ner_msg_list = [prompt_manager.render_ner(doc) for doc in doc_batch]
        ner_output, ner_metadata = batch_infer(
            ner_msg_list, json_template='ner', max_tokens=512)

        triple_msg_list = [
            prompt_manager.render_triple(doc=doc, ne_json=named_entities)
            for doc, named_entities in zip(doc_batch, ner_output)
        ]
        triple_output, triple_metadata = batch_infer(
            triple_msg_list, json_template='triples', max_tokens=2048)

        batch_triple_dict = post_process_triple(triple_output)
        for batch_idx, triple_list in batch_triple_dict.items():
            update_unique_triples_and_mapping(
                triple_to_id,
                unique_triples,
                triple_to_docs,
                start + batch_idx,
                triple_list,
            )

        total_ner_prompt_tokens += ner_metadata["prompt_tokens"]
        total_ner_completion_tokens += ner_metadata["completion_tokens"]
        total_triple_prompt_tokens += triple_metadata["prompt_tokens"]
        total_triple_completion_tokens += triple_metadata["completion_tokens"]

        del ner_msg_list
        del ner_output
        del triple_msg_list
        del triple_output
        del batch_triple_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.update(len(doc_batch))
        doc_batch = []
        maybe_save_progress()

    if doc_batch:
        start = num_docs
        num_docs += len(doc_batch)

        ner_msg_list = [prompt_manager.render_ner(doc) for doc in doc_batch]
        ner_output, ner_metadata = batch_infer(
            ner_msg_list, json_template='ner', max_tokens=512)

        triple_msg_list = [
            prompt_manager.render_triple(doc=doc, ne_json=named_entities)
            for doc, named_entities in zip(doc_batch, ner_output)
        ]
        triple_output, triple_metadata = batch_infer(
            triple_msg_list, json_template='triples', max_tokens=2048)

        batch_triple_dict = post_process_triple(triple_output)
        for batch_idx, triple_list in batch_triple_dict.items():
            update_unique_triples_and_mapping(
                triple_to_id,
                unique_triples,
                triple_to_docs,
                start + batch_idx,
                triple_list,
            )

        total_ner_prompt_tokens += ner_metadata["prompt_tokens"]
        total_ner_completion_tokens += ner_metadata["completion_tokens"]
        total_triple_prompt_tokens += triple_metadata["prompt_tokens"]
        total_triple_completion_tokens += triple_metadata["completion_tokens"]

        del ner_msg_list
        del ner_output
        del triple_msg_list
        del triple_output
        del batch_triple_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress.update(len(doc_batch))
        maybe_save_progress(force=True)

    progress.close()

    print(
        "KG extraction token usage: "
        f"ner_prompt_tokens={total_ner_prompt_tokens}, "
        f"ner_completion_tokens={total_ner_completion_tokens}, "
        f"triple_prompt_tokens={total_triple_prompt_tokens}, "
        f"triple_completion_tokens={total_triple_completion_tokens}, "
        f"num_docs={num_docs}",
        flush=True,
    )
    
    # Create output directory if it doesn't exist
    output_dir = kg_index_dir(args.llm, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save unique triples to disk
    unique_triple_file = os.path.join(output_dir, "triples.json")
    atomic_write_json(unique_triple_file, unique_triples)

    upload_artifact(unique_triple_file)
    
    # Save triple ID to doc IDs mapping to disk
    triple_to_docs_file = os.path.join(output_dir, "triple2doc_id.json")
    atomic_write_json(triple_to_docs_file, triple_to_docs)

    upload_artifact(triple_to_docs_file)
    
    print(f"Saved {len(unique_triples)} unique triples to {unique_triple_file}")
    print(f"Saved triple-to-docs mapping to {triple_to_docs_file}")
    remove_checkpoint(args)
    
    return unique_triples

def get_unique_entities(triple_list):
    entity_set = set()
    for triple in triple_list:
        entity_set.add(triple[0])
        entity_set.add(triple[2])
    print('# unique entities: ', len(entity_set))
    return list(entity_set)

def main(args):
    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(os.environ, args.artifact_root, override=True)

    corpus_iter = iter_corpus(args.dataset)
    print(f"Streaming corpus from dataset/{args.dataset}_corpus.json", flush=True)
    
    triple_list = None
    if not args.force_rebuild:
        try:
            triple_list = load_triples(args.llm, args.dataset)
        except Exception as e:
            print(
                f"Could not load existing triples for {args.dataset} "
                f"({type(e).__name__}: {e}). Building them locally.",
                flush=True,
            )
    else:
        print(f"Force rebuild requested for KG triples: {args.dataset}", flush=True)

    if triple_list is None:
        triple_list = extract_relations(args, corpus_iter)
    
    entity_list = None
    if not args.force_rebuild:
        try:
            entity_list = load_entities(args.llm, args.dataset)
        except Exception as e:
            print(
                f"Could not load existing entities for {args.dataset} "
                f"({type(e).__name__}: {e}). Building them from triples.",
                flush=True,
            )
    else:
        print(f"Force rebuild requested for KG entities: {args.dataset}", flush=True)

    if entity_list is None:
        entity_list = get_unique_entities(triple_list)
        output_dir = kg_index_dir(args.llm, args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        unique_entity_file = os.path.join(output_dir, "entities.json")
        atomic_write_json(unique_entity_file, entity_list)

        upload_artifact(unique_entity_file)

    metadata = build_kg_metadata(args.llm, args.dataset, args.text_encoder, iter_corpus(args.dataset))
    metadata["triple_count"] = len(triple_list)
    metadata["entity_count"] = len(entity_list)
    metadata_path = kg_metadata_file(args.llm, args.dataset)
    atomic_write_json(metadata_path, metadata)
    print(f"Saved KG metadata to {metadata_path}", flush=True)

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
    parser.add_argument("--artifact_root", type=str, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()
    
    main(args)
