import json
import os
import time
import torch
import torch.multiprocessing as mp

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import load_corpus, load_qa, filter_cached
from src.emb.infer import emb_corpus
from src.artifacts import ARTIFACT_ROOT_ENV, apply_cache_environment, embedding_dir, embedding_file
from src.prompt import PromptTemplateManager
from src.llm_hf import query_llm_hf
from src.metric import get_recall, get_f1, get_ndcg

from utils import eval

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def resolve_llm_device():
    requested_device = os.getenv("RAGCOMPASS_LLM_DEVICE", "").strip()
    if requested_device:
        return requested_device

    if torch.cuda.is_available():
        return "cuda:0"

    return "cpu"


def load_model(llm, model_id=None):
    if model_id is None:
        if llm == 'Llama-3.1-8B-Instruct':
            model_id = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        else:
            raise ValueError(f"Unsupported llm alias without --model_id override: {llm}")

    llm_device = resolve_llm_device()
    torch_dtype = torch.bfloat16 if llm_device != "cpu" and torch.cuda.is_available() else torch.float32
    print(
        f"Loading LLM model {model_id} on {llm_device} "
        f"(visible CUDA devices: {torch.cuda.device_count()})",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Set pad token to avoid warning - use eos_token as pad_token for Llama models
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if llm_device == "auto":
        model_kwargs["device_map"] = "auto"
    elif llm_device != "cpu":
        model_kwargs["device_map"] = {"": llm_device}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs,
    )

    return tokenizer, model


def ircot(
    args,
    retriever,
    prompt_manager,
    sample_i,
    tokenizer,
    model
):
    retriever.reset()
    it = 0
    thoughts = []
    new_thought = sample_i['question']
    while it < args.max_steps:
        top_k_indices, top_k_docs, top_k_scores = retriever(
            new_thought,
            excluded_ids=sample_i.get('excluded_ids'),
        )
        
        prompt = prompt_manager.render_qa_ircot(
            top_k_docs,
            sample_i['question'],
            thoughts)
        new_thought = query_llm_hf(
            prompt, 
            tokenizer, 
            model,
            args.max_new_tokens,
            args.temperature)
        thoughts.append(new_thought)
        
        if 'Answer:' in new_thought:
            break
        
        it += 1
    
    recall = get_recall(sample_i['gold_doc_list'], top_k_docs)
    ndcg = get_ndcg(sample_i['gold_doc_list'], top_k_docs)

    try:
        pred_ans_raw = new_thought.split('Answer:')[1].strip()
        # Handle multiple answers separated by [ANS]
        if '[ANS]' in pred_ans_raw:
            pred_answers = [ans.strip() for ans in pred_ans_raw.split('[ANS]') if ans.strip()]
            # Compute F1 for each predicted answer and take the maximum
            f1_scores = [get_f1(sample_i['answers'], pred_ans) for pred_ans in pred_answers]
            f1 = float(max(f1_scores)) if f1_scores else 0.0
            pred_ans = pred_answers
        else:
            pred_ans = pred_ans_raw
            f1 = get_f1(sample_i['answers'], pred_ans)
    except:
        pred_ans = new_thought
        f1 = get_f1(sample_i['answers'], pred_ans)
    
    return recall, ndcg, thoughts, pred_ans, f1, top_k_indices, top_k_scores

def infer_worker(
    samples,
    corpus, 
    corpus_emb,
    args,
    out_file
):
    prompt_manager = PromptTemplateManager()
    tokenizer = None
    model = None

    print(
        f"Initializing retriever {args.retriever} for dataset={args.dataset} "
        f"with {len(corpus)} corpus documents and {len(samples)} QA samples.",
        flush=True,
    )

    if args.retriever == 'rag':
        from src.retriever.dense import DenseRetriever

        retriever = DenseRetriever(
            args.text_encoder, 
            args.emb_port,
            corpus_emb,
            args.top_k,
            corpus
        )
    elif args.retriever == 'bm25':
        from src.retriever.bm25 import BM25Retriever

        retriever = BM25Retriever(
            corpus=corpus,
            top_k=args.top_k,
        )
    elif args.retriever == 'splade_v3':
        from src.retriever.splade_v3 import SPLADEv3Retriever

        retriever = SPLADEv3Retriever(
            corpus=corpus,
            top_k=args.top_k,
        )
    elif args.retriever == 'colbertv2':
        from src.retriever.colbertv2 import ColBERTv2Retriever

        retriever = ColBERTv2Retriever(
            corpus=corpus,
            top_k=args.top_k,
            dataset=args.dataset,
        )
    elif args.retriever.startswith('hipporag2'):
        print("Loading LLM before HippoRAG2 retriever initialization.", flush=True)
        tokenizer, model = load_model(args.llm, args.model_id)

        from src.retriever.hippo import HippoRAG2Retriever

        _, damping = args.retriever.split('-')

        retriever = HippoRAG2Retriever(
            args.llm,
            args.dataset,
            prompt_manager,
            args.text_encoder, 
            args.emb_port,
            corpus_emb,
            args.top_k,
            corpus,
            float(damping),
            model=model,
            tokenizer=tokenizer
        )
    else:
        raise NotImplementedError

    if model is None or tokenizer is None:
        print("Retriever initialization complete. Loading LLM for QA generation.", flush=True)
        tokenizer, model = load_model(args.llm, args.model_id)

    with open(out_file, 'a', encoding='utf-8') as fout:
        for _, sample_i in enumerate(tqdm(samples, desc="Performing QA")):
            start_time = time.time()
            recall, ndcg, thoughts, pred_ans, f1, top_k_indices, top_k_scores = ircot(
                args,
                retriever,
                prompt_manager,
                sample_i,
                tokenizer,
                model)
            latency = time.time() - start_time
            out_item = {
                'id': sample_i['id'],
                'gold_doc_list': sample_i['gold_doc_list'],
                'recall': recall,
                'question': sample_i['question'],
                'answers': list(sample_i['answers']),
                'ndcg': ndcg,
                'top_k_indices': top_k_indices,
                'top_k_scores': [float(score) for score in top_k_scores],
                'raw_response': thoughts,
                'pred_ans': pred_ans,
                'f1': f1,
                'latency': latency,
                'candidate_facts': getattr(retriever, 'latest_candidate_facts', None),
                'filter_facts': getattr(retriever, 'latest_filter_facts', None)
            }
            fout.write(json.dumps(out_item, ensure_ascii=False) + '\n')
            fout.flush()

def infer(args, out_file):
    corpus = load_corpus(args.dataset)
    corpus_emb = None
    if args.retriever in {'rag'} or args.retriever.startswith('hipporag2'):
        emb_file = 'chunk_emb.parquet'
        corpus_emb = emb_corpus(
            args.text_encoder,
            corpus,
            args.emb_port,
            emb_dir=str(embedding_dir(args.text_encoder, args.dataset)),
            emb_file=emb_file,
        )
        
    full_samples = load_qa(args.dataset)
    samples = filter_cached(out_file, full_samples)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    if len(samples) == 0:
        return
    
    if args.n_proc > 1:
        samples_subsets = [samples[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for i in range(args.n_proc):
            p = mp.Process(target=infer_worker, args=(
                samples_subsets[i],
                corpus,
                corpus_emb,
                args,
                out_file,
            ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        infer_worker(
            samples, 
            corpus, 
            corpus_emb, 
            args, 
            out_file)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                        choices=['nq', 'nq_train', 'nq_val', 
                                 'musique', 'musique_train', 'musique_val',
                                 'hotpotqa', 'hotpotqa_train', 'hotpotqa_val',
                                 '2wikimultihopqa', '2wikimultihopqa_train', '2wikimultihopqa_val'])
    parser.add_argument('--text_encoder', type=str, default='Qwen3-Embedding-0.6B')
    parser.add_argument('--retriever', type=str,
                        choices=['rag', 'bm25', 'splade_v3', 'colbertv2',
                                 'hipporag2-0.5', 'hipporag2-0.7', 'hipporag2-0.3'])
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top documents to retrieve')
    parser.add_argument("--llm", type=str, default="Llama-3.1-8B-Instruct")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Optional Hugging Face model id override, e.g. meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--emb_port", type=int,
                        help="Optional port for the embedding server (vLLM). If omitted, embeddings run in-process.")
    parser.add_argument("--artifact_root", type=str, default=None,
                        help="Root directory for embeddings and KG artifacts. Overrides RAGCOMPASS_ARTIFACT_ROOT.")
    parser.add_argument("--results_base_dir", type=str, default=None,
                        help="Base directory for JSONL outputs. Defaults to <llm>_<text_encoder>.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Optional cap on pending samples processed in this invocation.")
    parser.add_argument('--max_steps', type=int)
    parser.add_argument("--n_proc", "-n", type=int, default=1,
                        help="Number of processes to use for inference")
    args = parser.parse_args()

    if args.artifact_root:
        os.environ[ARTIFACT_ROOT_ENV] = args.artifact_root
        apply_cache_environment(os.environ, args.artifact_root, override=True)

    save_root = args.results_base_dir or f'{args.llm}_{args.text_encoder}'
    save_dir = os.path.join(save_root, args.dataset)
    os.makedirs(save_dir, exist_ok=True)
    out_file = os.path.join(save_dir, f'{args.retriever}_{args.max_steps}_step.jsonl')
    
    infer(args, out_file)
    eval(out_file)
