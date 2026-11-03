import torch
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class Qwen3EmbeddingModel:
    def __init__(self, model_name='Qwen/Qwen3-Embedding-0.6B'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        # Explicit fp32: newer transformers default to the checkpoint's bf16, and
        # batch_encode's .cpu().numpy() cannot convert a bf16 tensor directly.
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32).to(self.device)
    
    def batch_encode_query_for_doc(self, q_list):
        task_query2doc = 'Given a web search query, retrieve relevant passages that answer the query.'
        text_list = [get_detailed_instruct(task_query2doc, q) for q in q_list]
        return self.batch_encode(text_list)

    @torch.no_grad()
    def batch_encode(self, text_list, batch_size=8, log=False):
        if isinstance(text_list, str):
            text_list = [text_list]
        
        results = []
        if log:
            pbar = tqdm(total=len(text_list), desc="Batch Encoding")
        for i in range(0, len(text_list), batch_size):
            batch_dict = self.tokenizer(
                text_list[i:i + batch_size],
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt"
            )
            batch_dict.to(self.model.device)
            outputs = self.model(**batch_dict)
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            results.append(embeddings)
            if log:
                pbar.update(batch_size)
        
        if log:
            pbar.close()
        
        if len(results) > 1:
            results = torch.cat(results, dim=0)
        else:
            results = results[0]
        results = F.normalize(results, p=2, dim=1)
        
        return results.cpu().numpy()
