"""
R3AG's dual-encoder + fusion-head architecture (Zhao et al. 2026, Section 4).

Faithful to the paper's central design choice (Appendix B, "Retriever Encoding
Strategy"): a retriever is represented, at both training and inference time,
by a single trainable identity token -- never by the documents it returned for
a given query. Two independent encoder towers (retrieval-quality phi_r,
generation-utility phi_g) embed both queries and retriever tokens into the
same metric space; a query-conditioned multi-head-attention head then fuses
each candidate's own [r_Ri, g_Ri] pair before a cosine-similarity routing
decision. See r3ag_train.py for the two-stage training loop that fits these
modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.emb.encoder import last_token_pool

QWEN3_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def retriever_token_string(retriever_name):
    return f"<RET_{retriever_name}>"


def add_retriever_tokens(tokenizer, model, retriever_names):
    special_tokens = [retriever_token_string(name) for name in retriever_names]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added:
        model.resize_token_embeddings(len(tokenizer))
    return {name: tokenizer.convert_tokens_to_ids(retriever_token_string(name)) for name in retriever_names}


def get_transformer_layers(model):
    """Locate the transformer layer ModuleList on an AutoModel-loaded backbone.
    Verified against the actual loaded Qwen3-Embedding-0.6B checkpoint during
    the smoke-test step (build order step 3) -- both attribute paths are
    common across HF encoder wrappers, so this stays defensive rather than
    hardcoding one."""
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError(f"Could not locate a transformer layer list on {type(model)}")


def freeze_backbone_partial(model, num_trainable_layers=2):
    """R3AG Appendix A.5: freeze everything except the (resized) input
    embedding layer and the last `num_trainable_layers` transformer layers."""
    for param in model.parameters():
        param.requires_grad = False
    model.get_input_embeddings().weight.requires_grad = True
    layers = get_transformer_layers(model)
    for layer in layers[-num_trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


class R3AGCapabilityEncoder(nn.Module):
    """One of R3AG's two capability encoders (phi_r or phi_g)."""

    def __init__(self, retriever_names, model_name=QWEN3_EMBEDDING_MODEL, num_trainable_layers=2):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # Explicit fp32: newer transformers default to the checkpoint's bf16, which
        # cannot hold Stage 1's 2e-6 updates and mismatches the fp32 fusion head.
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.retriever_names = list(retriever_names)
        self.retriever_token_ids = add_retriever_tokens(self.tokenizer, self.backbone, self.retriever_names)
        freeze_backbone_partial(self.backbone, num_trainable_layers)

    def encode_query(self, questions, max_length=64):
        device = next(self.backbone.parameters()).device
        batch = self.tokenizer(
            questions, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        outputs = self.backbone(**batch)
        pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(pooled, p=2, dim=-1)

    def encode_all_retrievers(self):
        """Batched single-token forward pass over every retriever's identity
        token, returning [R, H]. Gradient only reaches a retriever's token
        embedding through this call -- excluding a retriever's index from a
        training batch (mechanism-shift protocol, r3ag_train.py
        --holdout_retriever) leaves its embedding at random initialization."""
        device = next(self.backbone.parameters()).device
        token_ids = torch.tensor(
            [[self.retriever_token_ids[name]] for name in self.retriever_names], device=device
        )
        attention_mask = torch.ones_like(token_ids)
        outputs = self.backbone(input_ids=token_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def encode_retrievers_subset(self, retriever_names):
        """Like encode_all_retrievers but restricted to a subset (used to keep
        a held-out retriever's token out of the Stage-1 contrastive batch)."""
        device = next(self.backbone.parameters()).device
        token_ids = torch.tensor(
            [[self.retriever_token_ids[name]] for name in retriever_names], device=device
        )
        attention_mask = torch.ones_like(token_ids)
        outputs = self.backbone(input_ids=token_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)


class R3AGFusionHead(nn.Module):
    """Section 4.1, 'Capability Fusion' (Eq. 3) + 'Similarity-Based Retriever
    Routing' (Eq. 4). Query-conditioned multi-head attention over each
    candidate's OWN [r_Ri, g_Ri] pair -- distinct from our own comparison head
    (train_head.py's LateInterHeadModel), which self-attends
    across candidates' single pooled vectors so candidates can compare against
    each other; here there is no attention between candidates at all, only
    between the query and one candidate's two capability vectors at a time."""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, r_q, r_candidates, g_candidates):
        """
        r_q: [B, H] query retrieval-intent embedding (query encoded by phi_r).
        r_candidates, g_candidates: [B, R, H] per-candidate quality/utility embeddings.
        Returns h: [B, R, H], the fused per-candidate representation (Eq. 3).
        """
        batch_size, num_candidates, hidden_size = r_candidates.shape
        query = r_q.unsqueeze(1).unsqueeze(1).expand(-1, num_candidates, -1, -1)
        query = query.reshape(batch_size * num_candidates, 1, hidden_size)
        key_value = torch.stack([r_candidates, g_candidates], dim=2)  # [B, R, 2, H]
        key_value = key_value.reshape(batch_size * num_candidates, 2, hidden_size)
        fused, _ = self.attn(query, key_value, key_value)  # [B*R, 1, H]
        return fused.reshape(batch_size, num_candidates, hidden_size)


def routing_scores(g_q, fused):
    """Eq. 4: cosine similarity between the query's generation-utility
    embedding and each candidate's fused representation. g_q: [B, H];
    fused: [B, R, H]. Returns [B, R]."""
    g_q_norm = F.normalize(g_q, p=2, dim=-1).unsqueeze(1)  # [B, 1, H]
    fused_norm = F.normalize(fused, p=2, dim=-1)
    return (g_q_norm * fused_norm).sum(dim=-1)
