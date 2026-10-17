"""
SouL Flame — Token + Positional Embeddings
==========================================
STATUS: skeleton. (Can also live directly inside model.py — your call.)

MATH
    E_tok = TokEmb[token_ids]          lookup = one-hot @ matrix
    E_pos = PosEmb[0, 1, ..., T-1]     learned absolute positions (GPT-2)
    x     = E_tok + E_pos              broadcasting over the batch

SHAPES
    token_ids : (B, T)            integers
    TokEmb    : (vocab_size, d)   nn.Embedding
    PosEmb    : (context_length, d)
    E_tok     : (B, T, d)
    E_pos     : (T, d)  -> broadcasts to (B, T, d)

HINTS
    - torch.arange(T, device=token_ids.device) for the positions.
    - T must be <= context_length, otherwise PosEmb has no row for it.
"""
import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, emb_dim: int):
        super().__init__()
        # 1. self.tok_emb = nn.Embedding(vocab_size, emb_dim)
        #    one row per token in the vocab. its a lookup table, and its TRAINABLE
        # 2. self.pos_emb = nn.Embedding(context_length, emb_dim)
        #    one row per POSITION. GPT-2 learns positions, it doesnt use sin/cos
        pass

    def token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, T, d)"""
        # self.tok_emb(token_ids). thats it
        # remember what u proved in ch2: this lookup == one-hot @ matrix, just way faster
        pass

    def position_ids(self, seq_len: int, device) -> torch.Tensor:
        """0, 1, ..., T-1 -> (T,)"""
        # torch.arange(seq_len, device=device)
        # pass the device !! if not, cpu tensor + gpu tensor = crash later
        pass

    def positional_embeddings(self, seq_len: int, device) -> torch.Tensor:
        """(T,) -> (T, d)"""
        # self.pos_emb(self.position_ids(seq_len, device))
        # shape (T, d), no batch dim, on purpose. all sequences in the batch share the same positions
        pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_embeddings + positional_embeddings -> (B, T, d)"""
        # 1. T = token_ids.shape[1], device = token_ids.device
        # 2. just ADD them. (B, T, d) + (T, d) -> broadcasting does the job
        # why add and not concat ? cheaper, and the model learns to separate the 2 signals itself
        # if T > context_length this crashes: pos_emb has no row for it. thats the context limit
        pass
