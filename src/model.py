"""
the full model



  token_ids (B, T)
    -> tok_emb + pos_emb          (B, T, d)
    -> dropout
    -> TransformerBlock x n_layers (B, T, d)
    -> final LayerNorm            (B, T, d)
    -> out_head: Linear(d, vocab_size, bias=False)
    -> logits                     (B, T, vocab_size)

ATTRIBUTE NAMES
    self.tok_emb, self.pos_emb, self.drop_emb, self.trf_blocks,
    self.final_norm, self.out_head

CHECKPOINT NUMBERS 
    total parameters = 163,009,536
    (124,412,160 if you subtract out_head, i.e. with weight tying)
"""
import torch
import torch.nn as nn

from src.layer_norm import LayerNorm
from src.transformer_block import TransformerBlock


class SoulFlameModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # names below are the ones the tests use, keep them exactly
        # 1. self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 2. self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 3. self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 4. self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        #    the * unpacks the list. a python list alone would NOT register the params, careful
        # 5. self.final_norm = LayerNorm(cfg["emb_dim"])
        # 6. self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        #    bias=False, GPT-2 does it like that. its the biggest layer of the whole model
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def embed(self, in_idx: torch.Tensor) -> torch.Tensor:
        """tok_emb + pos_emb -> drop_emb, (B, T) -> (B, T, d)"""
        # 1. T = in_idx.shape[1]
        # 2. tok = self.tok_emb(in_idx)
        # 3. pos = self.pos_emb(torch.arange(T, device=in_idx.device))
        # 4. self.drop_emb(tok + pos)
        T = in_idx.shape[1]
        tok = self.tok_emb(in_idx)
        pos = self.pos_emb(torch.arange(T, device=in_idx.device))
        return self.drop_emb(tok + pos)

    def run_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """trf_blocks x n_layers, (B, T, d) -> (B, T, d)"""
        # self.trf_blocks(x). Sequential chains them for u, no loop to write
        # shape never changes across the 12 blocks. thats the whole point of the design
        return self.trf_blocks(x)

    def to_logits(self, x: torch.Tensor) -> torch.Tensor:
        """final_norm -> out_head, (B, T, d) -> (B, T, vocab_size)"""
        # 1. final_norm (one last clean up before the big projection)
        # 2. out_head -> one score per word of the vocab, for EVERY position
        # NO softmax here. raw logits. cross_entropy does the softmax itself
        # if u softmax twice the loss is wrong and it trains like trash
        x = self.final_norm(x)
        return self.out_head(x)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """embed -> run_blocks -> to_logits"""
        # 3 lines, thats ur whole LLM
        # in: (B, T) ints. out: (B, T, vocab_size) floats
        x = self.embed(in_idx)
        x = self.run_blocks(x)
        return self.to_logits(x)

    def count_parameters(self) -> int:
        """total trainable parameters (GPT-2 small: 163,009,536)"""
        # sum(p.numel() for p in self.parameters())
        # add "if p.requires_grad" if u want to be strict about it
        return sum(p.numel() for p in self.parameters())

    def model_size_mb(self) -> float:
        """parameters * 4 bytes (float32) / 1024^2"""
        # count_parameters() * 4 / (1024 ** 2)
        # ~621 MB for the weights alone. add gradients + adam states -> x4 in training
        # thats why u train the TINY config on ur laptop, not the 124M
        return self.count_parameters() * 4 / (1024 ** 2)
