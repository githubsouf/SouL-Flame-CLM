
import torch
import torch.nn as nn

from src.layer_norm import LayerNorm
from src.attention import MultiHeadAttention
from src.feed_forward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        # 1. d = cfg["emb_dim"]
        # 2. self.att = MultiHeadAttention(d, d, cfg["context_length"], cfg["drop_rate"],
        #                                  cfg["n_heads"], cfg["qkv_bias"])
        #    d_in == d_out == d here, thats what makes the block stackable
        # 3. self.ff = FeedForward(cfg)
        # 4. TWO separate LayerNorms, self.norm1 and self.norm2. NOT the same object twice !
        #    each one has its own gamma/beta to learn
        # 5. self.drop = nn.Dropout(cfg["drop_rate"])
        d = cfg["emb_dim"]
        self.att = MultiHeadAttention(
            d_in=d,
            d_out=d,
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(d)
        self.norm2 = LayerNorm(d)
        self.drop = nn.Dropout(cfg["drop_rate"])

    def attention_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """x + dropout(attention(norm_1(x)))"""
        # 1. keep shortcut = x  (save the input BEFORE touching it)
        # 2. norm1 -> att -> dropout
        # 3. return shortcut + that
        # pre-LN = norm BEFORE attention (GPT-2 style). the shortcut stays clean, never normalized
        # thats what keeps training stable on 12 layers
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop(x)
        return shortcut + x

    def feed_forward_sublayer(self, x: torch.Tensor) -> torch.Tensor:
        """x + dropout(ffn(norm_2(x)))"""
        # exact same pattern but with norm2 and self.ff
        # new shortcut, taken from the x that comes out of the attention sublayer
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        return shortcut + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """attention_sublayer -> feed_forward_sublayer, (B, T, d) -> (B, T, d)"""
        # 2 lines. attention first (tokens talk to each other), then ffn (each token thinks alone)
        # if u forget a "x +" the residual is gone: test_block_residuals_exist catches it
        # and with 12 layers the gradient would die before reaching layer 1
        x = self.attention_sublayer(x)
        x = self.feed_forward_sublayer(x)
        return x
