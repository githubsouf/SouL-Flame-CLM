"""
SouL Flame BOSSSSSS : Multi-Head Causal Self-Attention
=============================================
STATUS: skeleton. Tests: tests/test_attention.py

MATH (per head h, head_dim = d_out / num_heads)
    Q = X W_Q      K = X W_K      V = X W_V
    scores  = Q K^T / sqrt(head_dim)
    scores  = scores masked with -inf where key position j > query position i
    weights = softmax(scores, dim=-1)
    weights = dropout(weights)
    context = weights V
    concat heads -> out_proj (W_O)

SHAPES (B = batch, T = tokens, H = heads, hd = head_dim)
    x            : (B, T, d_in)
    Q, K, V      : (B, T, d_out) -> view (B, T, H, hd) -> transpose (B, H, T, hd)
    scores       : (B, H, T, T)
    context      : (B, H, T, hd) -> transpose (B, T, H, hd) -> view (B, T, d_out)
    output       : (B, T, d_out)

ATTRIBUTE NAMES (the tests rely on these)
    self.W_query, self.W_key, self.W_value : nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj                          : nn.Linear(d_out, d_out)
    self.num_heads, self.head_dim
    mask -> register_buffer("mask", ...) so it follows .to(device) but is NOT trained

HINTS
    - torch.triu(torch.ones(context_length, context_length), diagonal=1) marks the future.
    - Slice the mask to [:T, :T] because T can be shorter than context_length.
    - masked_fill_ with -torch.inf BEFORE softmax, never after.
    - After transpose, call .contiguous() before .view().
    - assert d_out % num_heads == 0 in __init__.
    - F.scaled_dot_product_attention / nn.MultiheadAttention: NOT allowed here
      (tests use one as a reference only).

QUESTIONS
    - Why divide by sqrt(head_dim)? What happens to softmax without it?
    - Why -inf and not 0 in the mask?
    - How do you PROVE token 2 never sees token 3? (see the causality test)
"""
import torch
import torch.nn as nn


def build_causal_mask(context_length: int) -> torch.Tensor:
    """(context_length, context_length), 1 above the diagonal = future"""
    # 1. torch.ones(context_length, context_length)
    # 2. torch.triu(..., diagonal=1) -> keeps only STRICTLY above the diagonal, rest = 0
    # so mask[i][j] = 1 means "j is in the future of i" = forbidden
    # diagonal=1 not 0 !! a token must be allowed to look at ITSELF
    return torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        # 1. assert d_out % num_heads == 0  <- theres a test for this, dont skip it
        #    we split d_out between heads, so it has to divide cleanly
        # 2. store self.d_out, self.num_heads, self.head_dim = d_out // num_heads
        # 3. self.W_query / self.W_key / self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #    exactly these names, the tests grab them
        # 4. self.out_proj = nn.Linear(d_out, d_out)  <- mixes the heads back together
        # 5. self.dropout = nn.Dropout(dropout)
        # 6. self.register_buffer("mask", build_causal_mask(context_length))
        #    buffer NOT parameter: it moves with .to(device) but it is never trained
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", build_causal_mask(context_length))

    def project_qkv(self, x: torch.Tensor):
        """x (B, T, d_in) -> Q, K, V each (B, T, d_out)"""
        # 3 separate calls on the SAME x. thats why its called SELF attention
        # Q = "what am i looking for", K = "what do i offer", V = "what i actually give"
        # return them as a tuple, u unpack in forward
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)
        return q, k, v

    def split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, d_out) -> (B, H, T, head_dim)"""
        # 1. read B and T from t.shape
        # 2. .view(B, T, self.num_heads, self.head_dim) -> cut the last dim in H slices
        # 3. .transpose(1, 2) -> put heads in front: (B, H, T, head_dim)
        # why transpose ? so the matmul after works per head automatically, no loop over heads
        B, T, _ = t.shape
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def attention_scores(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Q K^T -> (B, H, T, T)"""
        # q @ k.transpose(2, 3)  <- transpose the last 2 dims only, NOT (1,2), heads stay in place
        # result (B, H, T, T): scores[b][h][i][j] = how much token i cares about token j
        # not scaled yet, thats the next method
        return q @ k.transpose(2, 3)

    def apply_causal_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """future positions -> -inf, uses self.mask[:T, :T]"""
        # 1. T = scores.shape[-1]
        # 2. slice the mask [:T, :T] -> the real seq can be shorter than context_length
        # 3. .bool() then masked_fill with -torch.inf where its True
        # -inf and NOT 0 !! because softmax(-inf) = 0 exactly.
        # if u put 0 u get exp(0)=1, so the future would still get weight. wrong
        # and do it BEFORE softmax, after is too late, everything is already normalized
        T = scores.shape[-1]
        mask = self.mask[:T, :T].bool()
        return scores.masked_fill(mask, -torch.inf)

    def attention_weights(self, scores: torch.Tensor) -> torch.Tensor:
        """scale by sqrt(head_dim) -> softmax(dim=-1) -> dropout, (B, H, T, T)"""
        # 1. divide by self.head_dim ** 0.5
        #    why: big dot products -> softmax goes super peaky -> gradients ~ 0 -> no learning
        # 2. torch.softmax(..., dim=-1)  <- dim=-1 = over the keys. each ROW sums to 1
        #    if u put dim=-2 the math is garbage and no test will tell u nicely, be careful
        # 3. self.dropout(...) on the weights
        scores = scores / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return self.dropout(weights)

    def context_vectors(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """weights V -> (B, H, T, head_dim)"""
        # weights @ v. (B,H,T,T) @ (B,H,T,hd) -> (B,H,T,hd)
        # this is the actual "attention": a weighted average of the values
        # token i gets a mix of the V of all the tokens it was allowed to look at
        return weights @ v

    def merge_heads(self, context: torch.Tensor) -> torch.Tensor:
        """(B, H, T, head_dim) -> (B, T, d_out)"""
        # 1. .transpose(1, 2) -> (B, T, H, hd), put T back in front
        # 2. .contiguous() -> after a transpose the memory is scrambled, view() refuses without this
        # 3. .view(B, T, self.d_out) -> glue the heads back into one vector
        # u can use .reshape() to skip step 2, but do it once by hand to see the error at least
        context = context.transpose(1, 2).contiguous()
        return context.view(context.shape[0], context.shape[1], self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """project_qkv -> split_heads -> scores -> mask -> weights -> context -> merge_heads -> out_proj"""
        # follow the order in the docstring, split_heads is called 3 times (q, k, v)
        # last line: self.out_proj(merged). the heads worked separately, out_proj mixes them
        # shape check: in (B, T, d_in) -> out (B, T, d_out)
        q, k, v = self.project_qkv(x)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scores = self.attention_scores(q, k)
        scores = self.apply_causal_mask(scores)
        weights = self.attention_weights(scores)
        context = self.context_vectors(weights, v)
        merged = self.merge_heads(context)
        return self.out_proj(merged)
