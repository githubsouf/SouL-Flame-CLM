"""
SouL Flame — configs
====================
STATUS: skeleton — fill the two dicts yourself (values below are the reference).

SOULFLAME_124M (GPT-2 small, the book reference)
    vocab_size     = 50257
    context_length = 1024     (use 256 while training on a laptop)
    emb_dim        = 768
    n_heads        = 12
    n_layers       = 12
    drop_rate      = 0.1
    qkv_bias       = False

SOULFLAME_TINY (for tests + tiny overfit on CPU, runs in seconds)
    vocab_size     = 50       (or 50257 with real tokens)
    context_length = 16
    emb_dim        = 32
    n_heads        = 4
    n_layers       = 2
    drop_rate      = 0.0
    qkv_bias       = False

QUESTION
    emb_dim must be divisible by n_heads — why?
"""

SOULFLAME_124M: dict = {}   # TODO Soufiane
SOULFLAME_TINY: dict = {}   # TODO Soufiane
