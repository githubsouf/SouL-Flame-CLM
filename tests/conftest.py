# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import pytest
import torch

TINY_CFG = {
    "vocab_size": 50,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}

GPT2_124M_CFG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


@pytest.fixture(autouse=True)
def fixed_seed():
    torch.manual_seed(123)
