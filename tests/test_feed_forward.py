# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import torch

from src.feed_forward import FeedForward
from tests.conftest import TINY_CFG


def test_shape_is_preserved():
    x = torch.randn(2, 3, TINY_CFG["emb_dim"])
    assert FeedForward(TINY_CFG)(x).shape == x.shape


def test_parameter_count_for_768():
    ff = FeedForward({**TINY_CFG, "emb_dim": 768})
    assert sum(p.numel() for p in ff.parameters()) == 4_722_432   # 768*3072+3072 + 3072*768+768


def test_tokens_are_processed_independently():
    ff = FeedForward(TINY_CFG).eval()
    x = torch.randn(1, 4, TINY_CFG["emb_dim"])
    y = ff(x)
    x2 = x.clone(); x2[0, 3] += 5.0              # change only token 3
    y2 = ff(x2)
    assert torch.allclose(y[0, :3], y2[0, :3])   # tokens 0..2 untouched
