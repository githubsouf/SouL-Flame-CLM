# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.layer_norm import LayerNorm


def test_shape_is_preserved():
    x = torch.randn(2, 3, 5)
    assert LayerNorm(5)(x).shape == (2, 3, 5)


def test_zero_mean_unit_variance_at_init():
    x = torch.randn(2, 3, 5) * 7 + 3          # far from normalized on purpose
    y = LayerNorm(5)(x)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(2, 3), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), torch.ones(2, 3), atol=1e-3)


def test_gamma_beta_are_trainable_and_initialized():
    ln = LayerNorm(5)
    assert isinstance(ln.gamma, nn.Parameter) and isinstance(ln.beta, nn.Parameter)
    assert torch.equal(ln.gamma.data, torch.ones(5))
    assert torch.equal(ln.beta.data, torch.zeros(5))
    assert len(list(ln.parameters())) == 2


def test_matches_pytorch_reference():
    x = torch.randn(4, 7, 16)
    ln = LayerNorm(16)
    with torch.no_grad():                      # non-trivial gamma/beta
        ln.gamma.copy_(torch.randn(16))
        ln.beta.copy_(torch.randn(16))
    ref = F.layer_norm(x, (16,), ln.gamma, ln.beta, eps=1e-5)
    assert torch.allclose(ln(x), ref, atol=1e-5)


def test_gradients_reach_gamma_and_beta():
    ln = LayerNorm(5)
    x = torch.randn(2, 3, 5, requires_grad=True)
    (ln(x) * torch.randn(2, 3, 5)).sum().backward()
    assert ln.gamma.grad is not None and ln.gamma.grad.abs().sum() > 0
    assert ln.beta.grad is not None and ln.beta.grad.abs().sum() > 0
    assert x.grad is not None


def test_constant_input_does_not_produce_nan():
    y = LayerNorm(5)(torch.full((1, 2, 5), 3.0))   # variance = 0 -> eps saves us
    assert torch.isfinite(y).all()
