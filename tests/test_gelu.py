# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import torch
import torch.nn.functional as F

from src.gelu import GELU


def test_matches_tanh_reference():
    x = torch.linspace(-6, 6, 201)
    assert torch.allclose(GELU()(x), F.gelu(x, approximate="tanh"), atol=1e-6)


def test_known_values_and_shape():
    g = GELU()
    assert g(torch.tensor(0.0)).item() == 0.0
    assert abs(g(torch.tensor(10.0)).item() - 10.0) < 1e-4
    assert abs(g(torch.tensor(-10.0)).item()) < 1e-4
    assert g(torch.randn(2, 3, 5)).shape == (2, 3, 5)


def test_is_differentiable():
    x = torch.randn(10, requires_grad=True)
    GELU()(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
