# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import pytest
import torch
import torch.nn.functional as F

from src.attention import MultiHeadAttention


def make_mha(d_in=8, d_out=8, T=6, heads=2, dropout=0.0):
    return MultiHeadAttention(d_in, d_out, context_length=T, dropout=dropout,
                              num_heads=heads, qkv_bias=False)


def test_output_shape():
    x = torch.randn(2, 6, 8)
    assert make_mha(d_out=12, heads=3)(x).shape == (2, 6, 12)


def test_shorter_sequence_than_context_length():
    x = torch.randn(2, 3, 8)                     # T=3 < context_length=6
    assert make_mha()(x).shape == (2, 3, 8)


def test_d_out_must_be_divisible_by_heads():
    with pytest.raises(AssertionError):
        make_mha(d_out=10, heads=3)


def test_causality_future_tokens_cannot_leak():
    mha = make_mha().eval()
    x = torch.randn(1, 6, 8)
    y = mha(x)
    x_future_changed = x.clone()
    x_future_changed[0, 4:] = torch.randn(2, 8) * 100
    y2 = mha(x_future_changed)
    assert torch.allclose(y[0, :4], y2[0, :4], atol=1e-6)   # positions 0..3 identical
    assert not torch.allclose(y[0, 4:], y2[0, 4:])          # 4..5 did change


def test_matches_pytorch_reference_with_same_weights():
    B, T, d, H = 2, 6, 8, 2
    mha = make_mha(d_in=d, d_out=d, T=T, heads=H).eval()
    x = torch.randn(B, T, d)

    def heads(t):                               # (B, T, d) -> (B, H, T, hd)
        return t.view(B, T, H, d // H).transpose(1, 2)

    q, k, v = heads(mha.W_query(x)), heads(mha.W_key(x)), heads(mha.W_value(x))
    ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ref = mha.out_proj(ctx.transpose(1, 2).reshape(B, T, d))
    assert torch.allclose(mha(x), ref, atol=1e-5)


def test_mask_is_buffer_not_parameter():
    mha = make_mha()
    assert "mask" in dict(mha.named_buffers())
    assert all("mask" not in n for n, _ in mha.named_parameters())


def test_gradients_reach_all_projections():
    mha = make_mha()
    mha(torch.randn(2, 6, 8)).sum().backward()
    for layer in (mha.W_query, mha.W_key, mha.W_value, mha.out_proj):
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0
