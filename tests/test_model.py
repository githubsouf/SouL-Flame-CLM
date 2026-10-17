# Tests designed with AI assistance (allowed by the Current Delivery rule: "AI may design tests").
# Soufiane must be able to explain every assertion before relying on it.
# They FAIL until the matching src/ module is implemented — that is the point (test-first).
import torch
import torch.nn.functional as F

from src.transformer_block import TransformerBlock
from src.model import SoulFlameModel
from src.generate import generate_text_simple
from tests.conftest import TINY_CFG, GPT2_124M_CFG


# ---------- TransformerBlock ----------

def test_block_preserves_shape():
    x = torch.randn(2, 5, TINY_CFG["emb_dim"])
    assert TransformerBlock(TINY_CFG)(x).shape == x.shape


def test_block_is_causal():
    blk = TransformerBlock(TINY_CFG).eval()
    x = torch.randn(1, 6, TINY_CFG["emb_dim"])
    x2 = x.clone(); x2[0, 5] += 10.0
    assert torch.allclose(blk(x)[0, :5], blk(x2)[0, :5], atol=1e-5)


def test_block_residuals_exist():
    # zero every Linear -> attention and FFN output 0 -> only the shortcuts remain -> block(x) == x
    blk = TransformerBlock(TINY_CFG).eval()
    with torch.no_grad():
        for m in blk.modules():
            if isinstance(m, torch.nn.Linear):
                m.weight.zero_()
                if m.bias is not None:
                    m.bias.zero_()
    x = torch.randn(1, 4, TINY_CFG["emb_dim"])
    assert torch.allclose(blk(x), x, atol=1e-6)


# ---------- SoulFlameModel ----------

def test_logits_shape():
    ids = torch.randint(0, TINY_CFG["vocab_size"], (2, 7))
    assert SoulFlameModel(TINY_CFG)(ids).shape == (2, 7, TINY_CFG["vocab_size"])


def test_gpt2_small_parameter_count():
    model = SoulFlameModel(GPT2_124M_CFG)
    total = sum(p.numel() for p in model.parameters())
    assert total == 163_009_536
    assert total - model.out_head.weight.numel() == 124_412_160


def test_model_is_causal():
    model = SoulFlameModel(TINY_CFG).eval()
    ids = torch.randint(0, TINY_CFG["vocab_size"], (1, 8))
    ids2 = ids.clone(); ids2[0, 6:] = (ids2[0, 6:] + 1) % TINY_CFG["vocab_size"]
    assert torch.allclose(model(ids)[0, :6], model(ids2)[0, :6], atol=1e-5)


def test_every_parameter_gets_a_gradient():
    model = SoulFlameModel(TINY_CFG)
    ids = torch.randint(0, TINY_CFG["vocab_size"], (2, 8))
    logits = model(ids[:, :-1])
    loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient for: {missing}"


def test_initial_loss_is_close_to_log_vocab():
    # random init -> almost uniform prediction -> loss ~ ln(V)
    model = SoulFlameModel(TINY_CFG).eval()
    ids = torch.randint(0, TINY_CFG["vocab_size"], (8, 16))
    with torch.no_grad():
        loss = F.cross_entropy(model(ids[:, :-1]).flatten(0, 1), ids[:, 1:].flatten())
    expected = torch.log(torch.tensor(float(TINY_CFG["vocab_size"])))
    assert abs(loss.item() - expected.item()) < 1.0


def test_tiny_overfit_sanity():
    # If the model cannot memorize ONE tiny batch, something is broken.
    model = SoulFlameModel(TINY_CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = torch.randint(0, TINY_CFG["vocab_size"], (2, 16))
    for _ in range(200):
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten())
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.1


# ---------- generation ----------

def test_generate_appends_tokens_and_is_deterministic():
    model = SoulFlameModel(TINY_CFG).eval()
    start = torch.tensor([[1, 2, 3]])
    out1 = generate_text_simple(model, start, max_new_tokens=5, context_size=TINY_CFG["context_length"])
    out2 = generate_text_simple(model, start, max_new_tokens=5, context_size=TINY_CFG["context_length"])
    assert out1.shape == (1, 8)
    assert torch.equal(out1[:, :3], start)
    assert torch.equal(out1, out2)


def test_generate_beyond_context_length_does_not_crash():
    model = SoulFlameModel(TINY_CFG).eval()
    start = torch.randint(0, TINY_CFG["vocab_size"], (1, TINY_CFG["context_length"]))
    out = generate_text_simple(model, start, max_new_tokens=4, context_size=TINY_CFG["context_length"])
    assert out.shape == (1, TINY_CFG["context_length"] + 4)
