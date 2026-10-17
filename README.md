# SoulFlame-CLM

**A decoder-only causal Transformer language model built from first principles in PyTorch.**

SoulFlame-CLM is a learning-driven language-model engineering project focused on understanding, implementing, testing, training, and evaluating a Transformer without relying on a ready-made GPT or Transformer implementation.

The project is intentionally built component by component. Every core mechanism - LayerNorm, GELU, feed-forward networks, causal multi-head self-attention, residual connections, Transformer blocks, embeddings, and the language-model head - is implemented explicitly and tested before the next layer is added.



Repository: [https://github.com/githubsouf/SouL-Flame-CLM](https://github.com/githubsouf/SouL-Flame-CLM)

---


Mathematically:

$$
x_1 = x + \mathrm{Dropout}(\mathrm{Attention}(\mathrm{LayerNorm}_1(x)))
$$

$$
x_2 = x_1 + \mathrm{Dropout}(\mathrm{FFN}(\mathrm{LayerNorm}_2(x_1)))
$$

The block preserves the model shape:

$$
(B, T, d) \rightarrow (B, T, d)
$$

which makes Transformer blocks stackable.

---

## Core Data Flow

```text
text
-> tokenizer
-> token IDs
-> token embeddings + positional embeddings
-> embedding dropout
-> TransformerBlock x N
-> final LayerNorm
-> vocabulary projection
-> logits
-> loss during training / decoding during generation
```

During training, the target path will become:

```text
input token IDs
-> forward pass
-> logits
-> shifted next-token targets
-> cross-entropy loss
-> backward()
-> optimizer.step()
-> updated model parameters
```

---

## Multi-Head Causal Self-Attention

For each head:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

$$
S = \frac{QK^T}{\sqrt{d_{head}}}
$$

Future positions are masked with $-\infty$, then:

$$
A = \mathrm{softmax}(S)
$$

$$
Z = AV
$$

The heads are concatenated and mixed with the output projection $W_O$.

Tensor flow:

```text
x                 (B, T, d_in)
Q, K, V           (B, T, d_out)
split heads       (B, H, T, head_dim)
attention scores  (B, H, T, T)
attention output  (B, H, T, head_dim)
merge heads       (B, T, d_out)
out projection    (B, T, d_out)
```

### What The Attention Tests Verify

The current attention suite contains **7 passing tests** covering output shape, shorter sequences, head-dimension divisibility, causal future-token isolation, numerical agreement with PyTorch's scaled-dot-product attention as a **test oracle only**, the causal mask being a non-trainable buffer, and gradient flow through $W_Q$, $W_K$, $W_V$, and $W_O$.

The key behavioral invariant is:

> **Changing future token representations must not change earlier token outputs.**

That tests real causal behavior rather than merely checking that a triangular mask exists.

---

## LayerNorm

SoulFlame implements LayerNorm directly instead of using `nn.LayerNorm`.

$$
\mu = \mathrm{mean}(x)
$$

$$
\sigma^2 = \mathrm{var}(x)
$$

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
y = \gamma \hat{x} + \beta
$$

Implementation details:

- normalize across the last feature dimension,
- use population-style variance (`unbiased=False`),
- use `eps` for numerical stability,
- initialize `gamma` to ones,
- initialize `beta` to zeros.

**Mental model:** LayerNorm = numerical stability and calibration.

---

## GELU

SoulFlame implements the GPT-style tanh approximation:

$$
\mathrm{GELU}(x) = 0.5x \left(1 + \tanh \left[\sqrt{\frac{2}{\pi}} \left(x + 0.044715x^3\right)\right]\right)
$$

GELU has no trainable parameters.

**Mental model:** GELU = nonlinear feature gating.

---

## Feed-Forward Network

Each token is processed independently using:

$$
d \rightarrow 4d \rightarrow \mathrm{GELU} \rightarrow d
$$

For a GPT-2-small-style configuration:

$$
768 \rightarrow 3072 \rightarrow 768
$$

The first projection creates a larger learned nonlinear workspace; the second maps it back to the model dimension so residual connections and subsequent Transformer blocks remain shape-compatible.

> **Attention = tokens communicate.**
>
> **FFN = each token processes its own features.**

---

## Residual Connections

Each Transformer sublayer follows:

$$
y = x + F(x)
$$

The residual stream preserves the existing representation while the branch learns a correction and provides a direct gradient path through deep stacks.

> **Residual = old representation + learned update / gradient highway.**

---

## Embeddings

The token embedding layer owns a trainable matrix:

$$
E_{token} \in \mathbb{R}^{V \times d}
$$

Token IDs are integer addresses. Calling the embedding layer performs a row lookup.

```text
token IDs: [125, 900, 42]

lookup:
E[125]
E[900]
E[42]
```

The IDs themselves are not learned. The embedding-table values are.

SoulFlame currently uses learned absolute positional embeddings:

$$
X_i = E_{token}[t_i] + E_{position}[i]
$$

---

## Full Model

```text
token_ids                         (B, T)
-> token + position embeddings    (B, T, d)
-> dropout
-> TransformerBlock x n_layers    (B, T, d)
-> final LayerNorm                (B, T, d)
-> Linear(d, vocab_size)          (B, T, vocab_size)
-> logits
```

The model deliberately returns **raw logits**. Softmax is not applied inside the model because training will use cross-entropy directly on the logits.

The current implementation uses a separate output-head matrix rather than weight tying.

For the GPT-2-small-style reference configuration:

- vocabulary size: `50,257`
- embedding dimension: `768`
- attention heads: `12`
- Transformer blocks: `12`
- context length: `1,024`
- FFN hidden width: `3,072`

With an untied output head, the current implementation contains approximately **163,009,536 parameters**.

With token-embedding/output weight tying, the unique parameter count would be approximately **124,412,160**.

Development and tests use much smaller configurations so experiments remain practical.

---

## Testing Philosophy

SoulFlame is not considered correct simply because a forward pass runs. The project tests **invariants**.

Examples:

- future tokens cannot influence past outputs,
- tensor shapes remain valid,
- attention heads divide the output dimension correctly,
- gradients reach trainable projections,
- the causal mask is not trainable,
- the full model returns finite logits with the expected shape,
- gradients can propagate from outputs into early model components.

Current verified results:

```text
tests/test_attention.py    7 passed
tests/test_feed_forward.py 3 passed
tests/test_gelu.py         3 passed
tests/test_layer_norm.py   6 passed
tests/test_model.py       11 passed
total                     30 passed
```

Longer-term tests will include:

- tokenizer round-trip,
- checkpoint reproducibility,
- deterministic decoding beyond greedy,
- temperature/sampling behavior,
- malformed-input handling,
- latency and memory benchmarks,
- model-regression tests.

---

## AI Assistance Policy

AI may be used to explain concepts, review code, propose pseudocode, identify bugs, inspect tensor shapes, design tests, and challenge assumptions.

AI is not used to silently replace the implementation-learning process.

Tests may be designed with AI assistance, but every assertion must be understood before it is trusted.

> **If I cannot explain a line of code, it does not belong in SoulFlame.**

---

## Training Plan

The model is currently **untrained**. Its trainable weights begin from random initialization.

Training will jointly update:

- token embeddings,
- positional embeddings,
- $W_Q$, $W_K$, $W_V$, $W_O$,
- FFN projections,
- LayerNorm $\gamma$, $\beta$,
- vocabulary output projection.

Planned training pipeline:

```text
private corpus
-> tokenize
-> chunk into independent sequences
-> shifted next-token targets
-> forward pass
-> cross-entropy loss
-> backward()
-> optimizer step
-> validation
-> checkpoint
```

A mandatory first milestone will be a **tiny-overfit test**: prove the implementation can intentionally overfit a very small dataset before attempting larger runs.

---

## Dataset Direction And Privacy

The long-term plan is to train SoulFlame on a **private autobiographical corpus** derived from the author's own writings and experiences.

The raw personal corpus will not be published.

**Public:** architecture, training pipeline, tests, evaluation code, configuration, sanitized examples, benchmark methodology, and technical documentation.

**Private:** raw autobiographical data, sensitive logs, personally identifying source material, and initially trained checkpoints that may memorize private text.

Evaluation should use a **held-out personal dataset** separate from training examples.

This creates an interesting evaluation environment for studying:

- memorization vs. generalization,
- personal-fact recall,
- hallucinated personal facts,
- checkpoint regressions,
- consistency and reliability.

---

## Roadmap

### Phase 1 - Finish The Language Model

- LayerNorm
- GELU
- FeedForward
- Multi-Head Causal Self-Attention
- TransformerBlock
- SoulFlameModel
- Attention tests
- Model tests
- Greedy autoregressive generation
- Temperature / sampling / top-k
- Training data pipeline
- Cross-entropy training loop
- Validation loop
- Tiny-overfit proof
- Checkpoint save/load
- Reproducible training run

### Phase 2 - SoulFlame Eval

Build an automated evaluation runner covering:

- training/validation loss,
- perplexity,
- curated QA cases,
- factual correctness,
- hallucination analysis,
- structured-output correctness,
- prompt-variation robustness,
- deterministic behavior,
- checkpoint/model comparisons.

### Phase 3 - Reliability And Regression Engineering

```text
new checkpoint / model version
-> automated evaluation suite
-> quality metrics
-> regression checks
-> reliability checks
-> performance benchmarks
-> PASS / FAIL release gate
```

Example future report:

```text
Checkpoint v0.8

Perplexity            improved
Causal leakage        pass
Personal-fact recall  improved
Hallucination rate    regressed
Structured output     pass
Latency p95           pass
Regression suite      4 failures

Release decision: FAIL
```

### Phase 4 - Modernize Experimentally

Only after the baseline trains correctly, introduce modern components one at a time and measure them against the baseline:

- learned positions -> RoPE,
- LayerNorm -> RMSNorm,
- GELU FFN -> SwiGLU,
- standard MHA -> GQA,
- KV cache,
- mixed precision,
- optimized attention,
- larger model/data configurations.

The goal is to **measure the effect of each change**, not merely accumulate fashionable components.

---

## Project Philosophy

> **Complexity is earned.**

First:

```text
understand
-> implement
-> test
-> train
-> measure
```

Then:

```text
change architecture
-> benchmark against baseline
-> keep the change only if evidence supports it
```

The final engineering story is:

> **Understand an LLM -> build it -> train it -> test it rigorously -> evaluate it -> detect regressions -> improve it with evidence.**

---

## Why SoulFlame-CLM?

The goal is not to reproduce a production-scale frontier model or simply wrap an existing pretrained model.

The goal is to build a technically honest foundation for:

1. understanding how decoder-only Transformer language models work internally,
2. training a language model from random initialization,
3. testing mathematical and architectural invariants,
4. studying memorization, generalization, hallucination, and regression behavior,
5. building a production-style AI evaluation and reliability system around the model.

SoulFlame-CLM is therefore both a **language-model implementation project** and the future foundation for **SoulFlame Eval**, an evaluation/reliability platform.

---

## Author

Built as a hands-on language-model engineering and AI reliability project by **Soufiane Abdelouahab**.