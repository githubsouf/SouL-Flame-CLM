# SouL Flame — Repository map

```
SoulFlame/
├── configs/config.py          SOULFLAME_124M + SOULFLAME_TINY dicts          (you fill)
├── data/
│   ├── raw/the-verdict.txt    untouched source
│   ├── processed/             train.txt, val.txt, split_stats.json
│   ├── tiny/overfit.txt       sanity-check text
│   └── DATA_CARD.md
├── src/                       skeletons: math + shapes + hints, you write the code
│   ├── tokenizer.py  dataset.py  embeddings.py
│   ├── layer_norm.py → gelu.py → feed_forward.py → attention.py
│   ├── transformer_block.py → model.py → generate.py
│   └── (your older tokenizer_demo.py / tokenizer_BTE.ipynb stay as experiments)
├── tests/                     30 tests, red until you implement → green = correct
├── experiments/               one folder per training run
├── checkpoints/               weights (git-ignored)
├── notebooks/                 exploration
└── docs/                      ARCHITECTURE.md, failures/
```

## Build order = test order
| Step | File | Run |
|---|---|---|
| 1 | `src/layer_norm.py` | `pytest tests/test_layer_norm.py` |
| 2 | `src/gelu.py` | `pytest tests/test_gelu.py` |
| 3 | `src/feed_forward.py` | `pytest tests/test_feed_forward.py` |
| 4 | `src/attention.py` | `pytest tests/test_attention.py` |
| 5 | `src/transformer_block.py` | `pytest tests/test_model.py -k block` |
| 6 | `src/model.py` | `pytest tests/test_model.py -k "not generate"` |
| 7 | `src/generate.py` | `pytest tests/test_model.py -k generate` |

Run from the repo root (`pytest.ini` puts `.` on the path so `from src.x import …` works).

## Shape table (GPT-2 small, B=2, T=4)
| Tensor | Shape |
|---|---|
| token ids | (2, 4) |
| tok_emb + pos_emb | (2, 4, 768) |
| Q / K / V per head | (2, 12, 4, 64) |
| attention scores | (2, 12, 4, 4) |
| block output | (2, 4, 768) |
| FFN hidden | (2, 4, 3072) |
| logits | (2, 4, 50257) |
| parameters | 163,009,536 (124,412,160 with weight tying) |
