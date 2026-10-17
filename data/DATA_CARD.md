# SouL Flame — Data Card

| Folder | Content | Rule |
|---|---|---|
| `raw/` | `the-verdict.txt` — Edith Wharton, 1908, public domain (20,479 chars) | never modified |
| `processed/` | `train.txt` / `val.txt` + `split_stats.json` | regenerable from raw |
| `tiny/` | `overfit.txt` — one repeated sentence | sanity checks only |

## Split
- First 90 % of characters → `train.txt` (18,434 chars), rest → `val.txt` (2,045 chars).
- Cut moved forward to the next whitespace so no word is split in half.
- SHA-256 of the raw file: see `split_stats.json` — if it changes, the split is invalid.
- Validation starts at *"collapsed like a house of cards…"* — the model never trains on it.

## Numbers to check yourself (Chapter 2 exercise)
- GPT-2 tokens (tiktoken `gpt2`) for the full text: ≈ 5,145 (book value) — verify it.
- With `max_length=256, stride=256`: how many train windows? how many val windows?
  (val is small → check you still get at least 1.)

## Next datasets (when the pipeline works)
- More public-domain books from Project Gutenberg → `raw/gutenberg/`
- Always: keep raw untouched, write processed versions + a stats file next to them.
