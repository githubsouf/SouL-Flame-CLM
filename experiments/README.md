# Experiments

One folder per run: `experiments/YYYY-MM-DD_short-name/`

Each run keeps:
- `config.json` — exact hyperparameters + git commit hash + seed
- `log.csv` — step, train_loss, val_loss, lr
- `notes.md` — hypothesis → what happened → what I learned

Rule: a run you cannot reproduce from its folder does not count.
