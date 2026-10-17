# SouL Flame — setup

One-time install of Python + PyTorch for this project, in a virtual env
(`.venv/`) that lives inside this folder. Nothing is installed system-wide,
so nothing else on the machine can break.

## 1. Install (once)

Double-click **`setup.bat`**, or from a terminal opened in this folder:

```
setup.bat
```

PowerShell version: `.\setup.ps1`

It creates `.venv/`, installs `torch`, `tiktoken`, `numpy`, `pytest`, then runs
`check_env.py`. Takes a few minutes the first time — torch is ~200 MB.

## 2. Every time you open a new terminal

```
.venv\Scripts\activate
```

You should see `(.venv)` at the start of the prompt. Then:

```
pytest -q
```

30 tests, all red until you implement `src/`. That is the starting point.

## 3. Re-check the environment any time

```
python check_env.py
```

It verifies Python, the packages, the GPU, a real autograd gradient
(`-24.0` on a hand-checkable example), the data files, and that `src/` imports.

## VS Code

`Ctrl+Shift+P` → *Python: Select Interpreter* → pick the one inside `.venv`.
Then the Testing tab picks up the 30 tests automatically.

## Notes

- `.venv/` is in `.gitignore` — never commit it. `requirements.txt` is what
  rebuilds it on another machine.
- The setup installs the **CPU** build of torch. That is enough for
  `SOULFLAME_TINY`. For the 124M config on an NVIDIA GPU, install the CUDA
  build instead: see https://pytorch.org/get-started/locally/
- Python 3.10–3.12 recommended. 3.13 works but some wheels lag behind.
