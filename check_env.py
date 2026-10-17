"""
SouL Flame — environment check
==============================
Run this after setup. If everything prints OK, you can start coding.

    python check_env.py
"""
import sys
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
problems = []


def ok(msg):
    print(f"  [OK]   {msg}")


def bad(msg):
    print(f"  [FAIL] {msg}")
    problems.append(msg)


print("\n=== 1. Python ===")
v = sys.version_info
print(f"  version : {platform.python_version()}")
print(f"  exe     : {sys.executable}")
if v >= (3, 9):
    ok("python 3.9+")
else:
    bad("python too old, install 3.10 or newer")
if ".venv" in sys.executable or "venv" in sys.executable:
    ok("running inside the virtual env")
else:
    bad("NOT inside the venv -> activate it first (see README_SETUP.md)")

print("\n=== 2. Packages ===")
try:
    import torch
    ok(f"torch {torch.__version__}")
except ImportError:
    bad("torch missing -> pip install -r requirements.txt")
    torch = None

try:
    import tiktoken
    ok(f"tiktoken {tiktoken.__version__}")
except (ImportError, AttributeError) as e:
    if isinstance(e, AttributeError):
        ok("tiktoken installed")
    else:
        bad("tiktoken missing -> pip install -r requirements.txt")

try:
    import numpy
    ok(f"numpy {numpy.__version__}")
except ImportError:
    bad("numpy missing")

try:
    import pytest
    ok(f"pytest {pytest.__version__}")
except ImportError:
    bad("pytest missing")

print("\n=== 3. Hardware ===")
if torch is not None:
    if torch.cuda.is_available():
        ok(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  [INFO] no CUDA GPU -> CPU only.")
        print("         fine for SOULFLAME_TINY. the 124M config will be slow.")

print("\n=== 4. Autograd smoke test ===")
if torch is not None:
    w = torch.tensor([2.0], requires_grad=True)
    x = torch.tensor([3.0])
    loss = ((w * x - 10.0) ** 2).sum()   # d/dw = 2*(w*x-10)*x = 2*(6-10)*3 = -24
    loss.backward()
    got = w.grad.item()
    if abs(got - (-24.0)) < 1e-6:
        ok(f"gradient = {got} (expected -24.0) -> autograd works")
    else:
        bad(f"gradient = {got}, expected -24.0")

print("\n=== 5. Project files ===")
for rel in ["src", "tests", "configs/config.py", "pytest.ini",
            "data/raw/the-verdict.txt", "data/processed/train.txt",
            "data/processed/val.txt", "data/tiny/overfit.txt"]:
    if (ROOT / rel).exists():
        ok(rel)
    else:
        bad(f"missing: {rel}")

print("\n=== 6. Imports from src/ ===")
sys.path.insert(0, str(ROOT))
try:
    from src.layer_norm import LayerNorm          # noqa: F401
    from src.model import SoulFlameModel          # noqa: F401
    ok("src is importable (pytest.ini handles the path too)")
except Exception as e:
    bad(f"cannot import src: {e}")

print("\n" + "=" * 46)
if problems:
    print(f"{len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)
print("environment ready.")
print("next: open src/layer_norm.py, write it, then run:")
print("    pytest tests/test_layer_norm.py -v")
sys.exit(0)
