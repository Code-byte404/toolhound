"""backend.py must be the only module importing the version-drifty backends
(mlx/mlx_lm and the constrained-decoding libs outlines/xgrammar) -- design doc
discipline: their APIs churn, so isolation keeps the rest of the tree stable."""
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "toolprobe"

# substring patterns that may only appear in backend.py
BACKEND_ONLY = ("import mlx", "from mlx",
                "import outlines", "from outlines",
                "import xgrammar", "from xgrammar")


def test_only_backend_imports_backend_libs():
    for py in SRC.rglob("*.py"):  # rglob: also cover subpackages (methods/*)
        if py.name == "backend.py":
            continue
        text = py.read_text()
        for pat in BACKEND_ONLY:
            assert pat not in text, f"{py.name} imports a backend-only lib: {pat!r}"
