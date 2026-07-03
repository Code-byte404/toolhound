"""backend.py must be the only module importing mlx/mlx_lm (design doc discipline)."""
from pathlib import Path

SRC = Path(__file__).parent.parent / "src" / "toolprobe"


def test_only_backend_imports_mlx():
    for py in SRC.rglob("*.py"):  # rglob: also cover subpackages (methods/*)
        if py.name == "backend.py":
            continue
        text = py.read_text()
        assert "import mlx" not in text and "from mlx" not in text, f"{py.name} imports mlx"
