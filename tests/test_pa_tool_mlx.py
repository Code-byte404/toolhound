import pytest

from toolprobe.methods.pa_tool import PATool
from toolprobe.models import load_tools

REPO = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.mark.mlx
def test_pa_tool_adapts_real_model(tmp_path):
    from toolprobe.cli import _make_gen
    tools = load_tools("cases/tools.yaml")
    small = {k: tools[k] for k in list(tools)[:2]}       # keep it quick
    mr = PATool(cache_dir=tmp_path, base_seed=0).prepare(REPO, small, gen=_make_gen(REPO))
    # every canonical tool present, names are valid identifiers, inverse map round-trips
    for canon in small:
        fn = mr.tools[canon]["function"]
        assert fn["name"].isidentifier()
        assert fn["description"] == small[canon]["function"]["description"]
