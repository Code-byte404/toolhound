import pytest

from toolprobe.templates import template_sanity


@pytest.mark.mlx
def test_smoke_model_template_sane():
    assert template_sanity("mlx-community/Qwen2.5-0.5B-Instruct-4bit") is True
