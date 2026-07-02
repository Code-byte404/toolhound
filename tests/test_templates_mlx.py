import pytest

from toolprobe.templates import template_sanity


@pytest.mark.mlx
def test_smoke_model_template_bug_detected():
    # Real finding (2026-07-02): the Qwen2.5-0.5B-Instruct chat template (both
    # this mlx-community copy and upstream Qwen2.5) renders its tool-call
    # format example with doubled braces -- {{"name": ...}} -- teaching the
    # model malformed JSON. Verified against raw jinja2: the doubled braces
    # are in the template source string itself, not a renderer bug.
    # template_sanity must flag this repo as template-buggy.
    assert template_sanity("mlx-community/Qwen2.5-0.5B-Instruct-4bit") is False
