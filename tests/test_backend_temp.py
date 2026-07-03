import pytest

from toolprobe.backend import generate


def test_grammar_still_reserved_with_new_signature():
    with pytest.raises(NotImplementedError):
        generate("any-repo", "hi", grammar="{}", temp=0.4, seed=1)
