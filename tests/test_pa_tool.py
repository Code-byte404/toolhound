from toolprobe.methods.pa_tool import (edit_distance, peakedness_scores,
                                       select_name, _extract_name)


def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_extract_name():
    assert _extract_name("get_weather") == "get_weather"
    assert _extract_name("  The name is: FetchForecast. ") == "the"  # first identifier
    assert _extract_name("123 456") is None


def test_peakedness_prefers_the_cluster():
    # four near-duplicates cluster; one outlier is isolated
    cands = ["get_weather", "get_wether", "get_weathr", "get_weater", "zzzzzzzz"]
    scores = peakedness_scores(cands, alpha=0.2)
    assert scores[-1] == 0                       # outlier: no neighbours within tau
    assert max(scores[:4]) >= 2                   # cluster members have neighbours
    assert select_name(cands, reference="get_weather", alpha=0.2).startswith("get_weath")


def test_select_name_tiebreak_by_reference():
    # two singletons tie at peakedness 0; reference breaks the tie
    cands = ["alpha", "omega"]
    assert select_name(cands, reference="alphb", alpha=0.2) == "alpha"
