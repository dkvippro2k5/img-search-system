import pytest

from imgsearch.retrieval.fusion import rrf


def test_rrf_rewards_items_in_multiple_lists():
    fused = rrf([["a", "b", "c"], ["b", "d"]])
    assert fused[0][0] == "b"


def test_rrf_score_formula():
    fused = dict(rrf([["a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61)


def test_rrf_weights_length_mismatch():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])
