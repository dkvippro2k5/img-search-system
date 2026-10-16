import pytest

from imgsearch.eval.metrics import (
    intent_coverage_at_k,
    min_intent_share_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "x", "b", "y"]
REL = {"a", "b", "c"}


def test_precision_recall():
    assert precision_at_k(RANKED, REL, 2) == 0.5
    assert recall_at_k(RANKED, REL, 4) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a"], REL) == 0.5
    assert reciprocal_rank(["x"], REL) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)


def test_intent_metrics():
    by_intent = {"tran_thanh": {"a", "b"}, "hari_won": {"y"}}
    assert intent_coverage_at_k(RANKED, by_intent, 2) == 0.5
    assert intent_coverage_at_k(RANKED, by_intent, 4) == 1.0
    assert min_intent_share_at_k(RANKED, by_intent, 4) == 0.25
