"""Gộp nhiều danh sách xếp hạng bằng Reciprocal Rank Fusion."""

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

RRF_K = 60


def rrf(
    rankings: Iterable[Sequence[Hashable]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Điểm = Σ w_i / (k + hạng_i), hạng bắt đầu từ 1. Trả về giảm dần theo điểm."""
    rankings = list(rankings)
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights phải cùng độ dài với rankings")

    scores: dict[Hashable, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
