"""Chỉ số đánh giá truy xuất (PLAN.md mục 5.2).

Quy ước: `ranked` là danh sách image_id theo thứ tự trả về; `relevant` là tập ảnh
đúng (nhị phân) hoặc dict image_id -> mức liên quan (0/1/2) cho nDCG.
"""

import math
from collections.abc import Mapping, Sequence


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float], k: int) -> float:
    def dcg(values: Sequence[float]) -> float:
        return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(values))

    actual = dcg([gains.get(d, 0.0) for d in ranked[:k]])
    ideal = dcg(sorted(gains.values(), reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def intent_coverage_at_k(
    ranked: Sequence[str], relevant_by_intent: Mapping[str, set[str]], k: int
) -> float:
    """Tỷ lệ ý định có ít nhất một kết quả đúng trong top-k (multi-intent)."""
    if not relevant_by_intent:
        return 0.0
    top = set(ranked[:k])
    hit = sum(1 for rel in relevant_by_intent.values() if top & rel)
    return hit / len(relevant_by_intent)


def min_intent_share_at_k(
    ranked: Sequence[str], relevant_by_intent: Mapping[str, set[str]], k: int
) -> float:
    """Tỷ lệ nhỏ nhất của một ý định trong top-k (độ cân bằng, SCENARIOS M6)."""
    if not relevant_by_intent or k <= 0:
        return 0.0
    top = ranked[:k]
    return min(sum(1 for d in top if d in rel) / k for rel in relevant_by_intent.values())
