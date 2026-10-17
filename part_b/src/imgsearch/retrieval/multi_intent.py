"""Hàm phụ cho multi-intent và truy vấn chủ đề. Engine quyết định dùng khi nào:

- co_occurrence=True  -> giao: chỉ ảnh có đủ mọi người; rỗng thì báo và hiện từng nhóm (M2).
- co_occurrence=False -> mỗi ý định một nhóm riêng, nên vế ít ảnh vẫn có mặt (M6);
  có ảnh chung thì thêm nhóm "cùng xuất hiện" (M1).
- truy vấn chủ đề không tên người -> đa dạng hóa, không để một người chiếm đầu danh sách (S3, L4).
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

Ranked = list[tuple[str, float]]


def intersect(sets: Iterable[set[str]]) -> set[str]:
    sets = list(sets)
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


def restrict(ranked: Ranked, allowed: set[str], fallback_order: Sequence[str] = ()) -> Ranked:
    """Giữ ảnh thuộc `allowed` theo thứ tự `ranked`, rồi nối phần còn lại của `allowed`
    theo `fallback_order` (điểm 0) để không mất ảnh nào chỉ vì không khớp từ khóa."""
    head = [(i, s) for i, s in ranked if i in allowed]
    seen = {i for i, _ in head}
    tail = [(i, 0.0) for i in fallback_order if i in allowed and i not in seen]
    return head + tail


def diversify(ranked: Ranked, persons_of: Mapping[str, Sequence[str]], cap: int) -> Ranked:
    """Mỗi người tối đa `cap` ảnh ở phần đầu; phần vượt hạn ngạch dồn xuống cuối, giữ thứ tự.
    Ảnh không gắn person được tính chung một nhóm."""
    if cap <= 0:
        return list(ranked)
    count: Counter = Counter()
    head, tail = [], []
    for item in ranked:
        key = tuple(sorted(persons_of.get(item[0], ()))) or ("",)
        if count[key] < cap:
            head.append(item)
            count[key] += 1
        else:
            tail.append(item)
    return head + tail
