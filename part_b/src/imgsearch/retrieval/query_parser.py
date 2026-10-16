"""Phân tích truy vấn bằng quy tắc + từ điển alias person (PLAN.md 4.3, 4.4).

Đường chính là quy tắc (vài ms, xác định). LLM chỉ là fallback có timeout + cache,
không bao giờ là nguồn sự thật về danh tính hay quan hệ.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from imgsearch.text.normalize import fold

# Từ báo hiệu "cùng xuất hiện" -> giao kết quả thay vì hợp có hạn ngạch (M1 vs M2).
CO_OCCURRENCE_MARKERS = ("cung", "chung", "voi nhau", "chup cung")

# Từ quan hệ -> predicate trong bảng relation. Cần kiểm tra ngữ cảnh để tránh H8
# ("anh trai say hi" là tên chương trình, không phải quan hệ).
RELATION_WORDS = {
    "con trai": "child",
    "con gai": "child",
    "con": "child",
    "vo": "spouse",
    "chong": "spouse",
    "ba xa": "spouse",
    "ong xa": "spouse",
    "bo": "parent",
    "cha": "parent",
    "me": "parent",
    "anh trai": "sibling",
    "chi gai": "sibling",
    "em gai": "sibling",
    "em trai": "sibling",
}


class QueryType(StrEnum):
    SINGLE = "single"
    MULTI_INTENT = "multi_intent"
    MULTI_HOP = "multi_hop"


@dataclass
class Intent:
    text: str
    person_ids: list[str] = field(default_factory=list)


@dataclass
class ParsedQuery:
    raw: str
    folded: str
    type: QueryType
    intents: list[Intent]
    co_occurrence: bool = False
    relation_path: list[str] = field(default_factory=list)  # ví dụ ["child"]
    anchor_person_ids: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)  # phủ định, đếm số... (L7, L8)


def parse(query: str) -> ParsedQuery:
    """TODO: liên kết person qua alias (khớp dài nhất + rapidfuzz), tách ý định,
    phát hiện quan hệ."""
    folded = fold(query)
    co = any(m in folded for m in CO_OCCURRENCE_MARKERS)
    return ParsedQuery(
        raw=query,
        folded=folded,
        type=QueryType.SINGLE,
        intents=[Intent(text=query)],
        co_occurrence=co,
    )
