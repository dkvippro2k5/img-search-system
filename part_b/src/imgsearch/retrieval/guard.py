"""Chặn truy vấn tìm ảnh nhạy cảm của người thật (SCENARIOS X2): từ chối, không trả ảnh, ghi log.

Chỉ là lớp phụ ở mức truy vấn: từ khóa luôn lách được bằng cách diễn đạt khác. Lọc ảnh nhạy cảm
thật sự cần điểm của Phần A (Q4); dữ liệu hiện tại chưa có điểm nào. Từ vựng ở lexicon.SENSITIVE,
so dấu như parser ("khoa than" khớp "khỏa thân", "khoa thận" thì không). Gỡ hai kiểu lách
thường gặp trước khi so: chữ cách nhau bằng dấu ("k.h.ỏ.a t.h.â.n", "s e x") và số thay chữ
("s3x", "nud3").
"""

import re
from dataclasses import dataclass, field

from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.query_parser import tokenize
from imgsearch.text.normalize import nfc

# >= 2 chữ cái đơn lẻ nối bằng dấu ("k.h.ỏ.a"), rồi >= 3 chữ cái đơn lẻ cách nhau bằng khoảng
# trắng ("s e x"). Làm hai bước: gộp chung thì "k.h.ỏ.a t.h.â.n" dính thành một từ.
_DOTTED = re.compile(r"(?<!\w)(?:\w[.\-_*·]+)+\w(?!\w)")
_SPACED = re.compile(r"(?<!\w)(?:\w\s+){2,}\w(?!\w)")
_SEPS = re.compile(r"[.\-_*·\s]+")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a", "$": "s"})
_ADULT = re.compile(r"(?<!\d)18\s*\+")


@dataclass
class GuardResult:
    terms: list[str] = field(default_factory=list)  # từ nhạy cảm tìm thấy (dạng gõ của người dùng)

    @property
    def blocked(self) -> bool:
        return bool(self.terms)


def _variants(query: str) -> list[str]:
    q = nfc(query)
    join = lambda m: _SEPS.sub("", m.group())  # noqa: E731
    joined = _SPACED.sub(join, _DOTTED.sub(join, q))
    out = [q, joined, joined.translate(_LEET)]
    return list(dict.fromkeys(out))


def check(query: str, has_person: bool) -> GuardResult:
    """has_person: truy vấn có nhắc tên người trong dữ liệu (parser đã liên kết)."""
    terms: list[str] = []
    if _ADULT.search(query):
        terms.append("18+")
    for text in _variants(query):
        tokens = tokenize(text)
        for span in lx.SENSITIVE.find(tokens, [False] * len(tokens)):
            if span.value == "strict" or has_person:
                terms.append(" ".join(t.surface for t in tokens[span.start : span.end]))
    return GuardResult(list(dict.fromkeys(terms)))
