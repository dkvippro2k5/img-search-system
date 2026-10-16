"""Chuẩn hóa văn bản tiếng Việt.

Luôn giữ song song hai dạng: có dấu (ưu tiên khi khớp) và không dấu (tăng recall
cho truy vấn gõ không dấu). Xem SCENARIOS.md L1/L2.
"""

import re
import unicodedata

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize(text: str) -> str:
    """Chữ thường, NFC, bỏ dấu câu, gộp khoảng trắng. Giữ dấu tiếng Việt."""
    text = nfc(text).lower()
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt, kể cả đ/Đ (không phải dấu tổ hợp nên phải thay riêng)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def fold(text: str) -> str:
    """Dạng khóa dùng để so khớp không dấu."""
    return strip_diacritics(normalize(text))
