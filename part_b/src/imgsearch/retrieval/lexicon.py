"""Từ vựng cho bộ phân tích truy vấn. Viết có dấu; so khớp không dấu được suy ra tự động.

Khớp theo từng token: token người dùng gõ có dấu phải khớp đúng dấu, token không dấu
khớp với dạng bỏ dấu (tránh "chùa" bị hiểu là "chưa" khi người dùng đã gõ dấu).
"""

from dataclasses import dataclass

from imgsearch.text.normalize import fold, normalize


@dataclass(frozen=True)
class Token:
    surface: str  # nguyên văn (giữ hoa/thường)
    norm: str  # chữ thường, có dấu
    fold: str  # chữ thường, không dấu

    @property
    def has_marks(self) -> bool:
        return self.norm != self.fold

    @property
    def is_word(self) -> bool:
        return self.norm not in STRONG_SEPS and self.norm not in DASHES


@dataclass(frozen=True)
class Span:
    start: int  # chỉ số token, nửa mở [start, end)
    end: int
    value: str


class Lexicon:
    def __init__(self, entries: dict[str, str]):
        self._entries = sorted(
            ((tuple(normalize(k).split()), tuple(fold(k).split()), v) for k, v in entries.items()),
            key=lambda e: len(e[0]),
            reverse=True,  # cụm dài khớp trước: "con trai" trước "con"
        )

    def match_at(self, tokens: list[Token], i: int, blocked: list[bool]) -> Span | None:
        for norm_key, fold_key, value in self._entries:
            k = len(norm_key)
            seq = tokens[i : i + k]
            if len(seq) < k or any(blocked[i : i + k]) or not all(t.is_word for t in seq):
                continue
            if all(
                (t.norm == nk) if t.has_marks else (t.fold == fk)
                for t, nk, fk in zip(seq, norm_key, fold_key, strict=True)
            ):
                return Span(i, i + k, value)
        return None

    def find(self, tokens: list[Token], blocked: list[bool]) -> list[Span]:
        spans, i = [], 0
        while i < len(tokens):
            span = self.match_at(tokens, i, blocked)
            if span:
                spans.append(span)
                i = span.end
            else:
                i += 1
        return spans


STRONG_SEPS = {",", ";", "&"}
DASHES = {"-", "–", "—"}

# Tên chương trình / cụm cố định chứa từ quan hệ nhưng không phải quan hệ (H8).
PROTECTED = Lexicon(
    {
        "anh trai say hi": "program",
        "anh trai vượt ngàn chông gai": "program",
        "chị đẹp đạp gió rẽ sóng": "program",
        "chị đẹp đạp gió": "program",
    }
)

# Dấu hiệu "cùng xuất hiện" -> giao kết quả (M2, M7). "vợ chồng" đứng trước RELATIONS
# để không bị hiểu thành quan hệ "chồng của ...".
CO_OCCURRENCE = Lexicon(
    {
        k: "co"
        for k in (
            "cùng",
            "chung",
            "chụp cùng",
            "chụp chung",
            "với",
            "với nhau",
            "bên cạnh",
            "gia đình",
            "vợ chồng",
            "cặp đôi",
            "song ca",
        )
    }
)

# Từ quan hệ -> predicate trong bảng relation.
RELATIONS = Lexicon(
    {
        "con trai": "child",
        "con gái": "child",
        "con": "child",
        "quý tử": "child",
        "ái nữ": "child",
        "vợ": "spouse",
        "chồng": "spouse",
        "bà xã": "spouse",
        "ông xã": "spouse",
        "bạn đời": "spouse",
        "bố": "parent",
        "cha": "parent",
        "mẹ": "parent",
        "anh trai": "sibling",
        "chị gái": "sibling",
        "em trai": "sibling",
        "em gái": "sibling",
    }
)

# Từ được phép nằm giữa từ quan hệ và người gốc: "con trai (thứ hai) của (tỷ phú) X".
RELATION_FILLERS = {
    fold(w)
    for w in (
        "của thứ hai ba cả út lớn nhỏ ông bà cố chủ tịch tỷ phú doanh nhân mc ca sĩ "
        "nhạc rapper nhà văn thơ diễn viên danh hài nghệ người mẫu hoa hậu cầu thủ "
        "huấn luyện vận động viên nam nữ"
    ).split()
}

# Cụm chỉ nghề -> category trong dữ liệu (singer, celeb, athletes, historic, politicer,
# businessman, scientist, writer). Dùng để phân biệt trùng tên (E3) và truy vấn chủ đề (S3, M4).
CATEGORY_HINTS = Lexicon(
    {
        "ca sĩ": "singer",
        "nhạc sĩ": "singer",
        "rapper": "singer",
        "diễn viên": "celeb",
        "mc": "celeb",
        "người dẫn chương trình": "celeb",
        "hoa hậu": "celeb",
        "người mẫu": "celeb",
        "danh hài": "celeb",
        "vận động viên": "athletes",
        "cầu thủ": "athletes",
        "huấn luyện viên": "athletes",
        "doanh nhân": "businessman",
        "tỷ phú": "businessman",
        "nhà văn": "writer",
        "nhà thơ": "writer",
        "nhà khoa học": "scientist",
        "giáo sư": "scientist",
        "chính trị gia": "politicer",
        "nhân vật lịch sử": "historic",
    }
)

# Phủ định (L7): dense retrieval kiểu CLIP xử lý kém, chỉ gắn cờ.
NEGATIONS = Lexicon({"không": "negation", "chẳng": "negation", "chưa": "negation"})

# Tách ý định (M1, M3, M4). Dấu phẩy/chấm phẩy/& xử lý riêng như STRONG_SEPS.
CONJUNCTIONS = Lexicon({"và": "and", "hoặc": "or"})

# Đếm số lượng (L8): số + danh từ đếm được.
NUMBERS = Lexicon({w: "number" for w in "một hai ba bốn năm sáu bảy tám chín mười".split()})
COUNT_NOUNS = Lexicon(
    {
        k: "count"
        for k in ("người", "cầu thủ", "ca sĩ", "thành viên", "nhân vật", "em bé", "đứa trẻ")
    }
)

# Từ chức năng: không dùng làm cửa sổ so khớp mờ, không tính là tên người lạ.
STOPWORDS = {
    fold(w)
    for w in (
        "ảnh hình của và với cùng trên tại trong ở mặc là những các một chụp người năm khi lúc đang"
    ).split()
}
