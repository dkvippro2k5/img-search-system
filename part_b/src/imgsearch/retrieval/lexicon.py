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

# Cụm cố định chứa từ quan hệ nhưng không phải quan hệ: tên chương trình (H8) và từ ghép
# ("tuyên bố" không phải "bố", "có con với X" không phải "con của X").
PROTECTED = Lexicon(
    {
        "anh trai say hi": "program",
        "anh trai vượt ngàn chông gai": "program",
        "chị đẹp đạp gió rẽ sóng": "program",
        "chị đẹp đạp gió": "program",
        **{
            k: "compound"
            for k in (
                "tuyên bố",
                "công bố",
                "phân bố",
                "ban bố",
                "bố trí",
                "bố cục",
                "bố cáo",
                "có con",
                "con số",
                "con đường",
                "con người",
                "con giáp",
                "mẹ thiên nhiên",
                "cha xứ",
            )
        },
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
        "của thứ hai ba cả út lớn nhỏ ông bà cố chủ tịch tỷ tỉ phú doanh nhân mc ca sĩ "
        "nhạc rapper nhà văn thơ diễn viên danh hài nghệ người mẫu hoa á hậu cầu thủ "
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

# Nhãn hiển thị trong thông báo / nhóm kết quả.
PREDICATE_LABELS = {
    "child": "con",
    "spouse": "vợ/chồng",
    "parent": "bố/mẹ",
    "sibling": "anh chị em",
    "member_of": "thành viên",
}
CATEGORY_LABELS = {
    "singer": "ca sĩ",
    "celeb": "người nổi tiếng",
    "athletes": "vận động viên",
    "historic": "nhân vật lịch sử",
    "politicer": "chính khách",
    "businessman": "doanh nhân",
    "scientist": "nhà khoa học",
    "writer": "nhà văn",
}

# --- Gộp entity (data/entity_merge.py) ---
# Cụm mô tả hay đứng đầu tên entity, bỏ đi để lấy tên người:
# "Sếp Phạm Nhật Vượng Vingroup…", "Tiền vệ Nguyễn Quang Hải", "Streamer Độ Mixi (giữ)".
ENTITY_PREFIXES = Lexicon(
    {
        k: "prefix"
        for k in (
            "sếp",
            "tỷ phú",
            "tỉ phú",
            "doanh nhân",
            "chủ tịch",
            "ceo",
            "tiền vệ",
            "tiền đạo",
            "hậu vệ",
            "thủ môn",
            "cầu thủ",
            "huấn luyện viên",
            "tay vợt",
            "tay vợt cầu lông",
            "tay vợt tennis",
            "tay vợt tennis trẻ",
            "võ sĩ",
            "vận động viên",
            "vđv",
            "kình ngư",
            "streamer",
            "youtuber",
            "tiktoker",
            "rapper",
            "ca sĩ",
            "nhạc sĩ",
            "mc",
            "diễn viên",
            "người mẫu",
            "siêu mẫu",
            "hoa hậu",
            "á hậu",
            "nhà văn",
            "nhà thơ",
            "giáo sư",
            "gs",
            "nsnd",
            "nsưt",
            "cố",
            "chân dung",
            "chân dung studio",
            "headshot",
            "toàn thân",
            "ảnh",
            "sinh nhật",
            "surprise",
            "thổi nến",
        )
    }
)
# Nhóm nghề coi là "cùng giới": cùng tên + cùng nhóm -> cùng một người.
# singer và celeb gộp chung vì dữ liệu gán lẫn (Hari Won có entity singer và celeb).
CATEGORY_FAMILY = {"singer": "ent", "celeb": "ent"}
# Họ / chữ lót không dùng làm đầu biệt danh hai âm tiết ("Thị Trang", "Văn Bình").
NOT_ALIAS_START = {fold(w) for w in "thị văn".split()}

# --- Trích quan hệ từ tiêu đề bài báo (kg/article_relations.py) ---
# Bổ sung cho RELATIONS: cách nói chỉ gặp trong tiêu đề.
TITLE_RELATIONS = Lexicon({"cưới": "spouse", "kết hôn với": "spouse", "lấy vợ": "spouse"})
# Từ đứng ngay trước từ quan hệ khiến nó không phải quan hệ thật ("đóng vợ Trấn Thành").
NON_RELATION_PRECEDERS = {fold(w) for w in "đóng vai".split()}

# Cụm viết hoa nhưng không phải tên người: địa danh, môn thể thao, chương trình, giải đấu.
# Không dùng làm biệt danh, không coi là "tên người lạ" (E6) và không gộp entity theo chúng.
NON_PERSON_NAMES = {
    fold(w)
    for w in (
        "Việt Nam",
        "Hà Nội",
        "Sài Gòn",
        "TP HCM",
        "TP Hồ Chí Minh",
        "Hồ Chí Minh",
        "Đà Nẵng",
        "Hải Phòng",
        "Cần Thơ",
        "Huế",
        "Nha Trang",
        "Đà Lạt",
        "Phú Quốc",
        "Hạ Long",
        "Vũng Tàu",
        "Hàn Quốc",
        "Nhật Bản",
        "Trung Quốc",
        "Đài Loan",
        "Thái Lan",
        "Hoa Kỳ",
        "Mỹ",
        "Pháp",
        "Muay Thái",
        "Taekwondo",
        "Davis Cup",
        "SEA Games",
        "Asian Games",
        "Olympic",
        "Rap Việt",
        "Sing My Song",
        "Anh Trai Say Hi",
        "Chị Đẹp Đạp Gió",
        "Miss Universe",
        "Nhạc Viện",
        "Nhạc viện Hà Nội",
    )
}
