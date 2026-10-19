"""Trích quan hệ có bằng chứng từ tiêu đề bài báo (`description`). PLAN.md 4.5 (v0).

Mẫu: <người X> ... <từ quan hệ> (tối đa vài từ đệm) <người Y>, ví dụ
  "Doanh nhân Phạm Nhật Minh Hoàng - con trai thứ 2 của tỷ phú Phạm Nhật Vượng"
  -> (Phạm Nhật Vượng, child, Phạm Nhật Minh Hoàng)  [obj là <predicate> của subj]
  "Vợ chồng Trấn Thành – Hari Won ..." -> (Trấn Thành, spouse, Hari Won)

Chỉ nhận khi CẢ HAI đầu liên kết được tới đúng một person, và lưu tiêu đề + URL làm bằng chứng.
Ưu tiên độ chính xác: tên phải khớp đúng (không sửa chính tả), viết hoa, >= 2 từ; mỗi mệnh đề
(tách theo ; : | /) xét riêng; đuôi tên báo bị cắt ("- Báo Người Lao Động" không thành "Bảo").
Không suy quan hệ qua hai bước ("Con trai ông Vượng làm Chủ tịch VinFast" + "Quân Anh làm
Chủ tịch VinFast" không thành cạnh). Bẫy được loại:
  - từ quan hệ trong ngoặc kép: Tuấn Trần – “con trai” Trấn Thành (cách gọi thân mật)
  - đứng sau "đóng", "vai": đóng vợ Trấn Thành (vai diễn)
  - tên chương trình: Anh Trai Say Hi
  - câu nói trong ngoặc kép (mọi độ dài): "Tôi muốn … của chồng"
  - địa danh trùng tên người: TP. Hồ Chí Minh
  - "X <động từ> em trai Y", "X cùng vợ ông Y": quan hệ là của Y, không phải của X
  - "chị vợ", "mẹ chồng"… (họ hàng bên vợ/chồng), "đám cưới của X"
"""

import re
from collections.abc import Iterable

from imgsearch.data.dataset import RelationRec
from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.aliases import MatchType
from imgsearch.retrieval.query_parser import Mention, QueryParser, tokenize
from imgsearch.text.normalize import fold

ANCHOR_WINDOW = 6  # số từ tối đa giữa từ quan hệ và người được nhắc sau nó
OTHER_WINDOW = 10  # số từ tối đa giữa chủ ngữ và từ tự xưng ("ông xã", "cưới")
SYMMETRIC = {"spouse", "sibling"}
MAX_EVIDENCE_PER_EDGE = 3
# Ngoặc kép: mở không đứng sau chữ, đóng không đứng trước chữ (giữ "H'Hen Niê").
_QUOTED = re.compile(r"(?<!\w)[“\"«‘'][^”\"»’]*?[”\"»’'](?!\w)")
_QUOTE_CHARS = "“\"«‘'"
_CLAUSE_SEP = re.compile(r"[;:!?|]|\s/\s")
_TAIL_SEP = re.compile(r"\s[-–—]\s")
_ACCEPTED = {MatchType.EXACT, MatchType.COMPACT}


def extract_relations(
    titles: Iterable[tuple[str, str | None]], parser: QueryParser
) -> list[RelationRec]:
    """titles: (tiêu đề, URL nguồn). Mỗi cạnh giữ tối đa MAX_EVIDENCE_PER_EDGE bằng chứng."""
    edges: dict[tuple[str, str, str], list[RelationRec]] = {}
    for title, url in titles:
        for subj, predicate, obj in relations_in_title(title, parser):
            key = (subj, predicate, obj)
            if predicate in SYMMETRIC:  # đối xứng: một cạnh cho mỗi cặp
                key = (min(subj, obj), predicate, max(subj, obj))
            found = edges.setdefault(key, [])
            if len(found) < MAX_EVIDENCE_PER_EDGE and all(r.evidence != title for r in found):
                found.append(RelationRec(key[0], predicate, key[2], "article", title, url))
    return [r for rs in edges.values() for r in rs]


def clauses(title: str) -> list[str]:
    """Bỏ ngoặc kép, tách mệnh đề, cắt đuôi tên báo."""
    out = []
    for clause in _CLAUSE_SEP.split(_QUOTED.sub(" ", title)):
        parts = _TAIL_SEP.split(clause)
        while len(parts) > 1 and fold(parts[-1].strip()).startswith(lx.PUBLISHER_MARKERS):
            parts.pop()
        text = " - ".join(parts).strip()
        # Mệnh đề là câu nói (còn dấu ngoặc mở do lồng ngoặc) hoặc chỉ là tên báo: bỏ.
        if text and text[0] not in _QUOTE_CHARS and not fold(text).startswith(lx.PUBLISHER_MARKERS):
            out.append(text)
    return out


def relations_in_title(title: str, parser: QueryParser) -> list[tuple[str, str, str]]:
    return [edge for clause in clauses(title) for edge in _relations_in_clause(clause, parser)]


def _capital(t) -> bool:
    return t.is_word and (t.surface[:1].isupper() or t.surface[:1].isdigit())


def _cuts_name_on_left(t) -> bool:
    """Từ đứng trước làm tên bị cụt: viết hoa, không phải chữ viết tắt (MC, NSND) hay kính ngữ
    / nghề (Bà, Ông, Ca sĩ)."""
    return (
        t.is_word
        and t.surface[:1].isupper()
        and not t.surface.isupper()
        and t.fold not in lx.APPOSITION_WORDS
    )


def _usable(m: Mention, tokens) -> bool:
    """Tên trong tiêu đề: khớp đúng, không mơ hồ, >= 2 từ, viết hoa, không phải địa danh, và
    không bị cụt hai đầu ("Nguyễn Ánh 9" là nhạc sĩ, không phải "Nguyễn Ánh"; "Đàm Thu Trang"
    không phải "Thu Trang"). Từ viết hoa toàn bộ đứng trước được phép (MC, NSND)."""
    words = [t for t in tokens[m.start : m.end] if t.is_word]
    folds = [t.fold for t in words]
    return (
        not m.ambiguous
        and m.match in _ACCEPTED
        and len(words) >= 2
        and all(_capital(t) for t in words)
        and not (m.end < len(tokens) and _capital(tokens[m.end]))
        and not (m.start > 0 and _cuts_name_on_left(tokens[m.start - 1]))
        and not (m.start > 0 and tokens[m.start - 1].fold in lx.PLACE_PRECEDERS)
        and not (
            m.start > 1 and [t.fold for t in tokens[m.start - 2 : m.start]] == ["thanh", "pho"]
        )
        and not _has_non_person(folds)
    )


def _has_non_person(folds: list[str]) -> bool:
    """Tên chứa cụm không phải người ("HCV SEA Games")."""
    n = len(folds)
    return any(
        " ".join(folds[i:j]) in lx.NON_PERSON_NAMES for i in range(n) for j in range(i + 1, n + 1)
    )


def _relations_in_clause(text: str, parser: QueryParser) -> list[tuple[str, str, str]]:
    tokens = tokenize(text)
    parsed = parser.parse(text)
    mentions = [m for m in parsed.mentions if _usable(m, tokens)]
    blocked = [False] * len(tokens)
    for m in parsed.mentions:
        blocked[m.start : m.end] = [True] * (m.end - m.start)
    protected = [False] * len(tokens)
    for span in lx.PROTECTED.find(tokens, blocked):
        blocked[span.start : span.end] = [True] * (span.end - span.start)
        protected[span.start : span.end] = [True] * (span.end - span.start)

    out = []
    i = 0
    while i < len(tokens):
        # "Vợ chồng A – B": hai người đứng sát nhau ngay sau (chỉ cách bởi –, &, và).
        # "Vợ chồng A đi…, B…" thì không: B không phải vợ/chồng của A.
        if _is_couple(tokens, i, blocked):
            pair = _adjacent_pair(tokens, mentions, i + 2)
            if pair:
                out.append((pair[0].person_ids[0], "spouse", pair[1].person_ids[0]))
            i += 2
            continue
        span = lx.RELATIONS.match_at(tokens, i, blocked)
        from_title = span is None
        if from_title:
            span = lx.TITLE_RELATIONS.match_at(tokens, i, blocked)
        if span is None:
            i += 1
            continue
        i = span.end
        if span.start > 0 and tokens[span.start - 1].fold in lx.NON_RELATION_PRECEDERS:
            continue
        anchor = _first_after(tokens, mentions, span.end)
        other = _last_before(mentions, span.start)
        surface = " ".join(t.norm for t in tokens[span.start : span.end])
        self_ref = from_title or surface in lx.SELF_RELATIONS
        if other and not _other_ok(tokens, protected, other, span, self_ref):
            continue
        if anchor and other and anchor.person_ids != other.person_ids:
            # "<other> - con trai của <anchor>": other là con của anchor.
            out.append((anchor.person_ids[0], span.value, other.person_ids[0]))
    return out


def _is_couple(tokens, i, blocked) -> bool:
    pair = tokens[i : i + 2]
    return (
        len(pair) == 2
        and not any(blocked[i : i + 2])
        and [t.fold for t in pair] == ["vo", "chong"]
        and (pair[0].norm == "vợ" or not pair[0].has_marks)
    )


def _first_after(tokens, mentions: list[Mention], pos: int) -> Mention | None:
    """Người được nhắc ngay sau từ quan hệ. Các từ ở giữa phải là từ đệm ("thứ 2 của tỷ phú")
    hoặc có "của" ("vợ kín tiếng của tỷ phú"); "có con với X" thì không nhận."""
    for m in mentions:
        if pos <= m.start <= pos + ANCHOR_WINDOW:
            if any(t.norm in lx.STRONG_SEPS for t in tokens[pos : m.start]):
                return None  # "…sinh con, Trấn Thành nhắn nhủ": sang mệnh đề khác
            between = [t for t in tokens[pos : m.start] if t.is_word]
            if all(t.fold in lx.RELATION_FILLERS or t.norm.isdigit() for t in between) or any(
                t.fold == "cua" for t in between
            ):
                return m
            return None
    return None


def _adjacent_pair(tokens, mentions: list[Mention], pos: int) -> tuple[Mention, Mention] | None:
    after = [m for m in mentions if m.start >= pos][:2]
    if len(after) < 2 or after[0].start != pos or after[0].person_ids == after[1].person_ids:
        return None
    gap = tokens[after[0].end : after[1].start]
    if all(t.norm in lx.DASHES or t.norm in {"&", "và"} for t in gap):
        return after[0], after[1]
    return None


def _other_ok(tokens, protected, other: Mention, span, self_ref: bool) -> bool:
    """Người đứng trước từ quan hệ có thật là người kia của quan hệ không. Chỉ nhận hai dạng:
    - đồng vị: giữa X và từ quan hệ chỉ có gạch, phẩy, từ đệm ("Thái Hiền - con gái nhạc sĩ Y",
      "Sơn Tuyền, em gái danh ca Y");
    - từ tự xưng, luôn nói về chủ ngữ (ông xã, bà xã, cưới, kết hôn với, làm vợ), cách tối đa
      OTHER_WINDOW từ ("Hari Won … cùng ông xã Trấn Thành").
    "X <động từ> em trai Y" thì không: thường là em trai của Y ("Sara Lưu đóng cảnh với em trai
    Đăng Khôi"). Cũng bỏ khi X là người sở hữu ("Vợ của X cưới Y") hoặc có từ ghép bị chặn ở
    giữa ("X có con riêng với em gái Y")."""
    if any(protected[other.end : span.start]):
        return False
    before = tokens[max(0, other.start - 3) : other.start]
    if lx.RELATIONS.find(before, [False] * len(before)):
        return False
    between = [t for t in tokens[other.end : span.start] if t.is_word]
    if self_ref:
        return len(between) <= OTHER_WINDOW
    if not all(t.fold in lx.APPOSITION_WORDS or t.norm.isdigit() for t in between):
        return False
    if any(t.norm == "," for t in tokens[other.end : span.start]):
        # "X, con gái Y" chỉ khi X đứng đầu mệnh đề; "… cưới ca sĩ X, con gái Y" thì phần sau
        # dấu phẩy nói về chủ ngữ, không phải X.
        lead = [t for t in tokens[: other.start] if t.is_word]
        return all(t.fold in lx.APPOSITION_WORDS or t.surface.isupper() for t in lead)
    return True


def _last_before(mentions: list[Mention], pos: int) -> Mention | None:
    before = [m for m in mentions if m.end <= pos]
    return before[-1] if before else None
