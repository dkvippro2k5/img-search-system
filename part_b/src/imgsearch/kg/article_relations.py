"""Trích quan hệ có bằng chứng từ tiêu đề bài báo (`description`). PLAN.md 4.5 (v0).

Mẫu: <người X> ... <từ quan hệ> (tối đa vài từ đệm) <người Y>, ví dụ
  "Doanh nhân Phạm Nhật Minh Hoàng - con trai thứ 2 của tỷ phú Phạm Nhật Vượng"
  -> (Phạm Nhật Vượng, child, Phạm Nhật Minh Hoàng)  [obj là <predicate> của subj]
  "Vợ chồng Trấn Thành – Hari Won ..." -> (Trấn Thành, spouse, Hari Won)

Chỉ nhận khi CẢ HAI đầu liên kết được tới đúng một person, và lưu tiêu đề + URL làm bằng chứng.
Không suy quan hệ qua hai bước ("Con trai ông Vượng làm Chủ tịch VinFast" + "Quân Anh làm
Chủ tịch VinFast" không thành cạnh). Bẫy được loại:
  - từ quan hệ trong ngoặc kép: Tuấn Trần – “con trai” Trấn Thành (cách gọi thân mật)
  - đứng sau "đóng", "vai": đóng vợ Trấn Thành (vai diễn)
  - tên chương trình: Anh Trai Say Hi
"""

import re
from collections.abc import Iterable

from imgsearch.data.dataset import RelationRec
from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.query_parser import Mention, QueryParser, tokenize

ANCHOR_WINDOW = 6  # số từ tối đa giữa từ quan hệ và người được nhắc sau nó
MAX_EVIDENCE_PER_EDGE = 3
_QUOTED = re.compile(r"[“\"«‘'][^”\"»’']{1,30}[”\"»’']")


def extract_relations(
    titles: Iterable[tuple[str, str | None]], parser: QueryParser
) -> list[RelationRec]:
    """titles: (tiêu đề, URL nguồn). Mỗi cạnh giữ tối đa MAX_EVIDENCE_PER_EDGE bằng chứng."""
    edges: dict[tuple[str, str, str], list[RelationRec]] = {}
    for title, url in titles:
        for subj, predicate, obj in relations_in_title(title, parser):
            key = (subj, predicate, obj)
            if predicate == "spouse":  # đối xứng: một cạnh cho mỗi cặp
                key = (min(subj, obj), predicate, max(subj, obj))
            found = edges.setdefault(key, [])
            if len(found) < MAX_EVIDENCE_PER_EDGE and all(r.evidence != title for r in found):
                found.append(RelationRec(key[0], predicate, key[2], "article", title, url))
    return [r for rs in edges.values() for r in rs]


def relations_in_title(title: str, parser: QueryParser) -> list[tuple[str, str, str]]:
    text = _QUOTED.sub(" ", title)
    tokens = tokenize(text)
    parsed = parser.parse(text)
    mentions = [m for m in parsed.mentions if not m.ambiguous]
    blocked = [False] * len(tokens)
    for m in parsed.mentions:
        blocked[m.start : m.end] = [True] * (m.end - m.start)
    for span in lx.PROTECTED.find(tokens, blocked):
        blocked[span.start : span.end] = [True] * (span.end - span.start)

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
        span = lx.RELATIONS.match_at(tokens, i, blocked) or lx.TITLE_RELATIONS.match_at(
            tokens, i, blocked
        )
        if span is None:
            i += 1
            continue
        i = span.end
        if span.start > 0 and tokens[span.start - 1].fold in lx.NON_RELATION_PRECEDERS:
            continue
        anchor = _first_after(tokens, mentions, span.end)
        other = _last_before(mentions, span.start)
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


def _last_before(mentions: list[Mention], pos: int) -> Mention | None:
    before = [m for m in mentions if m.end <= pos]
    return before[-1] if before else None
