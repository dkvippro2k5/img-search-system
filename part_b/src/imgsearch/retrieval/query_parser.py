"""Phân tích truy vấn bằng quy tắc + từ điển alias person (PLAN.md 4.3, 4.4).

Đường chính là quy tắc (vài ms, xác định). LLM chỉ là fallback có timeout + cache,
không bao giờ là nguồn sự thật về danh tính hay quan hệ.

Thứ tự xử lý (bước trước chặn token cho bước sau):
1. Liên kết tên người khớp chính xác, khớp dài nhất trước.
2. Cụm cố định: tên chương trình, dấu hiệu cùng xuất hiện, từ quan hệ, nghề, phủ định, liên từ.
3. Liên kết tên người khớp mờ trên phần còn lại (sai chính tả, L2).
4. Phân biệt trùng tên bằng nghề đứng ngay trước (E3), tìm quan hệ (H1), tách ý định (M1–M7).
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum

from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.aliases import AliasIndex, MatchType
from imgsearch.retrieval.lexicon import Span, Token
from imgsearch.text.normalize import nfc, strip_diacritics

_TOKEN = re.compile(r"\w+|[,;&\-–—]", flags=re.UNICODE)

FUZZY_CUTOFF = 88.0
FUZZY_MIN_CHARS = 6
FUZZY_MAX_WINDOW = 4
FUZZY_MAX_QUERY_WORDS = 15  # giới hạn chi phí so khớp mờ cho truy vấn dài
MAX_RELATION_FILLERS = 6  # "con trai thứ hai của tỷ phú X": 5 từ đệm


class QueryType(StrEnum):
    SINGLE = "single"
    MULTI_INTENT = "multi_intent"
    MULTI_HOP = "multi_hop"


@dataclass(frozen=True)
class Mention:
    start: int  # chỉ số token, nửa mở
    end: int
    surface: str
    person_ids: tuple[str, ...]
    match: MatchType
    suggest: bool = False  # khớp nhờ sửa dấu/chính tả -> hiện "Bạn có muốn tìm..."

    @property
    def ambiguous(self) -> bool:
        return len(self.person_ids) > 1


@dataclass
class Intent:
    text: str
    person_ids: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class ParsedQuery:
    raw: str
    type: QueryType
    intents: list[Intent]
    mentions: list[Mention] = field(default_factory=list)
    co_occurrence: bool = False  # True: chỉ lấy ảnh có mọi người trong truy vấn (giao)
    relation_path: list[str] = field(default_factory=list)  # ví dụ ["child"]
    anchor_person_ids: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)  # tên trùng nhiều person -> hỏi lại
    suggestions: list[str] = field(default_factory=list)  # tên chuẩn đã sửa
    unknown_names: list[str] = field(default_factory=list)  # trông như tên nhưng không có
    unsupported: list[str] = field(default_factory=list)  # "negation", "count"


def tokenize(text: str) -> list[Token]:
    tokens = []
    for m in _TOKEN.finditer(nfc(text)):
        surface = m.group()
        norm = surface.lower()
        tokens.append(Token(surface, norm, strip_diacritics(norm)))
    return tokens


class QueryParser:
    def __init__(self, aliases: AliasIndex):
        self.aliases = aliases

    def parse(self, query: str) -> ParsedQuery:
        raw = nfc(query).strip()
        tokens = tokenize(raw)
        blocked = [False] * len(tokens)
        n_words = sum(t.is_word for t in tokens)

        mentions = self._link_exact(tokens, blocked, n_words)

        def take(lexicon: lx.Lexicon) -> list[Span]:
            spans = lexicon.find(tokens, blocked)
            for s in spans:
                blocked[s.start : s.end] = [True] * (s.end - s.start)
            return spans

        take(lx.PROTECTED)
        co_spans = take(lx.CO_OCCURRENCE)
        rel_spans = take(lx.RELATIONS)
        cat_spans = take(lx.CATEGORY_HINTS)
        neg_spans = take(lx.NEGATIONS)
        conj_spans = take(lx.CONJUNCTIONS)

        if n_words <= FUZZY_MAX_QUERY_WORDS:
            mentions += self._link_fuzzy(tokens, blocked)
        mentions = [
            self._disambiguate(m, cat_spans) for m in sorted(mentions, key=lambda m: m.start)
        ]

        result = ParsedQuery(raw=raw, type=QueryType.SINGLE, intents=[], mentions=mentions)
        result.ambiguous = _unique(m.surface for m in mentions if m.ambiguous)
        result.suggestions = _unique(
            self.aliases.persons[m.person_ids[0]].name
            for m in mentions
            if m.suggest and not m.ambiguous
        )
        result.unknown_names = _unknown_names(tokens, blocked)
        if neg_spans:
            result.unsupported.append("negation")
        if _has_count(tokens, blocked):
            result.unsupported.append("count")

        relation = _find_relation(tokens, rel_spans, mentions)
        if relation is not None:
            predicate, anchor = relation
            result.type = QueryType.MULTI_HOP
            result.relation_path = [predicate]
            result.anchor_person_ids = list(anchor.person_ids)
            return result

        result.intents, result.co_occurrence = _split_intents(
            tokens, mentions, cat_spans, conj_spans, has_co_marker=bool(co_spans)
        )
        if len(result.intents) > 1:
            result.type = QueryType.MULTI_INTENT
        return result

    # --- liên kết tên người ---

    def _link_exact(self, tokens: list[Token], blocked: list[bool], n_words: int) -> list[Mention]:
        mentions = []
        for block in _word_blocks(tokens):
            a = 0
            while a < len(block):
                hit = None
                for size in range(min(self.aliases.max_words, len(block) - a), 0, -1):
                    idxs = block[a : a + size]
                    # Alias một từ ("Đen") chỉ nhận khi viết hoa hoặc đứng một mình,
                    # tránh "ảnh tư liệu đen trắng" bị hiểu là rapper Đen.
                    if size == 1 and n_words > 1 and not tokens[idxs[0]].surface[:1].isupper():
                        continue
                    found = self.aliases.lookup([tokens[i] for i in idxs])
                    if found:
                        hit = (idxs, *found)
                        break
                if hit is None:
                    a += 1
                    continue
                idxs, ids, match = hit
                window = [tokens[i] for i in idxs]
                suggest = match == MatchType.FOLDED and any(t.has_marks for t in window)
                mentions.append(_mention(tokens, idxs, ids, match, suggest))
                blocked[idxs[0] : idxs[-1] + 1] = [True] * (idxs[-1] + 1 - idxs[0])
                a += len(idxs)
        return mentions

    def _link_fuzzy(self, tokens: list[Token], blocked: list[bool]) -> list[Mention]:
        mentions = []
        max_size = min(FUZZY_MAX_WINDOW, self.aliases.max_words + 1)
        for size in range(max_size, 1, -1):
            for block in _word_blocks(tokens):
                for a in range(len(block) - size + 1):
                    idxs = block[a : a + size]
                    window = [tokens[i] for i in idxs]
                    if any(blocked[i] for i in idxs) or any(t.fold in lx.STOPWORDS for t in window):
                        continue
                    if sum(len(t.fold) for t in window) < FUZZY_MIN_CHARS:
                        continue
                    found = self.aliases.fuzzy(window, FUZZY_CUTOFF)
                    if found:
                        mentions.append(_mention(tokens, idxs, found[0], MatchType.FUZZY, True))
                        blocked[idxs[0] : idxs[-1] + 1] = [True] * (idxs[-1] + 1 - idxs[0])
        return mentions

    def _disambiguate(self, m: Mention, cat_spans: list[Span]) -> Mention:
        """Trùng tên: giữ person đúng nghề nếu nghề đứng ngay trước tên ("nhà văn Sơn Tùng")."""
        if not m.ambiguous:
            return m
        for span in cat_spans:
            if span.end == m.start:
                keep = tuple(
                    pid for pid in m.person_ids if self.aliases.persons[pid].category == span.value
                )
                if keep:
                    return Mention(m.start, m.end, m.surface, keep, m.match, m.suggest)
        return m


# --- hàm phụ ---


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def _word_blocks(tokens: list[Token]) -> list[list[int]]:
    """Chỉ số các token là từ, chia khối tại , ; & (gạch ngang không chia: "M-TP")."""
    blocks, cur = [], []
    for i, t in enumerate(tokens):
        if t.norm in lx.STRONG_SEPS:
            if cur:
                blocks.append(cur)
            cur = []
        elif t.is_word:
            cur.append(i)
    if cur:
        blocks.append(cur)
    return blocks


def _mention(tokens, idxs, ids, match, suggest) -> Mention:
    start, end = idxs[0], idxs[-1] + 1
    surface = " ".join(t.surface for t in tokens[start:end] if t.is_word)
    return Mention(start, end, surface, tuple(sorted(ids)), match, suggest)


def _find_relation(
    tokens: list[Token], rel_spans: list[Span], mentions: list[Mention]
) -> tuple[str, Mention] | None:
    """Từ quan hệ + (từ đệm)* + người gốc. Không có người gốc thì không phải quan hệ (H8)."""
    by_start = {m.start: m for m in mentions}
    for span in rel_spans:
        j, skipped = span.end, 0
        while j < len(tokens) and j not in by_start and skipped < MAX_RELATION_FILLERS:
            if not (tokens[j].is_word and tokens[j].fold in lx.RELATION_FILLERS):
                break
            j += 1
            skipped += 1
        if j in by_start:
            return span.value, by_start[j]
    return None


def _split_intents(
    tokens: list[Token],
    mentions: list[Mention],
    cat_spans: list[Span],
    conj_spans: list[Span],
    has_co_marker: bool,
) -> tuple[list[Intent], bool]:
    cuts = {s.start for s in conj_spans} | {
        i for i, t in enumerate(tokens) if t.norm in lx.STRONG_SEPS
    }
    # Đoạn giữa các liên từ: (start, end)
    segments, start = [], 0
    for i in sorted(cuts):
        segments.append((start, i))
        start = i + 1
    segments.append((start, len(tokens)))

    groups: list[dict] = []
    for s, e in segments:
        words = [t.surface for t in tokens[s:e] if t.is_word]
        if not words:
            continue
        seg_mentions = [m for m in mentions if s <= m.start < e]
        seg_cats = [c.value for c in cat_spans if s <= c.start < e]
        # Vế không có người, không có nghề là thuộc tính nối tiếp vế trước:
        # "Hồ Ngọc Hà mặc áo dài và đội nón lá" vẫn là một ý định.
        if groups and not seg_mentions and not seg_cats:
            groups[-1]["words"] += words
            continue
        groups.append({"words": words, "mentions": seg_mentions, "cats": seg_cats})

    if len(groups) == 1 and len({m.person_ids for m in groups[0]["mentions"]}) > 1:
        # Một vế nhắc nhiều người không qua liên từ ("chụp cùng", "Gia đình A – B"):
        # mỗi người một ý định, nhưng kết quả là ảnh có cả hai.
        g = groups[0]
        intents = [Intent(m.surface, list(m.person_ids), g["cats"]) for m in g["mentions"]]
        return intents, True

    intents = [
        Intent(
            " ".join(g["words"]),
            _unique(pid for m in g["mentions"] for pid in m.person_ids),
            _unique(g["cats"]),
        )
        for g in groups
    ]
    return intents, has_co_marker and len(intents) > 1


def _unknown_names(tokens: list[Token], blocked: list[bool]) -> list[str]:
    """Chuỗi >= 2 từ viết hoa liền nhau chưa liên kết được: có thể là người không có (E6, X1)."""
    names, run = [], []
    for i, t in enumerate(tokens + [Token("", "", "")]):
        capital = (
            t.is_word
            and t.surface[:1].isupper()
            and i < len(blocked)
            and not blocked[i]
            and t.fold not in lx.STOPWORDS
        )
        if capital:
            run.append(t.surface)
            continue
        if len(run) >= 2:
            names.append(" ".join(run))
        run = []
    return names


def _has_count(tokens: list[Token], blocked: list[bool]) -> bool:
    # Danh từ đếm ("cầu thủ", "ca sĩ") thường đã bị nhận là nghề, nên bỏ qua `blocked`
    # ở danh từ; chỉ yêu cầu bản thân con số chưa thuộc tên người hay cụm khác.
    free = [False] * len(tokens)
    for i, t in enumerate(tokens):
        if blocked[i]:
            continue
        is_number = (t.norm.isdigit() and len(t.norm) <= 2) or lx.NUMBERS.match_at(tokens, i, free)
        if is_number and lx.COUNT_NOUNS.match_at(tokens, i + 1, free):
            return True
    return False
