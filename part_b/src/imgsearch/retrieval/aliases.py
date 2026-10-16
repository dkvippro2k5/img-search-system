"""Từ điển alias person: tên chuẩn + biệt danh -> person_id (L1, L2, L3).

Khi có dữ liệu, dựng từ bảng `person` (cột `name`, `aliases` JSON, `category`).
Mỗi alias lưu bốn khóa: có dấu / không dấu, có khoảng trắng / viết liền
(viết liền để "Sơn Tùng MTP" khớp "Sơn Tùng M-TP").
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz, process

from imgsearch.retrieval.lexicon import Token
from imgsearch.text.normalize import normalize, strip_diacritics


class MatchType(StrEnum):
    EXACT = "exact"
    COMPACT = "compact"
    FOLDED = "folded"
    FUZZY = "fuzzy"


@dataclass(frozen=True)
class PersonEntry:
    person_id: str
    name: str
    category: str | None = None
    aliases: tuple[str, ...] = ()


class AliasIndex:
    def __init__(self, persons: Iterable[PersonEntry]):
        self.persons = {p.person_id: p for p in persons}
        self._exact: dict[str, set[str]] = defaultdict(set)
        self._compact: dict[str, set[str]] = defaultdict(set)
        self._fold: dict[str, set[str]] = defaultdict(set)
        self._fold_compact: dict[str, set[str]] = defaultdict(set)
        self.max_words = 1

        for p in self.persons.values():
            for alias in (p.name, *p.aliases):
                words = normalize(alias).split()
                if not words:
                    continue
                folds = [strip_diacritics(w) for w in words]
                self.max_words = max(self.max_words, len(words))
                self._exact[" ".join(words)].add(p.person_id)
                self._compact["".join(words)].add(p.person_id)
                self._fold[" ".join(folds)].add(p.person_id)
                self._fold_compact["".join(folds)].add(p.person_id)

        self._fuzzy_keys = list(self._fold_compact)

    def __len__(self) -> int:
        return len(self.persons)

    def lookup(self, tokens: Sequence[Token]) -> tuple[frozenset[str], MatchType] | None:
        norms = [t.norm for t in tokens]
        folds = [t.fold for t in tokens]
        for key, table, match in (
            (" ".join(norms), self._exact, MatchType.EXACT),
            ("".join(norms), self._compact, MatchType.COMPACT),
            (" ".join(folds), self._fold, MatchType.FOLDED),
            ("".join(folds), self._fold_compact, MatchType.FOLDED),
        ):
            if key in table:
                return frozenset(table[key]), match
        return None

    def fuzzy(self, tokens: Sequence[Token], cutoff: float) -> tuple[frozenset[str], float] | None:
        key = "".join(t.fold for t in tokens)
        if not self._fuzzy_keys:
            return None
        hit = process.extractOne(key, self._fuzzy_keys, scorer=fuzz.ratio, score_cutoff=cutoff)
        if hit is None:
            return None
        return frozenset(self._fold_compact[hit[0]]), hit[1]
