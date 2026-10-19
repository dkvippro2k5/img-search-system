"""Gộp entity thành person bằng tên (v0, chưa dùng khuôn mặt). PLAN 3.4 mục A3, SCENARIOS E4.

Tên entity trong dữ liệu thường là cụm mô tả: "Sếp Phạm Nhật Vượng Vingroup quản lý tập đoàn",
"Streamer Độ Mixi (giữ)", "Trấn Thành & Hari Won cặp đôi". Các bước:

1. Tách tên: bỏ phần sau " - ", tách ngoặc, bỏ cụm mô tả đầu tên (lexicon.ENTITY_PREFIXES).
   Trong ngoặc là 2–4 từ viết hoa và một trong hai tên dài >= 3 từ thì coi là biệt danh
   ("Phùng Thanh Độ (Độ Mixi)", "Will (Nguyễn Anh Tuấn)"); "Anh Tú (The Voice)" thì không.
2. Entity "sạch" (chỉ còn từ viết hoa) tạo cụm theo (tên, nhóm nghề). Cùng tên khác nhóm nghề
   là người khác nhau ("Quang Hải" cầu thủ / ca sĩ / nhạc trưởng). Biệt danh chỉ dùng để tra,
   KHÔNG gộp hai cụm đã có (một biệt danh trỏ hai cụm thì bỏ): tránh nối chuỗi nhiều người
   qua một tên chương trình, ban nhạc.
3. Tên 2 âm tiết là đuôi của đúng một tên dài hơn cùng nhóm nghề thì gộp
   ("Quang Hải" -> "Nguyễn Quang Hải").
4. Entity còn lại: tìm các tên đã biết (>= 2 từ) nằm trong nó, dài nhất trước. Thấy nhiều người
   thì gắn cho tất cả ("Trấn Thành & Hari Won cặp đôi"); không thấy ai thì lấy cụm từ viết hoa
   dài nhất.

Khóa so khớp GIỮ DẤU: "Tuấn Hưng" và "Tuấn Hùng" là hai người.

Giới hạn: không sửa được lỗi gán entity (ảnh ca sĩ dưới entity "Sơn Tùng" nhà văn, E8), không
tách được entity trộn hai người; cần so khuôn mặt (Phần A). Mỗi entity ghi `method` để rà lại.
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from imgsearch.retrieval import lexicon as lx
from imgsearch.retrieval.lexicon import Token
from imgsearch.text.normalize import nfc, strip_diacritics

_WORD = re.compile(r"\w[\w\-]*", flags=re.UNICODE)


@dataclass(frozen=True)
class EntityInput:
    entity_id: str
    name: str
    category: str | None = None
    n_images: int = 1


@dataclass
class MergedPerson:
    person_id: str
    name: str
    category: str | None
    aliases: list[str]
    entity_ids: list[str]


@dataclass
class MergeResult:
    persons: list[MergedPerson]
    entity_persons: dict[str, list[str]]  # entity_id -> person_id (nhiều người: ảnh nhóm)
    methods: dict[str, str]  # entity_id -> name | name_contains | name_fallback


@dataclass
class ParsedName:
    words: list[str]  # tên chính sau khi bỏ cụm mô tả
    aliases: list[list[str]]  # tên trong ngoặc
    clean: bool  # chỉ gồm từ viết hoa: là một tên người


def key_of(words: list[str]) -> tuple[str, ...]:
    """Khóa so khớp: chữ thường, giữ dấu, bỏ gạch nối ("M-TP" = "MTP")."""
    return tuple(nfc(w.lower()).replace("-", "") for w in words)


def id_of(words: list[str]) -> str:
    """person_id ASCII: "p_son_tung_mtp"."""
    return "p_" + "_".join(strip_diacritics(w.lower()).replace("-", "") for w in words)


def _is_name_word(w: str) -> bool:
    return w[:1].isupper() and not w.isdigit()


def _is_non_person(words: list[str]) -> bool:
    return strip_diacritics(" ".join(words).lower()) in lx.NON_PERSON_NAMES


def _family(category: str | None) -> str:
    return lx.CATEGORY_FAMILY.get(category or "", category or "")


def _strip_prefixes(words: list[str]) -> list[str]:
    while len(words) > 1 and "-" in words[0]:
        if not all(p.lower() in lx.ACADEMIC_TITLES for p in words[0].split("-")):
            break
        words = words[1:]
    while words:
        tokens = [Token(w, w.lower(), strip_diacritics(w.lower())) for w in words]
        span = lx.ENTITY_PREFIXES.match_at(tokens, 0, [False] * len(tokens))
        if span is None or span.end >= len(words):
            break
        words = words[span.end :]
    return words


def parse_name(name: str) -> ParsedName:
    name = nfc(name).strip()
    parens = re.findall(r"\(([^)]*)\)", name)
    main = re.sub(r"\([^)]*\)", " ", name)
    main = re.split(r"\s[-–—]\s", main)[0]
    words = _strip_prefixes(_WORD.findall(main))
    aliases = []
    for paren in parens:
        for part in paren.split(","):
            w = _WORD.findall(part)
            if (
                2 <= len(w) <= 4
                and max(len(w), len(words)) >= 3
                and all(_is_name_word(x) and not x.isupper() for x in w)
                and not _is_non_person(w)
            ):
                aliases.append(w)
    clean = 1 <= len(words) <= 5 and all(_is_name_word(w) for w in words)
    return ParsedName(words, aliases, clean)


def _longest_name_run(words: list[str]) -> list[str]:
    runs, cur = [], []
    for w in [*words, ""]:
        if w and _is_name_word(w):
            cur.append(w)
        else:
            if cur:
                runs.append(cur)
            cur = []
    long = [r for r in runs if len(r) >= 2]
    if long:
        return max(long, key=len)
    return runs[0] if runs else words


def derived_aliases(words: list[str]) -> list[str]:
    """Cách gọi tắt phổ biến: "Sơn Tùng M-TP" -> "Sơn Tùng"; "Trần Khả Như" -> "Khả Như"."""
    out = []
    if len(words) >= 3 and ("-" in words[-1] or words[-1].isupper()):
        out.append(" ".join(words[:-1]))
    elif len(words) >= 3:
        last2 = words[-2:]
        simple = all("-" not in w and not w.isupper() for w in last2)
        if simple and strip_diacritics(last2[0].lower()) not in lx.NOT_ALIAS_START:
            out.append(" ".join(last2))
    return [a for a in out if not _is_non_person(a.split())]


@dataclass
class _Cluster:
    family: str
    forms: Counter = field(default_factory=Counter)  # cách viết tên -> số ảnh
    aliases: set[str] = field(default_factory=set)
    categories: Counter = field(default_factory=Counter)
    entity_ids: list[str] = field(default_factory=list)


class _Merger:
    def __init__(self):
        self.clusters: list[_Cluster] = []
        self.parent: list[int] = []
        self.index: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
        self.main_keys: set[tuple[tuple[str, ...], str]] = set()  # tên chính của entity sạch
        self.bad_alias: set[tuple[tuple[str, ...], str]] = set()  # biệt danh trỏ >= 2 cụm

    def find(self, c: int) -> int:
        while self.parent[c] != c:
            self.parent[c] = self.parent[self.parent[c]]
            c = self.parent[c]
        return c

    def union(self, a: int, b: int) -> int:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a
        return a

    def cluster_for(self, key: tuple[str, ...], family: str) -> int:
        existing = self.index[key].get(family)
        if existing is not None:
            return self.find(existing)
        self.clusters.append(_Cluster(family))
        self.parent.append(len(self.clusters) - 1)
        self.index[key][family] = len(self.clusters) - 1
        self.main_keys.add((key, family))
        return len(self.clusters) - 1

    def register_alias(self, key: tuple[str, ...], family: str, c: int) -> bool:
        """Biệt danh trùng tên chính của cụm khác -> gộp ("Phùng Thanh Độ (Độ Mixi)" + "Độ Mixi").
        Hai biệt danh trùng nhau ở hai cụm -> mơ hồ, bỏ cả hai, KHÔNG gộp. Trả True nếu giữ."""
        if (key, family) in self.bad_alias:
            return False
        existing = self.index[key].get(family)
        if existing is None:
            self.index[key][family] = c
            return True
        if (key, family) in self.main_keys:
            self.union(existing, c)
            return True
        if self.find(existing) == self.find(c):
            return True
        del self.index[key][family]
        self.bad_alias.add((key, family))
        return False

    def lookup(self, key: tuple[str, ...], family: str) -> int | None:
        options = self.index.get(key)
        if not options:
            return None
        if family in options:
            return self.find(options[family])
        roots = {self.find(c) for c in options.values()}
        return roots.pop() if len(roots) == 1 else None


def merge_entities(entities: list[EntityInput]) -> MergeResult:
    m = _Merger()
    parsed = {e.entity_id: parse_name(e.name) for e in entities}
    assigned: dict[str, list[int]] = {}
    methods: dict[str, str] = {}

    def attach(c: int, e: EntityInput, form: str | None) -> None:
        cl = m.clusters[c]
        if form:
            cl.forms[form] += e.n_images
        cl.categories[e.category] += e.n_images
        cl.entity_ids.append(e.entity_id)

    # 1-2. Entity sạch tạo cụm; biệt danh trong ngoặc trỏ về cùng cụm.
    pending_aliases: list[tuple[list[list[str]], str, int]] = []
    ordered = sorted(entities, key=lambda e: (-e.n_images, e.entity_id))
    for e in ordered:
        p = parsed[e.entity_id]
        if not p.clean:
            continue
        family = _family(e.category)
        c = m.cluster_for(key_of(p.words), family)
        attach(c, e, " ".join(p.words))
        assigned[e.entity_id] = [c]
        methods[e.entity_id] = "name"
        pending_aliases.append((p.aliases, family, c))
    # Biệt danh đăng ký sau khi mọi tên chính đã có cụm, để không chiếm chỗ của tên thật.
    for aliases, family, c in pending_aliases:
        for alias in aliases:
            if m.register_alias(key_of(alias), family, c):
                m.clusters[m.find(c)].aliases.add(" ".join(alias))

    # 3. "Quang Hải" -> "Nguyễn Quang Hải" nếu là đuôi của đúng một tên dài hơn cùng nhóm nghề.
    by_suffix: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
    for k in m.index:
        if len(k) >= 3:
            by_suffix[k[-2:]].append(k)
    for key, options in list(m.index.items()):
        if len(key) != 2:
            continue
        for family, c in options.items():
            longer = {
                m.find(m.index[k2][family])
                for k2 in by_suffix.get(key, [])
                if family in m.index[k2]
            }
            if len(longer) == 1:
                m.union(longer.pop(), c)

    # 4. Entity mô tả: tìm tên đã biết nằm trong nó.
    max_len = max((len(k) for k in m.index), default=1)
    for e in ordered:
        if e.entity_id in assigned:
            continue
        p = parsed[e.entity_id]
        family = _family(e.category)
        keys = key_of(p.words)
        found, i = [], 0
        while i < len(keys):
            for size in range(min(max_len, len(keys) - i), 0, -1):
                # Tên một từ ("Nghệ", "Vũ") khớp nhầm quá nhiều: chỉ nhận tên >= 2 từ.
                if size == 1:
                    continue
                if _is_non_person(p.words[i : i + size]):
                    continue
                c = m.lookup(keys[i : i + size], family)
                if c is not None:
                    found.append(c)
                    i += size
                    break
            else:
                i += 1
        if found:
            assigned[e.entity_id] = list(dict.fromkeys(m.find(c) for c in found))
            methods[e.entity_id] = "name_contains"
            for c in assigned[e.entity_id]:
                attach(c, e, None)
        else:
            run = _longest_name_run(p.words)
            c = m.cluster_for(key_of(run), family)
            attach(c, e, " ".join(run))
            assigned[e.entity_id] = [c]
            methods[e.entity_id] = "name_fallback"

    return _build(m, entities, assigned, methods)


def _build(m: _Merger, entities, assigned, methods) -> MergeResult:
    roots: dict[int, _Cluster] = {}
    for i, cl in enumerate(m.clusters):
        r = m.find(i)
        root = roots.setdefault(r, _Cluster(m.clusters[r].family))
        root.forms.update(cl.forms)
        root.aliases |= cl.aliases
        root.categories.update(cl.categories)
        root.entity_ids += cl.entity_ids

    persons: dict[int, MergedPerson] = {}
    used_ids: set[str] = set()
    for r, cl in sorted(roots.items()):
        if not cl.entity_ids:
            continue
        name = max(cl.forms.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        category = cl.categories.most_common(1)[0][0]
        aliases = {f for f in cl.forms if f != name} | cl.aliases
        for form in [name, *cl.forms]:
            aliases.update(derived_aliases(form.split()))
        aliases.discard(name)
        pid = id_of(name.split())
        if pid in used_ids:
            pid = f"{pid}_{cl.family or 'x'}"
        n = 2
        while pid in used_ids:
            pid, n = f"{pid}_{n}", n + 1
        used_ids.add(pid)
        persons[r] = MergedPerson(pid, name, category, sorted(aliases), cl.entity_ids)

    entity_persons = {
        eid: list(dict.fromkeys(persons[m.find(c)].person_id for c in cs))
        for eid, cs in assigned.items()
    }
    return MergeResult(list(persons.values()), entity_persons, methods)
