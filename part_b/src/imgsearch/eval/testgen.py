"""Sinh truy vấn có ground truth tự động từ meta.sqlite (PLAN 5.1, testsets/README.md).

Chỉ đọc dữ liệu (person, ảnh, ảnh chung), KHÔNG gọi engine: chọn truy vấn không được thiên về
những câu hệ thống đã làm đúng. Ground truth "rule" = mọi ảnh approved của person, cộng các
person cùng tên bị tách mảnh (cùng tên gấp, cùng nhóm nghề, ví dụ "B Ray" / "BRay"): hệ thống
bỏ sót mảnh thì phải mất recall, không được che đi.

Hạn chế: person và ảnh của person đến từ entity_merge, nên lỗi gán nhãn gốc (E8) không đo được ở
đây; độ đúng của gộp entity đo riêng trên bộ golden.
"""

import random
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from imgsearch.eval.testset import TestQuery
from imgsearch.retrieval import lexicon as lx
from imgsearch.text.normalize import fold, nfc, strip_diacritics

BUCKETS = ((1, 4), (5, 19), (20, 49), (50, 10**9))
# Từ chỉ nghề đặt trước tên để phân biệt trùng tên (E3); phải có trong lexicon.CATEGORY_HINTS.
CATEGORY_WORD = {
    "singer": "ca sĩ",
    "celeb": "diễn viên",
    "athletes": "vận động viên",
    "businessman": "doanh nhân",
    "writer": "nhà văn",
    "scientist": "nhà khoa học",
    "politicer": "chính trị gia",
    "historic": "nhân vật lịch sử",
}
# Tên bịa (E6, M5); generator kiểm tra không trùng ai trong dữ liệu.
FAKE_NAMES = ("Lò Văn Pủa", "Đinh Thị Hoa Mơ", "Quàng Thị Sình", "Hà Văn Tẹo")
# Bảng person có cả tổ chức và nhãn nhóm ("REE Corporation", "HCV SEA Games"): lỗi dữ liệu, ghi
# trong báo cáo; không dùng làm truy vấn tên người.
NOT_PERSON = re.compile(
    r"\b(group|corporation|corp|milk|agris|sea games|olympic|paralympic|hcv|hcb|hcđ|forbes)\b",
    re.IGNORECASE,
)
TONES = "\u0300\u0301\u0303\u0309\u0323"  # huyền, sắc, ngã, hỏi, nặng


@dataclass
class Person:
    pid: str
    name: str
    aliases: list[str]
    category: str
    n: int


def compact(name: str) -> str:
    return fold(name).replace(" ", "").replace("-", "")


def load_people(conn: sqlite3.Connection) -> list[Person]:
    import json

    counts = dict(
        conn.execute("SELECT person_id, COUNT(*) FROM image_person GROUP BY person_id").fetchall()
    )
    return [
        Person(pid, nfc(name), json.loads(aliases or "[]"), category or "", counts.get(pid, 0))
        for pid, name, aliases, category in conn.execute(
            "SELECT person_id, name, aliases, category FROM person ORDER BY person_id"
        )
    ]


def family(category: str) -> str:
    return lx.CATEGORY_FAMILY.get(category, category)


def fragment_map(people: list[Person]) -> dict[str, list[str]]:
    """pid -> mọi pid cùng tên gấp + cùng nhóm nghề (gồm chính nó)."""
    groups: dict[tuple, list[str]] = defaultdict(list)
    for p in people:
        groups[(compact(p.name), family(p.category))].append(p.pid)
    return {pid: sorted(g) for g in groups.values() for pid in g}


def name_owners(people: list[Person]) -> dict[str, set[str]]:
    """Tên/alias gấp gọn -> các pid (để biết tên nào duy nhất)."""
    out: dict[str, set[str]] = defaultdict(set)
    for p in people:
        for n in [p.name, *p.aliases]:
            out[compact(n)].add(p.pid)
    return out


def typo(name: str, rng: random.Random) -> str | None:
    """Sai chính tả nhẹ (L2): đổi dấu thanh của một âm tiết có dấu thanh."""
    words = nfc(name).split()
    toned = [i for i, w in enumerate(words) if set(unicodedata.normalize("NFD", w)) & set(TONES)]
    if not toned:
        return None
    i = rng.choice(toned)
    nfd = unicodedata.normalize("NFD", words[i])
    old = next(c for c in nfd if c in TONES)
    new = rng.choice([t for t in TONES if t != old])
    words[i] = nfc(nfd.replace(old, new, 1))
    return " ".join(words)


def same_person_like(a: str, b: str) -> bool:
    """Tên này chứa trọn các âm tiết của tên kia ("Lâm Thu Hồng" / "Lâm Thị Thu Hồng"): nhiều
    khả năng là một người bị tách mảnh, không dùng làm cặp "chụp cùng"."""
    ta, tb = set(fold(a).split()), set(fold(b).split())
    return ta <= tb or tb <= ta or compact(a) in compact(b) or compact(b) in compact(a)


def unaccent(name: str) -> str:
    return strip_diacritics(name).lower()


def bucket_of(n: int) -> int:
    return next(i for i, (lo, hi) in enumerate(BUCKETS) if lo <= n <= hi)


class Generator:
    def __init__(self, conn: sqlite3.Connection, seed: int = 20261020):
        self.conn = conn
        self.rng = random.Random(seed)
        self.people = load_people(conn)
        self.by_pid = {p.pid: p for p in self.people}
        self.frags = fragment_map(self.people)
        self.owners = name_owners(self.people)
        self.queries: list[TestQuery] = []
        self._split = defaultdict(int)

    # --- tiện ích ---

    def unique_name(self, p: Person) -> bool:
        """Tên chỉ thuộc một người (tính cả mảnh của chính người đó)."""
        return self.owners[compact(p.name)] <= set(self.frags[p.pid])

    def gt_persons(self, pid: str) -> list[str]:
        return self.frags[pid]

    def add(self, q: TestQuery) -> None:
        """Chia dev/test xen kẽ trong từng mã kịch bản."""
        key = (q.type, q.scenario)
        q.split = "dev" if self._split[key] % 2 == 0 else "test"
        self._split[key] += 1
        self.queries.append(q)

    def pick(self, pool: list[Person], k: int, used: set[str]) -> list[Person]:
        cands = [p for p in pool if p.pid not in used]
        out = self.rng.sample(cands, min(k, len(cands)))
        used.update(p.pid for p in out)
        return out

    def eligible(self, min_n: int = 1) -> list[Person]:
        return [
            p
            for p in self.people
            if p.n >= min_n
            and self.unique_name(p)
            and 2 <= len(p.name.split()) <= 4  # dài hơn: thường là nhãn ghép hai người
            and not NOT_PERSON.search(p.name)
        ]

    # --- single ---

    def single_names(self, per_category: int = 3) -> list[Person]:
        """E1: phân tầng theo nghề x số ảnh."""
        picked, used = [], set()
        by_cat_bucket: dict[tuple, list[Person]] = defaultdict(list)
        for p in self.eligible():
            by_cat_bucket[(p.category, bucket_of(p.n))].append(p)
        for cat in sorted({p.category for p in self.people if p.category in CATEGORY_WORD}):
            buckets = [b for b in range(len(BUCKETS)) if by_cat_bucket[(cat, b)]]
            for b in self.rng.sample(buckets, min(per_category, len(buckets))):
                for p in self.pick(by_cat_bucket[(cat, b)], 1, used):
                    picked.append(p)
                    self.add(
                        TestQuery(
                            id=f"S-N-{len(picked):02d}",
                            type="single",
                            scenario="E1",
                            query=p.name,
                            expected_persons=self.gt_persons(p.pid),
                            notes=f"{p.category}, {p.n} ảnh",
                        )
                    )
        return picked

    def single_variants(self, base: list[Person], n_l1: int = 6, n_l2: int = 5) -> None:
        with_marks = [p for p in base if unaccent(p.name) != p.name.lower()]
        for k, p in enumerate(self.rng.sample(with_marks, min(n_l1, len(with_marks))), 1):
            self.add(
                TestQuery(
                    id=f"S-L1-{k:02d}",
                    type="single",
                    scenario="L1",
                    query=unaccent(p.name),
                    expected_persons=self.gt_persons(p.pid),
                )
            )
        k = 0
        for p in self.rng.sample(base, len(base)):
            wrong = typo(p.name, self.rng)
            if not wrong or len(compact(p.name)) < 6 or compact(wrong) in self.owners:
                continue
            k += 1
            self.add(
                TestQuery(
                    id=f"S-L2-{k:02d}",
                    type="single",
                    scenario="L2",
                    query=wrong,
                    expected_persons=self.gt_persons(p.pid),
                    notes=f"sai dấu của {p.name}",
                )
            )
            if k == n_l2:
                break

    def single_aliases(self, n: int = 4) -> None:
        """Không dùng trong run(): alias trong dữ liệu lẫn tên công ty ("Giao Hàng Nhanh"); L3 viết
        tay trong manual.v1.jsonl. Giữ lại để rà alias.

        L3: tên thật / tên khác (alias >= 3 từ, không phải đuôi tên chính, của một người)."""
        cands = []
        for p in self.people:
            if p.n < 3 or not self.unique_name(p):
                continue
            for a in p.aliases:
                if (
                    3 <= len(a.split()) <= 4  # tên thật đầy đủ; 2 từ hay là đuôi tên chính
                    and not fold(p.name).endswith(fold(a))
                    and not NOT_PERSON.search(a)
                    and self.owners[compact(a)] <= set(self.frags[p.pid])
                ):
                    cands.append((p, a))
        for k, (p, a) in enumerate(self.rng.sample(cands, min(n, len(cands))), 1):
            self.add(
                TestQuery(
                    id=f"S-L3-{k:02d}",
                    type="single",
                    scenario="L3",
                    query=a,
                    expected_persons=self.gt_persons(p.pid),
                    notes=f"alias của {p.name}",
                )
            )

    def homonyms(self, n_e2: int = 4, n_e3: int = 4) -> None:
        """E2: tên trùng nhiều người (khác nhóm nghề) -> chia nhóm; E3: nghề + tên."""
        groups: dict[str, list[Person]] = defaultdict(list)
        for p in self.people:
            groups[compact(p.name)].append(p)
        cands = []
        for ps in groups.values():
            fams = {family(p.category) for p in ps}
            if len(fams) >= 2 and all(p.n >= 3 for p in ps) and len(ps[0].name.split()) >= 2:
                cands.append(sorted(ps, key=lambda p: p.pid))
        chosen = self.rng.sample(cands, min(n_e2 + n_e3, len(cands)))
        for k, ps in enumerate(chosen[:n_e2], 1):
            self.add(
                TestQuery(
                    id=f"S-E2-{k:02d}",
                    type="single",
                    scenario="E2",
                    query=ps[0].name,
                    expected_persons=sorted(p.pid for p in ps),
                    expected_behavior="disambiguate",
                    notes=", ".join(f"{p.pid} ({p.category})" for p in ps),
                )
            )
        k = 0
        for ps in chosen[n_e2:]:
            by_cat = defaultdict(list)
            for p in ps:
                by_cat[p.category].append(p)
            target = self.rng.choice(ps)
            if len(by_cat[target.category]) > 1 or target.category not in CATEGORY_WORD:
                continue
            k += 1
            self.add(
                TestQuery(
                    id=f"S-E3-{k:02d}",
                    type="single",
                    scenario="E3",
                    query=f"{CATEGORY_WORD[target.category]} {target.name}",
                    expected_persons=[target.pid],
                    notes="trùng tên với " + ", ".join(p.pid for p in ps if p is not target),
                )
            )

    def unknown(self) -> list[str]:
        fakes = [n for n in FAKE_NAMES if compact(n) not in self.owners]
        for k, name in enumerate(fakes[:2], 1):
            self.add(
                TestQuery(
                    id=f"S-E6-{k:02d}",
                    type="single",
                    scenario="E6",
                    query=name,
                    gt=[],
                    expected_behavior="empty_or_notice",
                    notes="tên bịa, không có trong dữ liệu",
                )
            )
        return fakes

    # --- multi-intent ---

    def _mi(self, sid: str, scenario: str, ps: list[Person], query: str, **kw) -> None:
        self.add(
            TestQuery(
                id=sid,
                type="multi_intent",
                scenario=scenario,
                query=query,
                intents=[p.name for p in ps],
                intent_persons=[self.gt_persons(p.pid) for p in ps],
                expected_persons=sorted({x for p in ps for x in self.gt_persons(p.pid)}),
                **kw,
            )
        )

    def multi_intent(self, fakes: list[str]) -> None:
        used: set[str] = set()
        pool = self.eligible(min_n=5)
        by_cat = defaultdict(list)
        for p in pool:
            by_cat[p.category].append(p)
        cats = sorted(by_cat)
        # M1: hai người, nửa cùng nghề nửa khác nghề. Dữ liệu nhỏ (test) không đủ người thì bỏ.
        for k in range(1, 13):
            if k % 2:
                ps = self.pick(by_cat[self.rng.choice(cats)], 2, used) if cats else []
            elif len(cats) >= 2:
                c1, c2 = self.rng.sample(cats, 2)
                ps = self.pick(by_cat[c1], 1, used) + self.pick(by_cat[c2], 1, used)
            else:
                ps = []
            if len(ps) == 2:
                self._mi(f"M-P-{k:02d}", "M1", ps, f"{ps[0].name} và {ps[1].name}")
        for k in range(1, 4):
            ps = self.pick(pool, 3, used)
            if len(ps) == 3:
                query = f"{ps[0].name}, {ps[1].name} và {ps[2].name}"
                self._mi(f"M-T-{k:02d}", "M1", ps, query)
        # M6: một người rất nhiều ảnh, một người rất ít.
        big = [p for p in pool if p.n >= 100]
        small = [p for p in self.eligible(min_n=2) if p.n <= 5]
        for k in range(1, 5):
            ps = self.pick(big, 1, used) + self.pick(small, 1, used)
            if len(ps) == 2:
                a, b = ps
                self._mi(f"M-B-{k:02d}", "M6", ps, f"{a.name} và {b.name}", notes=f"{a.n}/{b.n}")
        # L1: không dấu.
        for k in range(1, 4):
            ps = self.pick(pool, 2, used)
            if len(ps) == 2:
                query = f"{unaccent(ps[0].name)} va {unaccent(ps[1].name)}"
                self._mi(f"M-L1-{k:02d}", "L1", ps, query)
        # M5: một vế là tên không có trong dữ liệu.
        for k, fake in enumerate(fakes[:3], 1):
            picked = self.pick(pool, 1, used)
            if not picked:
                break
            a = picked[0]
            self.add(
                TestQuery(
                    id=f"M-M5-{k:02d}",
                    type="multi_intent",
                    scenario="M5",
                    query=f"{a.name} và {fake}",
                    intents=[a.name, fake],
                    intent_persons=[self.gt_persons(a.pid), []],
                    expected_persons=self.gt_persons(a.pid),
                    expected_behavior="partial_with_notice",
                )
            )
        self.co_occurrence()

    def co_occurrence(self, n_with: int = 5, n_without: int = 2) -> None:
        """M2: "ảnh A chụp cùng B". gt = ảnh có cả hai (sau khi gộp mảnh)."""
        imgs: dict[str, set[str]] = defaultdict(set)
        for image_id, pid in self.conn.execute("SELECT image_id, person_id FROM image_person"):
            imgs[pid].add(image_id)

        def images_of(pid: str) -> set[str]:
            return set().union(*(imgs[x] for x in self.gt_persons(pid)))

        shared: dict[tuple, int] = defaultdict(int)
        for (image_id,) in self.conn.execute(
            "SELECT image_id FROM image_person GROUP BY image_id HAVING COUNT(*) >= 2"
        ):
            pids = sorted(
                {
                    self.frags[p][0]
                    for (p,) in self.conn.execute(
                        "SELECT person_id FROM image_person WHERE image_id = ?", (image_id,)
                    )
                }
            )
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    shared[(pids[i], pids[j])] += 1
        pairs = [
            (a, b)
            for (a, b), n in sorted(shared.items(), key=lambda x: (-x[1], x[0]))
            if n >= 3
            and self.unique_name(self.by_pid[a])
            and self.unique_name(self.by_pid[b])
            and not same_person_like(self.by_pid[a].name, self.by_pid[b].name)
            and not NOT_PERSON.search(self.by_pid[a].name + " " + self.by_pid[b].name)
        ][:n_with]
        for k, (a, b) in enumerate(pairs, 1):
            pa, pb = self.by_pid[a], self.by_pid[b]
            both = sorted(images_of(a) & images_of(b))
            self.add(
                TestQuery(
                    id=f"M-C2-{k:02d}",
                    type="multi_intent",
                    scenario="M2",
                    query=f"ảnh {pa.name} chụp cùng {pb.name}",
                    intents=[pa.name, pb.name],
                    expected_persons=sorted(set(self.gt_persons(a) + self.gt_persons(b))),
                    gt=both,
                    notes=f"{len(both)} ảnh chung",
                )
            )
        pool = self.eligible(min_n=20)
        k = tries = 0
        while k < n_without and len(pool) >= 2 and tries < 1000:
            tries += 1
            a, b = self.rng.sample(pool, 2)
            if images_of(a.pid) & images_of(b.pid):
                continue
            k += 1
            self.add(
                TestQuery(
                    id=f"M-C0-{k:02d}",
                    type="multi_intent",
                    scenario="M2",
                    query=f"ảnh {a.name} chụp cùng {b.name}",
                    intents=[a.name, b.name],
                    intent_persons=[self.gt_persons(a.pid), self.gt_persons(b.pid)],
                    expected_persons=sorted(set(self.gt_persons(a.pid) + self.gt_persons(b.pid))),
                    expected_behavior="no_co_occurrence",
                    notes="không có ảnh chung: báo rồi hiện từng người",
                )
            )

    def run(self) -> list[TestQuery]:
        base = self.single_names()
        self.single_variants(base)
        self.homonyms()
        fakes = self.unknown()
        self.multi_intent(fakes)
        return self.queries
