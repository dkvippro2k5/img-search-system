"""Test set JSONL (part_b/testsets/README.md) và nhãn chấm tay (PLAN 5.1).

Ground truth (`gt`) của mỗi truy vấn:
- "rule": mọi ảnh approved của `expected_persons` (multi-intent: theo `intent_persons`).
- "pool": nhãn 0/1/2 do người chấm trên ảnh gộp top-20 của nhiều cấu hình
  (scripts/make_pool.py -> scripts/label_server.py -> labels.v1.jsonl). Ảnh chưa chấm = 0.
- danh sách sha256: tập ảnh đúng cố định (độ liên quan 1).
- []: truy vấn phủ định, đáp án đúng là không trả ảnh.

Multi-intent có bối cảnh dùng `intent_refs`: mỗi vế lấy ground truth của một truy vấn single
đã chấm (cùng câu chữ), khỏi chấm lại.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

QUERY_TYPES = ("single", "multi_intent", "multi_hop")
BEHAVIORS = (
    "results",
    "no_evidence",
    "empty_or_notice",
    "disambiguate",
    "disambiguate_or_no_evidence",
    "partial_with_notice",
    "no_co_occurrence",
)


@dataclass
class TestQuery:
    __test__ = False  # không phải lớp test của pytest

    id: str
    type: str
    query: str
    scenario: str = ""
    split: str = "dev"
    gt: str | list[str] = "rule"
    expected_behavior: str = "results"
    intents: list[str] = field(default_factory=list)
    expected_persons: list[str] = field(default_factory=list)
    intent_persons: list[list[str]] = field(default_factory=list)
    intent_refs: list[str] = field(default_factory=list)
    relation_path: list[str] = field(default_factory=list)
    anchor_persons: list[str] = field(default_factory=list)  # multi-hop: người gốc
    notes: str = ""

    def validate(self) -> None:
        if self.type not in QUERY_TYPES:
            raise ValueError(f"{self.id}: type '{self.type}' không hợp lệ")
        if self.split not in ("dev", "test"):
            raise ValueError(f"{self.id}: split '{self.split}' không hợp lệ")
        if self.expected_behavior not in BEHAVIORS:
            raise ValueError(f"{self.id}: expected_behavior '{self.expected_behavior}'")
        if (
            self.gt == "rule"
            and self.expected_behavior == "results"
            and not (self.expected_persons or self.intent_persons)
        ):
            raise ValueError(f"{self.id}: gt=rule cần expected_persons")
        if self.gt not in ("rule", "pool") and not isinstance(self.gt, list):
            raise ValueError(f"{self.id}: gt phải là 'rule', 'pool' hoặc danh sách sha256")

    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v not in ("", [], None) or k == "gt"}
        return json.dumps(d, ensure_ascii=False)


def load_testset(path: str | Path) -> list[TestQuery]:
    queries, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            q = TestQuery(**json.loads(line))
        except TypeError as e:
            raise ValueError(f"{path}:{n}: {e}") from e
        q.validate()
        if q.id in seen:
            raise ValueError(f"{path}:{n}: id '{q.id}' bị lặp")
        seen.add(q.id)
        queries.append(q)
    return queries


def write_testset(path: str | Path, queries: list[TestQuery]) -> None:
    for q in queries:
        q.validate()
    Path(path).write_text("".join(q.to_json() + "\n" for q in queries), encoding="utf-8")


# --- nhãn chấm tay: mỗi dòng {"qid", "image_id", "grade"}, dòng sau ghi đè dòng trước ---


def load_labels(path: str | Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.setdefault(r["qid"], {})[r["image_id"]] = int(r["grade"])
    return out


def append_labels(path: str | Path, rows: list[dict]) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        for r in rows:
            if int(r["grade"]) not in (0, 1, 2):
                raise ValueError(f"grade phải là 0/1/2: {r}")
            f.write(
                json.dumps(
                    {"qid": r["qid"], "image_id": r["image_id"], "grade": int(r["grade"])},
                    ensure_ascii=False,
                )
                + "\n"
            )
