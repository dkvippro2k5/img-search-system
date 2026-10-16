"""Định dạng trung gian giữa bước hợp nhất dữ liệu và bước dựng index.

Một thư mục gồm 5 file JSONL (mỗi dòng một bản ghi):
persons.jsonl, entities.jsonl, images.jsonl, image_entity.jsonl, relations.jsonl.
`merge.py` ghi ra định dạng này; bộ dữ liệu giả cho test cũng vậy.
Khóa ảnh là sha256. Cột lạ trong file được bỏ qua.
"""

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class PersonRec:
    person_id: str
    name: str
    category: str | None = None
    aliases: list[str] = field(default_factory=list)
    wikidata_qid: str | None = None


@dataclass
class EntityRec:
    entity_id: str
    canonical_name: str
    person_id: str | None = None
    merge_method: str | None = None  # name | face | wikidata | manual


@dataclass
class ImageRec:
    image_id: str  # sha256
    image_url: str | None = None
    thumb_path: str | None = None
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    phash: str | None = None
    caption: str | None = None  # VLM của mentor
    description: str | None = None  # tiêu đề bài báo nguồn
    source_url: str | None = None
    quality_flags: str | None = None
    nsfw_score: float | None = None
    watermark_score: float | None = None
    sensitive_flag: str | None = None
    status: str = "approved"  # approved | rejected | review
    dup_of: str | None = None  # sha256 ảnh gốc nếu là bản gần trùng (Phần A)


@dataclass
class ImageEntityRec:
    image_id: str
    entity_id: str
    human_flag: str | None = None
    reject_reason: str | None = None


@dataclass
class RelationRec:
    subj_person: str
    predicate: str  # child, spouse, parent, sibling, member_of...
    obj_person: str
    source: str  # wikidata | article | manual
    evidence: str
    evidence_url: str | None = None


@dataclass
class Dataset:
    persons: list[PersonRec] = field(default_factory=list)
    entities: list[EntityRec] = field(default_factory=list)
    images: list[ImageRec] = field(default_factory=list)
    image_entity: list[ImageEntityRec] = field(default_factory=list)
    relations: list[RelationRec] = field(default_factory=list)


_FILES = {
    "persons": PersonRec,
    "entities": EntityRec,
    "images": ImageRec,
    "image_entity": ImageEntityRec,
    "relations": RelationRec,
}


def _from_dict(cls, row: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in row.items() if k in names})


def read_jsonl_dir(path: str | Path) -> Dataset:
    path = Path(path)
    ds = Dataset()
    for name, cls in _FILES.items():
        file = path / f"{name}.jsonl"
        if not file.exists():
            continue
        with file.open(encoding="utf-8") as fh:
            rows = [_from_dict(cls, json.loads(line)) for line in fh if line.strip()]
        setattr(ds, name, rows)
    return ds


def write_jsonl_dir(ds: Dataset, path: str | Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in _FILES:
        with (path / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
            for rec in getattr(ds, name):
                fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
