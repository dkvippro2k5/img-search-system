"""Chuyển bộ golden (data/golden/, xem README.md của nó) sang Dataset để dựng index.

Hai chế độ:
- "auto": đúng như hệ thống thật, CHỈ đọc metadata.json. Gộp entity bằng entity_merge,
  quan hệ trích từ tiêu đề bằng kg.article_relations.
- "oracle": chẩn đoán, lấy person và quan hệ từ gold/ (đáp án) để tách lỗi gộp entity / trích
  quan hệ khỏi lỗi truy xuất. Không bao giờ dùng số của chế độ này như kết quả hệ thống.
  Ảnh đôi được gold/persons.json ghi cho cả hai người, nên oracle cũng có "ảnh có cả hai".

Không dùng trường `set` (golden/distractor) và mọi thứ trong gold/ ở chế độ auto.
"""

import json
from collections import Counter
from pathlib import Path

from imgsearch.data.dataset import (
    Dataset,
    EntityRec,
    ImageEntityRec,
    ImageRec,
    PersonRec,
    RelationRec,
)
from imgsearch.data.entity_merge import EntityInput
from imgsearch.data.merge import attach_persons, attach_title_relations

MODES = ("auto", "oracle")


def load_metadata(golden_dir: str | Path) -> list[dict]:
    return json.loads((Path(golden_dir) / "metadata.json").read_text(encoding="utf-8"))


def _caption(row: dict) -> str | None:
    """Caption VLM tiếng Việt + tags. Ảnh nhiễu không có caption (khác dữ liệu thật)."""
    parts = [row.get("caption_vi") or "", " ".join(row.get("tags") or [])]
    text = " ".join(p for p in parts if p).strip()
    return text or None


def _images(rows: list[dict], with_caption: bool = True) -> list[ImageRec]:
    return [
        ImageRec(
            image_id=r["sha256"],
            image_url=r.get("image_url"),
            width=r.get("width"),
            height=r.get("height"),
            caption=_caption(r) if with_caption else None,
            description=r.get("source_title"),
            source_url=r.get("source_url"),
            status=r.get("review_status") or "approved",
        )
        for r in rows
    ]


def to_dataset(
    golden_dir: str | Path, mode: str = "auto", with_caption: bool = True
) -> tuple[Dataset, dict]:
    """with_caption=False: bỏ caption VLM, mô phỏng dữ liệu thật hiện tại (CSV không có caption)."""
    if mode not in MODES:
        raise ValueError(f"mode phải là một trong {MODES}")
    rows = load_metadata(golden_dir)
    ds = Dataset(images=_images(rows, with_caption))
    gold_dir = Path(golden_dir) / "gold"
    report = _auto(rows, ds) if mode == "auto" else _oracle(rows, ds, gold_dir)
    report.update(mode=mode, images=len(ds.images), persons=len(ds.persons))
    return ds, report


def _merge_into(rows: list[dict], ds: Dataset, id_prefix: str = "") -> dict:
    """Gộp entity của `rows` bằng tên và ghi person, entity, image_entity vào ds."""
    n_images = Counter(r["entity_id"] for r in rows)
    first = {r["entity_id"]: r for r in rows}
    entities = {
        eid: EntityInput(eid, first[eid]["entity_name"], first[eid].get("category"), n)
        for eid, n in n_images.items()
    }
    links = [
        (r["sha256"], r["entity_id"], r.get("review_status"), r.get("reject_reason")) for r in rows
    ]
    return attach_persons(ds, entities, links, id_prefix)


def _auto(rows: list[dict], ds: Dataset) -> dict:
    report = _merge_into(rows, ds)
    titles = list(dict.fromkeys((r["source_title"], r.get("source_url")) for r in rows))
    report["relations"] = attach_title_relations(ds, titles)
    return report


def _oracle(rows: list[dict], ds: Dataset, gold_dir: Path) -> dict:
    gold_persons = json.loads((gold_dir / "persons.json").read_text(encoding="utf-8"))
    by_short = {r["image_id"]: r for r in rows}
    covered = set()
    for p in gold_persons:
        imgs = [by_short[i] for i in p["image_ids"] if i in by_short]
        category = Counter(r.get("category") for r in imgs).most_common(1)
        eid = f"gold:{p['person_id']}"
        ds.persons.append(
            PersonRec(p["person_id"], p["name"], category[0][0] if category else None, p["aliases"])
        )
        ds.entities.append(EntityRec(eid, p["name"], p["person_id"], "manual"))
        ds.image_entity += [ImageEntityRec(r["sha256"], eid, r.get("review_status")) for r in imgs]
        covered |= {r["image_id"] for r in imgs}
    # Ảnh ngoài gold (nhiễu, ảnh bị loại): gộp bằng tên như chế độ auto.
    report = _merge_into([r for r in rows if r["image_id"] not in covered], ds, id_prefix="name:")

    gold_rel = json.loads((gold_dir / "relations.json").read_text(encoding="utf-8"))
    for r in gold_rel["relations"]:
        evidence = " | ".join(e["text"] for e in r["evidence"])
        url = next((e["url"] for e in r["evidence"] if e.get("url")), None)
        source = f"gold:{r['status']}"
        if r["predicate"] == "child_of":  # gold: subject là con của object
            ds.relations.append(
                RelationRec(r["object"], "child", r["subject"], source, evidence, url)
            )
        elif r["predicate"] == "spouse_of":
            ds.relations.append(
                RelationRec(r["subject"], "spouse", r["object"], source, evidence, url)
            )
        # chairman_of (person -> tổ chức) chưa có trong schema: bỏ (H5 chưa hỗ trợ).
    report["relations"] = len(ds.relations)
    report["gold_persons"] = len(gold_persons)
    return report
