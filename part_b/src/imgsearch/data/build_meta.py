"""Dựng thư mục index: meta.sqlite (metadata, image_person, quan hệ có bằng chứng)
và lexical.npz (BM25, xem retrieval/bm25.py).

Quy tắc lọc (đếm từng lý do vào stats để ghi manifest):
- ảnh: chỉ status == "approved" và không có dup_of;
- image_entity: bỏ nếu ảnh hoặc entity không tồn tại / không vào index;
- relation: bỏ nếu thiếu source/evidence hoặc một đầu không phải person đã biết.
"""

import hashlib
import json
import sqlite3
import time
from collections import Counter
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

from imgsearch.data.dataset import Dataset, read_jsonl_dir
from imgsearch.retrieval.bm25 import BM25Index, doc_terms


def _schema_sql() -> str:
    return resources.files("imgsearch.data").joinpath("schema.sql").read_text(encoding="utf-8")


def build_meta(ds: Dataset, out_path: str | Path) -> dict:
    """Ghi ra file tạm rồi đổi tên, để không bao giờ để lại một meta.sqlite dở dang."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)

    stats: Counter = Counter()
    conn = sqlite3.connect(tmp)
    try:
        conn.execute("PRAGMA journal_mode = OFF")
        conn.execute("PRAGMA synchronous = OFF")
        conn.executescript(_schema_sql())
        with conn:
            _insert_all(conn, ds, stats)
        conn.execute("ANALYZE")
        conn.commit()
    finally:
        conn.close()
    tmp.replace(out_path)
    return dict(stats)


def _insert_all(conn: sqlite3.Connection, ds: Dataset, stats: Counter) -> None:
    persons = {p.person_id: p for p in ds.persons}
    conn.executemany(
        "INSERT INTO person VALUES (?, ?, ?, ?, ?)",
        [
            (
                p.person_id,
                p.name,
                json.dumps(p.aliases, ensure_ascii=False),
                p.category,
                p.wikidata_qid,
            )
            for p in persons.values()
        ],
    )
    stats["persons"] = len(persons)

    entities = {e.entity_id: e for e in ds.entities}
    conn.executemany(
        "INSERT INTO entity VALUES (?, ?, ?, ?)",
        [
            (
                e.entity_id,
                e.canonical_name,
                e.person_id if e.person_id in persons else None,
                e.merge_method,
            )
            for e in entities.values()
        ],
    )
    stats["entities"] = len(entities)
    stats["entities_without_person"] = sum(e.person_id not in persons for e in entities.values())

    # Ảnh vào index, sắp theo image_id để vec_row ổn định giữa các lần build.
    kept = []
    for img in ds.images:
        if img.status != "approved":
            stats["images_skipped_not_approved"] += 1
        elif img.dup_of:
            stats["images_skipped_duplicate"] += 1
        else:
            kept.append(img)
    kept = list({img.image_id: img for img in kept}.values())
    kept.sort(key=lambda img: img.image_id)
    conn.executemany(
        "INSERT INTO image VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                i.image_id,
                i.image_url,
                i.thumb_path,
                i.width,
                i.height,
                i.bytes,
                i.phash,
                i.caption,
                i.description,
                i.source_url,
                i.quality_flags,
                i.nsfw_score,
                i.watermark_score,
                i.sensitive_flag,
                i.status,
                row,
            )
            for row, i in enumerate(kept)
        ],
    )
    stats["images_indexed"] = len(kept)
    kept_ids = {i.image_id for i in kept}

    links = []
    for link in ds.image_entity:
        if link.image_id not in kept_ids:
            stats["image_entity_skipped_image_not_indexed"] += 1
        elif link.entity_id not in entities:
            stats["image_entity_skipped_unknown_entity"] += 1
        else:
            links.append(link)
    conn.executemany(
        "INSERT OR IGNORE INTO image_entity VALUES (?, ?, ?, ?)",
        [(x.image_id, x.entity_id, x.human_flag, x.reject_reason) for x in links],
    )
    conn.execute(
        """INSERT OR IGNORE INTO image_person (image_id, person_id)
           SELECT ie.image_id, e.person_id FROM image_entity ie
           JOIN entity e ON e.entity_id = ie.entity_id
           WHERE e.person_id IS NOT NULL"""
    )
    stats["images_without_person"] = conn.execute(
        "SELECT COUNT(*) FROM image WHERE image_id NOT IN (SELECT image_id FROM image_person)"
    ).fetchone()[0]

    relations = []
    for r in ds.relations:
        if not (r.source and r.source.strip() and r.evidence and r.evidence.strip()):
            stats["relations_skipped_no_evidence"] += 1
        elif r.subj_person not in persons or r.obj_person not in persons:
            stats["relations_skipped_unknown_person"] += 1
        else:
            relations.append(r)
    conn.executemany(
        "INSERT INTO relation VALUES (?, ?, ?, ?, ?, ?)",
        [
            (r.subj_person, r.predicate, r.obj_person, r.source, r.evidence, r.evidence_url)
            for r in relations
        ],
    )
    stats["relations"] = len(relations)


def lexical_texts(meta_path: str | Path) -> list[tuple[str, str, str]]:
    """(tên + biệt danh person, tiêu đề bài báo + tên entity, caption) theo thứ tự vec_row."""
    conn = sqlite3.connect(meta_path)
    try:
        names: dict[str, list[str]] = {}
        for image_id, name, aliases in conn.execute(
            "SELECT ip.image_id, p.name, p.aliases FROM image_person ip "
            "JOIN person p USING (person_id)"
        ):
            names.setdefault(image_id, []).extend([name, *json.loads(aliases or "[]")])
        entity_names: dict[str, list[str]] = {}
        for image_id, canonical in conn.execute(
            "SELECT ie.image_id, e.canonical_name FROM image_entity ie "
            "JOIN entity e USING (entity_id)"
        ):
            entity_names.setdefault(image_id, []).append(canonical)
        return [
            (
                " ".join(dict.fromkeys(names.get(image_id, []))),
                " ".join([description or "", *entity_names.get(image_id, [])]),
                caption or "",
            )
            for image_id, description, caption in conn.execute(
                "SELECT image_id, description, caption FROM image ORDER BY vec_row"
            )
        ]
    finally:
        conn.close()


def build_lexical(meta_path: str | Path, out_path: str | Path) -> int:
    texts = lexical_texts(meta_path)
    BM25Index.build(doc_terms(*t) for t in texts).save(out_path)
    return len(texts)


def _file_info(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def build_index(input_dir: str | Path, out_dir: str | Path, version: str) -> dict:
    """Dựng thư mục index/v{version}/ từ thư mục JSONL; trả manifest (đồng thời ghi ra file).

    TODO (bước dense): thêm text.faiss vào cùng thư mục và manifest.
    """
    t0 = time.perf_counter()
    target = Path(out_dir) / f"v{version}"
    meta_path = target / "meta.sqlite"
    lexical_path = target / "lexical.npz"
    stats = build_meta(read_jsonl_dir(input_dir), meta_path)
    build_lexical(meta_path, lexical_path)

    manifest = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_dir": str(Path(input_dir).resolve()),
        "build_seconds": round(time.perf_counter() - t0, 3),
        "stats": stats,
        "files": {p.name: _file_info(p) for p in (meta_path, lexical_path)},
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
