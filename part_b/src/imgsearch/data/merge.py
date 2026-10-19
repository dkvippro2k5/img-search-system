"""Hợp nhất CSV ảnh người đã duyệt (data/raw/people-*.csv) -> Dataset (định dạng ở dataset.py).

Khóa ảnh là sha256; nối ảnh với entity theo (entity_id, sha256). Các bước, mỗi bước đếm vào
report để kiểm tra lệch cột / mất dữ liệu:

1. Đọc CSV, kiểm tra đủ cột bắt buộc.
2. Dòng thiếu sha256 (ảnh chưa tải về): lấy sha256 của ảnh cùng `image_url` đã có, hoặc từ nhật
   ký tải (data/fetch.py). Không có thì BỎ (không dùng hash URL: ảnh chưa qua kiểm duyệt Phần A
   thì không được hiện, Q4).
3. Nghề trống: lấy nghề phổ biến nhất trong các dòng khác của cùng entity.
4. Làm giàu văn bản khi chưa có caption VLM (`keywords`, chỉ dùng cho BM25 / dense):
   cột `query` của crawler ("chân dung Á hậu Thanh Tú") và slug của `source_url` khi không có
   tiêu đề bài (slug không dấu, chỉ khớp truy vấn gõ không dấu).
5. Gộp entity -> person (entity_merge), trích quan hệ từ tiêu đề (kg.article_relations).
"""

import csv
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import unquote, urlparse

from imgsearch.data.dataset import Dataset, EntityRec, ImageEntityRec, ImageRec, PersonRec
from imgsearch.data.entity_merge import EntityInput, merge_entities
from imgsearch.kg.article_relations import extract_relations
from imgsearch.retrieval.aliases import AliasIndex, PersonEntry
from imgsearch.retrieval.query_parser import QueryParser

RAW_GLOB = "people-*.csv"
REQUIRED_COLUMNS = {
    "entity_id",
    "canonical_name",
    "category",
    "sha256",
    "image_url",
    "source_url",
    "description",
    "flag",
}
_SLUG_EXT = re.compile(r"\.(html?|chn|ldo|aspx?|php|vnp|tpo|vov)$", re.IGNORECASE)
_SLUG_MIN_WORDS = 3


def read_raw(paths: Iterable[str | Path]) -> list[dict]:
    csv.field_size_limit(1 << 30)
    rows: list[dict] = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
            rows += [{k: (v or "").strip() for k, v in r.items() if k} for r in reader]
    return rows


def raw_files(raw_dir: str | Path, pattern: str = RAW_GLOB) -> list[Path]:
    return sorted(Path(raw_dir).glob(pattern))


def slug_words(url: str) -> str:
    """'…/rapper-16-typh-sap-ve-chung-nha-voi-thay-binz-381.html' -> 'rapper typh sap ve …'.
    Bỏ số và mã bài; slug quá ngắn (mã, 'index') trả rỗng."""
    path = unquote(urlparse(url).path).rstrip("/")
    segment = _SLUG_EXT.sub("", path.rsplit("/", 1)[-1])
    words = [w for w in re.split(r"[-_+.]", segment.lower()) if w.isalpha() and len(w) >= 2]
    return " ".join(words) if len(words) >= _SLUG_MIN_WORDS else ""


def _int(v: str) -> int | None:
    try:
        return int(float(v))
    except ValueError:
        return None


def resolve_sha(rows: list[dict], fetched: dict[str, dict], report: Counter) -> list[dict]:
    """Điền sha256 cho dòng thiếu (bước 2); trả các dòng giữ lại."""
    url_sha = {r["image_url"]: r["sha256"] for r in rows if r["sha256"] and r["image_url"]}
    kept = []
    for r in rows:
        if r["sha256"]:
            kept.append(r)
            continue
        report["rows_missing_sha"] += 1
        if r["image_url"] in url_sha:
            r = {**r, "sha256": url_sha[r["image_url"]]}
            report["sha_from_same_url"] += 1
        elif (f := fetched.get(r["image_url"])) and f.get("status") == "ok":
            r = {
                **r,
                "sha256": f["sha256"],
                "width": str(f.get("width") or ""),
                "height": str(f.get("height") or ""),
                "bytes": str(f.get("bytes") or ""),
            }
            report["sha_from_fetch"] += 1
        else:
            report["rows_dropped_no_sha"] += 1
            continue
        kept.append(r)
    return kept


def entity_categories(rows: list[dict]) -> dict[str, str | None]:
    """Nghề của entity = nghề không trống phổ biến nhất trong các dòng của nó (bước 3)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        counts[r["entity_id"]][r["category"]] += 1
    out = {}
    for eid, c in counts.items():
        filled = [(k, n) for k, n in c.most_common() if k]
        out[eid] = filled[0][0] if filled else None
    return out


def attach_persons(
    ds: Dataset,
    entities: dict[str, EntityInput],
    links: Iterable[tuple[str, str, str | None, str | None]],
    id_prefix: str = "",
) -> dict:
    """Gộp entity bằng tên, ghi person, entity, image_entity vào ds.
    links: (sha256, entity_id, human_flag, reject_reason). Entity nhắc nhiều người (ảnh nhóm)
    được tách thành entity con `{eid}#{i}`, mỗi entity con một person."""
    result = merge_entities(list(entities.values()))
    ds.persons += [
        PersonRec(id_prefix + p.person_id, p.name, p.category, p.aliases) for p in result.persons
    ]
    sub_ids: dict[str, list[str]] = {}
    for eid, pids in result.entity_persons.items():
        pids = [id_prefix + pid for pid in pids]
        name = entities[eid].name
        if len(pids) == 1:
            ds.entities.append(EntityRec(eid, name, pids[0], result.methods[eid]))
            sub_ids[eid] = [eid]
        else:
            sub_ids[eid] = [f"{eid}#{i}" for i in range(len(pids))]
            ds.entities += [
                EntityRec(sid, name, pid, result.methods[eid])
                for sid, pid in zip(sub_ids[eid], pids, strict=True)
            ]
    for sha, eid, flag, reason in links:
        ds.image_entity += [ImageEntityRec(sha, sid, flag, reason) for sid in sub_ids[eid]]
    return {
        "entities": len(entities),
        "merge_methods": dict(Counter(result.methods.values())),
        "multi_person_entities": sum(len(p) > 1 for p in result.entity_persons.values()),
    }


def attach_title_relations(ds: Dataset, titles: Iterable[tuple[str, str | None]]) -> int:
    parser = QueryParser(
        AliasIndex(
            PersonEntry(p.person_id, p.name, p.category, tuple(p.aliases)) for p in ds.persons
        )
    )
    ds.relations = extract_relations(titles, parser)
    return len(ds.relations)


def _keywords(rows: list[dict]) -> str | None:
    parts = []
    for r in rows:
        parts.append(r.get("query") or "")
        if not r["description"]:
            parts.append(slug_words(r["source_url"]))
    text = " ".join(dict.fromkeys(p for p in parts if p))
    return text or None


def merge(rows: list[dict], fetched: dict[str, dict] | None = None) -> tuple[Dataset, dict]:
    report: Counter = Counter(rows_read=len(rows))
    rows = resolve_sha(rows, fetched or {}, report)

    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r)
    report["images"] = len(by_sha)

    ds = Dataset()
    for sha, rs in sorted(by_sha.items()):
        first = rs[0]
        ds.images.append(
            ImageRec(
                image_id=sha,
                image_url=first["image_url"] or None,
                width=_int(first.get("width", "")),
                height=_int(first.get("height", "")),
                bytes=_int(first.get("bytes", "")),
                description=next((r["description"] for r in rs if r["description"]), None),
                source_url=first["source_url"] or None,
                keywords=_keywords(rs),
                # Một ảnh bị loại ở bất kỳ entity nào thì không index (an toàn hơn).
                status="approved" if all(r["flag"] == "approved" for r in rs) else "rejected",
            )
        )
    report["images_with_keywords"] = sum(i.keywords is not None for i in ds.images)
    report["images_without_description"] = sum(i.description is None for i in ds.images)

    categories = entity_categories(rows)
    n_images = Counter(r["entity_id"] for r in rows)
    names = {r["entity_id"]: r["canonical_name"] for r in rows}
    entities = {
        eid: EntityInput(eid, names[eid], categories[eid], n) for eid, n in n_images.items()
    }
    report["rows_category_filled"] = sum(
        1 for r in rows if not r["category"] and categories[r["entity_id"]]
    )
    links = dict.fromkeys(
        (r["sha256"], r["entity_id"], r["flag"], r.get("reject_reason") or None) for r in rows
    )
    report["image_entity_links"] = len(links)
    for k, v in attach_persons(ds, entities, links).items():  # Counter.update sẽ cộng dồn
        report[k] = v

    titles = dict.fromkeys((r["description"], r["source_url"] or None) for r in rows)
    report["relations"] = attach_title_relations(ds, [t for t in titles if t[0]])
    report["persons"] = len(ds.persons)
    return ds, dict(report)
