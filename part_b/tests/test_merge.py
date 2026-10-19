"""Hợp nhất CSV ảnh người -> Dataset (data/merge.py), trên CSV giả trong thư mục tạm."""

import csv

import pytest

from imgsearch.data.build_meta import build_index
from imgsearch.data.dataset import write_jsonl_dir
from imgsearch.data.merge import merge, read_raw, slug_words
from imgsearch.retrieval.lexical import LexicalIndex
from imgsearch.retrieval.store import MetaStore

COLUMNS = [
    "entity_id",
    "slug",
    "canonical_name",
    "category",
    "query",
    "sha256",
    "image_url",
    "source_url",
    "description",
    "width",
    "height",
    "flag",
    "reject_reason",
]


def raw(eid, name, sha, url, cat="singer", desc="", query="", src="", flag="approved"):
    return {
        "entity_id": eid,
        "slug": "",
        "canonical_name": name,
        "category": cat,
        "query": query,
        "sha256": sha,
        "image_url": url,
        "source_url": src,
        "description": desc,
        "width": "800",
        "height": "600",
        "flag": flag,
        "reject_reason": "",
    }


SHA = {k: k * 64 for k in "abcdef"}
ROWS = [
    raw("e1", "Hari Won", SHA["a"], "u/a", desc="Hari Won hát", query="ca sĩ Hari Won"),
    raw("e1", "Hari Won", SHA["b"], "u/b", cat="", src="https://x.vn/hari-won-di-dien-vong.html"),
    raw("e2", "Ca sĩ Hari Won (Hàn Quốc)", SHA["c"], "u/c", cat="celeb"),
    raw("e3", "Trấn Thành", SHA["d"], "u/d", cat="celeb"),
    raw("e4", "Trấn Thành & Hari Won cặp đôi", SHA["e"], "u/e", cat="celeb"),
    # thiếu sha256: cùng URL với ảnh đã có -> lấy sha đó (E7: ảnh ở hai entity)
    raw("e3", "Trấn Thành", "", "u/a", cat="celeb"),
    # thiếu sha256, đã tải được (nhật ký fetch)
    raw("e3", "Trấn Thành", "", "u/new", "celeb", desc="Vợ chồng Trấn Thành – Hari Won đi chơi"),
    # thiếu sha256, chưa tải / tải lỗi -> bỏ
    raw("e3", "Trấn Thành", "", "u/dead", cat="celeb"),
    raw("e3", "Trấn Thành", "", "u/404", cat="celeb"),
]
FETCHED = {
    "u/new": {"url": "u/new", "status": "ok", "sha256": SHA["f"], "width": 10, "height": 20},
    "u/404": {"url": "u/404", "status": "http_404"},
}


def test_read_raw_checks_columns(tmp_path):
    good = tmp_path / "people-1.csv"
    with good.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(ROWS[:2])
    assert [r["sha256"] for r in read_raw([good])] == [SHA["a"], SHA["b"]]
    bad = tmp_path / "people-2.csv"
    bad.write_text("entity_id,sha256\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu cột"):
        read_raw([bad])


def test_missing_sha_resolution():
    ds, report = merge([dict(r) for r in ROWS], FETCHED)
    assert report["rows_missing_sha"] == 4
    assert report["sha_from_same_url"] == 1 and report["sha_from_fetch"] == 1
    assert report["rows_dropped_no_sha"] == 2
    assert {i.image_id for i in ds.images} == set(SHA.values())
    new = next(i for i in ds.images if i.image_id == SHA["f"])
    assert (new.width, new.height) == (10, 20)


def test_E7_same_image_two_entities_one_image_record():
    ds, _ = merge([dict(r) for r in ROWS], FETCHED)
    assert sum(i.image_id == SHA["a"] for i in ds.images) == 1
    assert {x.entity_id for x in ds.image_entity if x.image_id == SHA["a"]} == {"e1", "e3"}


def test_entities_merge_to_persons_and_category_filled():
    ds, report = merge([dict(r) for r in ROWS], FETCHED)
    names = {p.person_id: p.name for p in ds.persons}
    person_of = {e.entity_id: names[e.person_id] for e in ds.entities}
    assert person_of["e1"] == person_of["e2"] == "Hari Won"  # E4 + singer/celeb cùng nhóm
    assert report["rows_category_filled"] == 1
    couple = {person_of[e] for e in person_of if e.startswith("e4#")}
    assert couple == {"Hari Won", "Trấn Thành"}


def test_relations_from_titles():
    ds, report = merge([dict(r) for r in ROWS], FETCHED)
    names = {p.person_id: p.name for p in ds.persons}
    assert [(r.predicate, {names[r.subj_person], names[r.obj_person]}) for r in ds.relations] == [
        ("spouse", {"Trấn Thành", "Hari Won"})
    ]
    assert report["relations"] == 1


def test_keywords_query_and_slug_only_without_title():
    ds, _ = merge([dict(r) for r in ROWS], FETCHED)
    kw = {i.image_id: i.keywords for i in ds.images}
    assert kw[SHA["a"]] == "ca sĩ Hari Won"  # có tiêu đề: không thêm slug
    assert kw[SHA["b"]] == "hari won di dien vong"
    assert kw[SHA["c"]] is None


def test_rejected_anywhere_is_not_approved():
    rows = [raw("e1", "Hari Won", SHA["a"], "u/a"), raw("e2", "Mỹ Tâm", SHA["a"], "u/a", flag="x")]
    ds, _ = merge(rows)
    assert ds.images[0].status == "rejected"


@pytest.mark.parametrize(
    ("url", "words"),
    [
        (
            "https://theinfluencer.vn/rapper-16-typh-sap-ve-chung-nha-voi-thay-binz-381.html",
            "rapper typh sap ve chung nha voi thay binz",
        ),
        ("https://kenh14.vn/son-tung-m-tp-ra-mat-mv-215231003.chn", "son tung tp ra mat mv"),
        ("https://x.vn/index.html", ""),
        ("https://x.vn/p/123456", ""),
        ("", ""),
    ],
)
def test_slug_words(url, words):
    assert slug_words(url) == words


def test_keywords_are_searchable(tmp_path):
    ds, _ = merge([dict(r) for r in ROWS], FETCHED)
    write_jsonl_dir(ds, tmp_path / "jsonl")
    build_index(tmp_path / "jsonl", tmp_path / "index", "m")
    store = MetaStore(tmp_path / "index" / "vm" / "meta.sqlite")
    lex = LexicalIndex.load(tmp_path / "index" / "vm" / "lexical.npz", store)
    assert [i for i, _ in lex.search("di dien vong")][0] == SHA["b"]
