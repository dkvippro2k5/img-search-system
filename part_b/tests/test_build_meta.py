import json
import sqlite3

from conftest import img
from imgsearch.data.build_meta import build_meta
from imgsearch.data.dataset import (
    Dataset,
    PersonRec,
    RelationRec,
    read_jsonl_dir,
    write_jsonl_dir,
)


def test_manifest_counts(mini_index):
    out, manifest = mini_index
    s = manifest["stats"]
    assert s["persons"] == 12
    assert s["images_indexed"] == 26
    assert s["images_skipped_not_approved"] == 2  # den4 rejected, rev1 review
    assert s["images_skipped_duplicate"] == 1  # hnh3
    assert s["image_entity_skipped_image_not_indexed"] == 3
    assert s["relations"] == 2
    assert s["relations_skipped_unknown_person"] == 1
    assert s["entities_without_person"] == 1
    assert s["images_without_person"] == 1  # grp1
    on_disk = json.loads((out / "vtest" / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["files"]["meta.sqlite"]["sha256"] == manifest["files"]["meta.sqlite"]["sha256"]


def test_only_approved_non_duplicate_images_indexed(mini_store):
    ids = set(mini_store.get_images([img(k) for k in ("den1", "den4", "hnh1", "hnh3", "rev1")]))
    assert ids == {img("den1"), img("hnh1")}


def test_vec_row_is_dense_and_unique(mini_store):
    rows = [r[0] for r in mini_store.conn.execute("SELECT vec_row FROM image ORDER BY vec_row")]
    assert rows == list(range(26))


def test_E4_fragmented_entities_map_to_one_person(mini_store):
    assert sorted(mini_store.images_of(["pham_nhat_vuong"])) == sorted(
        [img("pnv1"), img("pnv2"), img("pnv3")]
    )


def test_E7_image_in_two_entities_of_same_person_counted_once(mini_store):
    assert mini_store.persons_of([img("pnv3")]) == {img("pnv3"): ["pham_nhat_vuong"]}


def test_M2_group_photo_has_both_persons(mini_store):
    assert sorted(mini_store.persons_of([img("tt_hw1")])[img("tt_hw1")]) == [
        "hari_won",
        "tran_thanh",
    ]


def test_persons_keep_aliases(mini_store):
    by_id = {p.person_id: p for p in mini_store.persons()}
    assert by_id["son_tung_mtp"].aliases == ("Sơn Tùng", "Sếp Tùng")
    assert by_id["son_tung_writer"].category == "writer"


def test_relation_without_evidence_is_dropped(tmp_path):
    ds = Dataset(
        persons=[PersonRec("a", "A"), PersonRec("b", "B")],
        relations=[
            RelationRec("a", "spouse", "b", "manual", ""),
            RelationRec("a", "spouse", "b", "", "có bằng chứng nhưng thiếu nguồn"),
            RelationRec("a", "child", "b", "manual", "câu bằng chứng", None),
        ],
    )
    stats = build_meta(ds, tmp_path / "meta.sqlite")
    assert stats["relations_skipped_no_evidence"] == 2
    assert stats["relations"] == 1
    conn = sqlite3.connect(tmp_path / "meta.sqlite")
    assert conn.execute("SELECT predicate FROM relation").fetchall() == [("child",)]
    conn.close()


def test_rebuild_overwrites_without_leftovers(tmp_path):
    ds = Dataset(persons=[PersonRec("a", "A")])
    build_meta(ds, tmp_path / "meta.sqlite")
    build_meta(ds, tmp_path / "meta.sqlite")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meta.sqlite"]


def test_jsonl_roundtrip(tmp_path):
    from conftest import MINI

    ds = read_jsonl_dir(MINI)
    write_jsonl_dir(ds, tmp_path)
    again = read_jsonl_dir(tmp_path)
    assert again == ds
