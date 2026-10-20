"""Test set JSONL, sinh truy vấn tự động, pooling và chấm (imgsearch/eval/testset, testgen,
pooling, run_testset) trên bộ dữ liệu giả mini."""

import json
import random
import sqlite3
import unicodedata

import pytest

from conftest import img
from imgsearch.config import Settings
from imgsearch.eval import pooling
from imgsearch.eval.run_testset import build_truth, evaluate, render, summarize
from imgsearch.eval.testgen import (
    TONES,
    Generator,
    Person,
    compact,
    fragment_map,
    same_person_like,
    typo,
)
from imgsearch.eval.testset import (
    TestQuery,
    append_labels,
    load_labels,
    load_testset,
    write_testset,
)
from imgsearch.retrieval.engine import SearchEngine


@pytest.fixture(scope="module")
def engine(mini_index) -> SearchEngine:
    out, _ = mini_index
    e = SearchEngine(Settings(_env_file=None, index_dir=out, index_version="test"))
    e.load()
    return e


# --- định dạng ---


def test_testset_roundtrip_and_validation(tmp_path):
    qs = [
        TestQuery("a", "single", "Đen Vâu", expected_persons=["den_vau"]),
        TestQuery("b", "multi_hop", "con của Mỹ Tâm", gt=[], expected_behavior="no_evidence"),
    ]
    path = tmp_path / "q.jsonl"
    write_testset(path, qs)
    back = load_testset(path)
    assert [q.id for q in back] == ["a", "b"] and back[1].gt == []
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[1])["gt"] == []

    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="lặp"):
        load_testset(path)
    with pytest.raises(ValueError, match="expected_persons"):
        TestQuery("c", "single", "x").validate()  # gt=rule mà không có person
    with pytest.raises(ValueError, match="type"):
        TestQuery("d", "nope", "x", gt=[]).validate()


def test_labels_last_write_wins(tmp_path):
    path = tmp_path / "labels.jsonl"
    append_labels(path, [{"qid": "q", "image_id": "x", "grade": 2}])
    append_labels(path, [{"qid": "q", "image_id": "x", "grade": 0}])
    append_labels(path, [{"qid": "q", "image_id": "y", "grade": 1}])
    assert load_labels(path) == {"q": {"x": 0, "y": 1}}
    with pytest.raises(ValueError):
        append_labels(path, [{"qid": "q", "image_id": "z", "grade": 3}])
    assert load_labels(tmp_path / "missing.jsonl") == {}


# --- sinh truy vấn ---


def test_typo_changes_exactly_one_tone():
    rng = random.Random(0)
    for name in ["Phan Mạnh Quỳnh", "Hồ Ngọc Hà", "Đặng Văn Lâm"]:
        wrong = typo(name, rng)
        assert wrong != name and compact(wrong) == compact(name)  # chỉ khác dấu thanh
        diff = [(a, b) for a, b in zip(name.split(), wrong.split(), strict=True) if a != b]
        assert len(diff) == 1
        assert any(c in TONES for c in unicodedata.normalize("NFD", diff[0][1]))
    assert typo("Hari Won", rng) is None  # không có dấu thanh


def test_fragments_and_same_person_like():
    people = [
        Person("p_b_ray", "B Ray", [], "singer", 5),
        Person("p_bray", "BRay", [], "singer", 3),
        Person("p_bray_writer", "B Ray", [], "writer", 2),  # trùng tên khác nghề: không phải mảnh
    ]
    frags = fragment_map(people)
    assert frags["p_b_ray"] == ["p_b_ray", "p_bray"]
    assert frags["p_bray_writer"] == ["p_bray_writer"]
    assert same_person_like("Lâm Thu Hồng", "Lâm Thị Thu Hồng")
    assert same_person_like("16Typh", "Typh")
    assert not same_person_like("Trấn Thành", "Hari Won")


def test_generator_on_mini_is_valid_and_deterministic(mini_index):
    out, _ = mini_index
    conn = sqlite3.connect(out / "vtest" / "meta.sqlite")
    qs = Generator(conn, seed=1).run()
    again = Generator(conn, seed=1).run()
    assert [q.to_json() for q in qs] == [q.to_json() for q in again]
    persons = {r[0] for r in conn.execute("SELECT person_id FROM person")}
    names = {compact(r[0]) for r in conn.execute("SELECT name FROM person")}
    assert qs
    for q in qs:
        q.validate()
        assert set(q.expected_persons) <= persons
    fakes = [q for q in qs if q.scenario == "E6"]
    assert fakes and all(compact(q.query) not in names for q in fakes)
    # chia dev/test xen kẽ trong từng mã kịch bản
    e1 = [q.split for q in qs if q.scenario == "E1"]
    assert e1[:2] == ["dev", "test"]
    # M2: Trấn Thành + Hari Won có ảnh chung tt_hw1 nhưng mini chỉ có 1 ảnh chung (< 3) -> bỏ
    assert not [q for q in qs if q.id.startswith("M-C2")]


# --- pooling ---


def test_pooling_config_restores_engine(engine):
    dense, settings = engine.dense, engine.settings
    with pooling.config(engine, "bm25"):
        assert engine.dense is None
    assert engine.dense is dense and engine.settings is settings


def test_pool_merges_configs(engine):
    if engine.dense is None:  # mini không có dense: chỉ kiểm cấu hình bm25
        with pooling.config(engine, "bm25"):
            ranked = pooling.ranked(engine, "Mỹ Tâm trên sân khấu", 20)
        assert ranked[0] == img("mt1") and len(ranked[0]) == 64
        return
    found = pooling.pool(engine, "Mỹ Tâm trên sân khấu", 20)
    assert img("mt1") in found and set(found[img("mt1")]) == set(pooling.CONFIGS)


# --- chấm ---


def test_build_truth_kinds(engine):
    labels = {"ctx": {img("mt1"): 2, img("mt2"): 0}, "ctx2": {img("den1"): 1}}

    def truth(q, lab=labels):
        return build_truth(q, engine, lab)

    rule = truth(TestQuery("r", "single", "Đen Vâu", expected_persons=["den_vau"]))
    assert set(rule.gains) == {img("den1"), img("den2"), img("den3")}

    # ảnh bị lọc an toàn không nằm trong ground truth
    hw = truth(TestQuery("h", "single", "Hari Won", expected_persons=["hari_won"]))
    assert img("hw_nsfw") not in hw.gains

    pool = truth(TestQuery("ctx", "single", "Mỹ Tâm trên sân khấu", gt="pool"))
    assert pool.gains == {img("mt1"): 2.0}
    assert truth(TestQuery("x", "single", "y", gt="pool")) is None  # chưa chấm

    refs = TestQuery(
        "m", "multi_intent", "a và b", gt="pool", intent_refs=["ctx", "ctx2"], intents=["a", "b"]
    )
    t = build_truth(refs, engine, labels)
    assert t.by_intent == [{img("mt1")}, {img("den1")}]
    assert build_truth(TestQuery("n", "multi_hop", "q", gt=[]), engine, labels) is None


def test_evaluate_scores_and_flags_errors(engine):
    qs = [
        TestQuery("S1", "single", "Đen Vâu", scenario="E1", expected_persons=["den_vau"]),
        TestQuery(
            "M1",
            "multi_intent",
            "Trấn Thành và Hari Won",
            scenario="M1",
            intents=["Trấn Thành", "Hari Won"],
            intent_persons=[["tran_thanh"], ["hari_won"]],
            expected_persons=["hari_won", "tran_thanh"],
        ),
        TestQuery(
            "H1",
            "multi_hop",
            "vợ của Trấn Thành",
            scenario="H1",
            anchor_persons=["tran_thanh"],
            relation_path=["spouse"],
            expected_persons=["hari_won"],
        ),
        TestQuery(
            "H3",
            "multi_hop",
            "con của Mỹ Tâm",
            scenario="H3",
            anchor_persons=["my_tam"],
            gt=[],
            expected_behavior="no_evidence",
        ),
        # Sai người có chủ đích: kỳ vọng Hari Won cho truy vấn "Đen Vâu"
        TestQuery("BAD", "single", "Đen Vâu", scenario="E1", expected_persons=["hari_won"]),
    ]
    rs = {r["id"]: r for r in evaluate(engine, qs, {})}
    assert rs["S1"]["p_r"] == 1.0 and rs["S1"]["r50"] == 1.0 and rs["S1"]["error"] is None
    assert rs["M1"]["intent_cov10"] == 1.0 and rs["M1"]["link_ok"]
    assert rs["H1"]["anchor_ok"] and rs["H1"]["target_ok"] and rs["H1"]["p_r"] == 1.0
    assert rs["H3"]["behavior_ok"] and not rs["H3"]["hallucinated"]
    assert rs["BAD"]["error"] == "liên kết người sai" and rs["BAD"]["p10"] == 0.0

    summ = summarize(list(rs.values()))
    assert summ["by_type"]["multi_hop"]["hallucinated"] == "0/2"
    meta = {
        "testset": "t",
        "split": "dev",
        "index_version": "test",
        "when": "now",
        "n_images": engine.n_images,
        "n_persons": 0,
        "pool_total": 0,
        "pool_judged": 0,
    }
    md = render(meta, {"bm25": list(rs.values())}, "bm25")
    assert "## Multi-hop" in md and "liên kết người sai" in md and "| BAD |" in md
