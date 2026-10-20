"""Bộ chuyển golden -> Dataset và khung chấm, trên một bộ golden tí hon dựng trong thư mục tạm
(bộ thật ở data/golden không có trong repo)."""

import hashlib
import json

import pytest

from imgsearch.api.schemas import EvidenceStep, ImageHit, ResultGroup, SearchResponse
from imgsearch.config import Settings
from imgsearch.data.build_meta import build_index
from imgsearch.data.dataset import write_jsonl_dir
from imgsearch.data.golden import to_dataset
from imgsearch.eval.run_eval import (
    GoldQuery,
    evaluate,
    flatten,
    judge_behavior,
    merge_report,
    score_query,
    summarize,
)
from imgsearch.retrieval.engine import SearchEngine


def sha(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def row(key, entity_id, entity_name, category, title, caption=None, status="approved"):
    s = sha(key)
    return {
        "image_id": s[:16],
        "sha256": s,
        "set": "golden",
        "entity_id": entity_id,
        "entity_name": entity_name,
        "category": category,
        "source_title": title,
        "source_url": f"https://example.com/{key}",
        "image_url": f"https://example.com/{key}.jpg",
        "width": 100,
        "height": 100,
        "review_status": status,
        "reject_reason": None if status == "approved" else "wrong_person",
        "caption_vi": caption,
        "tags": ["sân khấu"] if caption else None,
    }


ROWS = [
    row("tt1", "e_tt", "Trấn Thành (MC)", "celeb", "Trấn Thành dẫn gameshow", "Nam MC cầm micro"),
    row("hw1", "e_hw", "Hari Won", "singer", "Hari Won hát", "Nữ ca sĩ trên sân khấu"),
    row(
        "both",
        "e_both",
        "Trấn Thành & Hari Won cặp đôi",
        "celeb",
        "Vợ chồng Trấn Thành – Hari Won tung bộ ảnh",
    ),
    row("bad", "e_hw", "Hari Won", "singer", "Hari Won", status="rejected"),
]


@pytest.fixture()
def golden_dir(tmp_path):
    d = tmp_path / "golden"
    d.mkdir()
    (d / "metadata.json").write_text(json.dumps(ROWS, ensure_ascii=False), encoding="utf-8")
    return d


def add_gold(d):
    g = d / "gold"
    g.mkdir()
    persons = [
        {
            "person_id": "g_tt",
            "name": "Trấn Thành",
            "aliases": [],
            "image_ids": [sha("tt1")[:16], sha("both")[:16]],
        },
        {
            "person_id": "g_hw",
            "name": "Hari Won",
            "aliases": [],
            "image_ids": [sha("hw1")[:16], sha("both")[:16]],
        },
    ]
    rel = {
        "relations": [
            {
                "subject": "g_tt",
                "predicate": "spouse_of",
                "object": "g_hw",
                "status": "explicit",
                "evidence": [
                    {
                        "text": "Vợ chồng Trấn Thành – Hari Won tung bộ ảnh",
                        "url": None,
                        "kind": "source_title",
                    }
                ],
            }
        ]
    }
    (g / "persons.json").write_text(json.dumps(persons, ensure_ascii=False), encoding="utf-8")
    (g / "relations.json").write_text(json.dumps(rel, ensure_ascii=False), encoding="utf-8")
    return persons


def test_auto_mode_reads_only_metadata(golden_dir):
    ds, report = to_dataset(golden_dir, "auto")  # không có thư mục gold/: không được đọc
    names = {p.person_id: p.name for p in ds.persons}
    assert sorted(names.values()) == ["Hari Won", "Trấn Thành"]
    both = [x.entity_id for x in ds.image_entity if x.image_id == sha("both")]
    assert len(both) == 2  # ảnh đôi gắn cho cả hai người
    # spouse đối xứng: thứ tự hai đầu theo id, nên so cặp không thứ tự
    assert [(r.predicate, {names[r.subj_person], names[r.obj_person]}) for r in ds.relations] == [
        ("spouse", {"Trấn Thành", "Hari Won"})
    ]
    assert report["multi_person_entities"] == 1
    img = next(i for i in ds.images if i.image_id == sha("tt1"))
    assert (
        img.caption == "Nam MC cầm micro sân khấu" and img.description == "Trấn Thành dẫn gameshow"
    )


def test_oracle_mode_uses_gold(golden_dir):
    add_gold(golden_dir)
    ds, report = to_dataset(golden_dir, "oracle")
    assert {p.person_id for p in ds.persons} >= {"g_tt", "g_hw"}
    assert [(r.subj_person, r.predicate, r.obj_person) for r in ds.relations] == [
        ("g_tt", "spouse", "g_hw")
    ]
    assert report["gold_persons"] == 2


def test_unknown_mode_rejected(golden_dir):
    with pytest.raises(ValueError):
        to_dataset(golden_dir, "magic")


@pytest.fixture()
def golden_engine(golden_dir, tmp_path):
    ds, _ = to_dataset(golden_dir, "auto")
    write_jsonl_dir(ds, tmp_path / "jsonl")
    build_index(tmp_path / "jsonl", tmp_path / "index", "g")
    e = SearchEngine(Settings(_env_file=None, index_dir=tmp_path / "index", index_version="g"))
    e.load()
    return e


def test_evaluate_end_to_end(golden_engine, golden_dir):
    persons = add_gold(golden_dir)
    s = lambda k: sha(k)[:16]  # noqa: E731
    queries = [
        GoldQuery("q1", "single", "Hari Won", "results", relevant={s("hw1"): 2, s("both"): 1}),
        GoldQuery("q2", "multi_hop", "chồng của Hari Won", "results", relevant={s("tt1"): 2}),
        GoldQuery("q3", "multi_hop", "con của Hari Won", "no_evidence"),
        GoldQuery("q4", "single", "Nguyễn Văn An", "empty"),
    ]
    results = evaluate(golden_engine, queries)
    assert [r["behavior_ok"] for r in results] == [True, True, True, True]
    assert results[0]["r50"] == 1.0  # ảnh rejected không vào index
    summary = summarize(results)
    assert summary["all"]["behavior_ok"] == 4 and summary["multi_hop"]["n"] == 2
    merge = merge_report(golden_engine.store, persons)
    assert merge["pair_precision"] == 1.0 and merge["pair_recall"] == 1.0


def _resp(groups=(), messages=(), reasoning=()):
    return SearchResponse(
        query="q",
        query_type="single",
        groups=[
            ResultGroup(
                label=f"g{n}",
                kind=kind,
                hits=[ImageHit(image_id=i * 4, thumb_url="", score=0) for i in ids],
                total=len(ids),
            )
            for n, (kind, ids) in enumerate(groups)
        ],
        messages=list(messages),
        reasoning=list(reasoning),
    )


def test_flatten_interleaves_groups_and_dedupes():
    resp = _resp([("intent", ["a", "b", "c"]), ("intent", ["x", "a"])])
    assert flatten(resp) == ["aaaa", "xxxx", "bbbb", "cccc"]


@pytest.mark.parametrize(
    ("expected", "resp", "ok"),
    [
        ("empty", _resp(), True),
        ("empty", _resp([("intent", ["a"])]), False),
        ("no_evidence", _resp(messages=["Không tìm thấy bằng chứng quan hệ"]), True),
        ("no_evidence", _resp(messages=["Không tìm thấy ảnh phù hợp."]), False),
        ("disambiguate", _resp([("person", ["a"]), ("person", ["b"])]), True),
        ("disambiguate", _resp([("intent", ["a"])]), False),
        ("partial_with_notice", _resp([("intent", ["a"])], ["Không tìm thấy “X”"]), True),
        ("partial_with_notice", _resp([("intent", ["a"])]), False),
        ("disambiguate_or_no_evidence", _resp(messages=["“A” khớp nhiều người"]), True),
    ],
)
def test_judge_behavior(expected, resp, ok):
    q = GoldQuery("q", "single", "q", expected)
    assert judge_behavior(q.expected, resp, flatten(resp))[0] is ok


def test_unjudged_images_are_skipped_and_p_at_r():
    q = GoldQuery("q", "single", "q", "results", relevant={"aaaa": 2, "bbbb": 1}, unjudged={"zzzz"})
    r = score_query(q, _resp([("intent", ["z", "a", "y", "b"])]))
    assert r["p_r"] == 0.5  # sau khi bỏ zzzz: aaaa, yyyy -> 1/2
    assert r["r50"] == 1.0


def test_reasoning_reported():
    step = EvidenceStep(
        subj="A",
        subj_id="a",
        predicate="spouse",
        predicate_label="vợ/chồng",
        obj="B",
        obj_id="b",
        source="article",
        evidence="e",
    )
    q = GoldQuery("q", "multi_hop", "q", "results")
    assert score_query(q, _resp([("relation_target", ["b"])], reasoning=[step]))["reasoning"] == [
        "B là vợ/chồng của A"
    ]
