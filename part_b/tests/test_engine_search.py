"""Test đầu-cuối của engine trên bộ dữ liệu giả mini, đặt tên theo mã kịch bản SCENARIOS.md."""

import pytest

from conftest import img
from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.retrieval.engine import MSG_NO_INDEX, SearchEngine


@pytest.fixture(scope="module")
def engine(mini_index) -> SearchEngine:
    out, _ = mini_index
    e = SearchEngine(Settings(index_dir=out, index_version="test"))
    e.load()
    return e


def run(engine, query, **kw):
    resp, timings = engine.search(SearchRequest(query=query, **kw))
    return resp


def ids(group) -> list[str]:
    return [h.image_id for h in group.hits]


def all_ids(resp) -> set[str]:
    return {h.image_id for g in resp.groups for h in g.hits}


def has_message(resp, fragment: str) -> bool:
    return any(fragment in m for m in resp.messages)


# --- Single-intent ---


@pytest.mark.parametrize("query", ["Đen Vâu", "den vau", "ảnh Đen Vâu"])
def test_E1_L1_person_query_returns_all_approved_images(engine, query):
    resp = run(engine, query)
    assert resp.query_type == "single"
    assert len(resp.groups) == 1
    g = resp.groups[0]
    assert g.label == "Đen Vâu"
    assert g.person_ids == ["den_vau"]
    assert set(ids(g)) == {img("den1"), img("den2"), img("den3")}  # không có den4 (rejected)


def test_E4_E7_fragmented_entities_one_group_no_duplicates(engine):
    g = run(engine, "Phạm Nhật Vượng").groups[0]
    assert sorted(ids(g)) == sorted([img("pnv1"), img("pnv2"), img("pnv3")])
    assert g.total == 3


def test_S1_person_with_context_ranks_context_first_and_keeps_rest(engine):
    g = run(engine, "Mỹ Tâm trên sân khấu").groups[0]
    assert g.label == "Mỹ Tâm · xếp theo “trên sân khấu”"
    assert ids(g)[0] == img("mt1")
    assert set(ids(g)) == {img("mt1"), img("mt2"), img("mt3"), img("mt4")}  # chỉ ảnh Mỹ Tâm


def test_S3_topic_query_uses_category(engine):
    g = run(engine, "vận động viên cầu lông").groups[0]
    assert g.person_ids == []
    assert set(ids(g)[:2]) == {img("ntli1"), img("ntli2")}
    # Giới hạn theo nghề "vận động viên": không lẫn ảnh ca sĩ, doanh nhân...
    persons = {p for h in g.hits for p in h.person_ids}
    assert persons <= {"nguyen_thuy_linh", "nguyen_tien_linh"}


def test_E2_same_name_split_by_person(engine):
    resp = run(engine, "Sơn Tùng")
    assert [g.kind for g in resp.groups] == ["person", "person"]
    assert {g.person_ids[0] for g in resp.groups} == {"son_tung_mtp", "son_tung_writer"}
    assert {g.label for g in resp.groups} == {"Sơn Tùng M-TP (ca sĩ)", "Sơn Tùng (nhà văn)"}
    assert has_message(resp, "khớp nhiều người")


def test_E3_occupation_picks_one_person(engine):
    resp = run(engine, "nhà văn Sơn Tùng")
    assert len(resp.groups) == 1
    assert ids(resp.groups[0]) == [img("stw1")]


def test_E6_X1_unknown_person_returns_nothing(engine):
    resp = run(engine, "Nguyễn Văn Bình")
    assert resp.groups == []
    assert has_message(resp, "Không tìm thấy “Nguyễn Văn Bình”")


def test_L2_suggestion_passed_through(engine):
    resp = run(engine, "Hồ Ngọc Hạ")
    assert resp.suggestions == ["Hồ Ngọc Hà"]
    assert set(ids(resp.groups[0])) == {img("hnh1"), img("hnh2")}  # hnh3 là bản trùng


def test_L7_unsupported_negation_is_reported(engine):
    resp = run(engine, "Trấn Thành không đeo kính")
    assert has_message(resp, "phủ định")
    assert resp.groups  # vẫn trả ảnh của Trấn Thành


def test_no_result_message(engine):
    resp = run(engine, "tàu vũ trụ")
    assert resp.groups == []
    assert resp.messages == ["Không tìm thấy ảnh phù hợp."]


# --- Multi-intent ---


def test_M1_each_person_plus_together_group(engine):
    resp = run(engine, "Trấn Thành và Hari Won")
    assert resp.query_type == "multi_intent"
    kinds = [g.kind for g in resp.groups]
    assert kinds == ["intent", "intent", "co_occurrence"]
    tt, hw, both = resp.groups
    assert set(ids(tt)) == {img("tt1"), img("tt2"), img("tt_hw1")}
    assert set(ids(hw)) == {img("hw1"), img("hw2"), img("tt_hw1")}  # hw_nsfw bị lọc an toàn
    assert ids(both) == [img("tt_hw1")]
    assert both.label == "Cùng xuất hiện: Trấn Thành + Hari Won"


@pytest.mark.parametrize(
    "query", ["ảnh Trấn Thành chụp cùng Hari Won", "Gia đình Trấn Thành – Hari Won"]
)
def test_M2_M7_together_returns_only_shared_images(engine, query):
    resp = run(engine, query)
    assert len(resp.groups) == 1
    assert resp.groups[0].kind == "co_occurrence"
    assert ids(resp.groups[0]) == [img("tt_hw1")]


def test_M2_no_shared_image_falls_back_to_each_person(engine):
    resp = run(engine, "Đen Vâu chụp cùng Mỹ Tâm")
    assert has_message(resp, "Không có ảnh nào có cả")
    assert [g.person_ids for g in resp.groups] == [["den_vau"], ["my_tam"]]


def test_M3_same_person_two_contexts(engine):
    resp = run(engine, "Mỹ Tâm trên sân khấu và Mỹ Tâm thời thơ ấu")
    stage, child = resp.groups  # không có nhóm "cùng xuất hiện" của một người với chính mình
    assert ids(stage)[0] == img("mt1")
    assert ids(child)[0] == img("mt2")


def test_M4_two_topics(engine):
    resp = run(engine, "ca sĩ trên sân khấu và cầu thủ bóng đá")
    assert len(resp.groups) == 2
    assert ids(resp.groups[1])[0] in {img("ntl1"), img("ntl2")}


def test_M5_missing_side_reported_other_side_shown(engine):
    resp = run(engine, "Đen Vâu và Nguyễn Văn Bình")
    assert [g.person_ids for g in resp.groups] == [["den_vau"]]
    assert has_message(resp, "Không tìm thấy “Nguyễn Văn Bình”")


def test_M6_small_side_still_present(engine):
    resp = run(engine, "Mỹ Tâm và nhà văn Sơn Tùng", top_k=2)
    assert [g.person_ids for g in resp.groups] == [["my_tam"], ["son_tung_writer"]]
    assert len(resp.groups[0].hits) == 2 and resp.groups[0].total == 4
    assert ids(resp.groups[1]) == [img("stw1")]


# --- Multi-hop ---


@pytest.mark.parametrize(
    ("query", "target", "expected"),
    [
        ("ảnh vợ của MC Trấn Thành", "hari_won", {img("hw1"), img("hw2"), img("tt_hw1")}),
        # Cạnh lưu (Trấn Thành, spouse, Hari Won); hỏi chiều ngược vẫn ra.
        ("ông xã Hari Won", "tran_thanh", {img("tt1"), img("tt2"), img("tt_hw1")}),
    ],
)
def test_H1_relation_with_evidence(engine, query, target, expected):
    resp = run(engine, query)
    assert resp.query_type == "multi_hop"
    assert len(resp.groups) == 1
    g = resp.groups[0]
    assert g.kind == "relation_target" and g.person_ids == [target]
    assert set(ids(g)) == expected
    step = resp.reasoning[0]
    assert step.subj_id == "tran_thanh" and step.obj_id == "hari_won"
    assert "ông xã Trấn Thành" in step.evidence
    assert step.evidence_url
    assert step.predicate == "spouse" and step.predicate_label == "vợ/chồng"


def test_H3_no_evidence_no_images(engine):
    resp = run(engine, "con trai của Mỹ Tâm")
    assert resp.groups == []
    assert resp.reasoning == []
    assert has_message(resp, "Không tìm thấy bằng chứng quan hệ")


def test_H4_evidence_but_target_has_no_images(engine):
    resp = run(engine, "ảnh con trai chủ tịch Phạm Nhật Vượng")
    assert resp.groups == []
    assert resp.reasoning[0].obj_id == "pham_nhat_minh_hoang"
    assert has_message(resp, "không có ảnh của Phạm Nhật Minh Hoàng")


def test_no_relation_inferred_from_name(engine):
    # Phạm Nhật Quân Anh có ảnh nhưng không có cạnh: không được trả về (PLAN 4.1).
    resp = run(engine, "con của Phạm Nhật Vượng")
    assert "pham_nhat_quan_anh" not in {p for g in resp.groups for p in g.person_ids}
    assert img("pnqa1") not in all_ids(resp)


def test_H7_ambiguous_anchor_asks_back(engine):
    resp = run(engine, "vợ của Sơn Tùng")
    assert resp.groups == []
    assert has_message(resp, "khớp nhiều người")


# --- Kết quả và an toàn ---


def test_Q4_safe_search_on_by_default(engine):
    assert img("hw_nsfw") not in all_ids(run(engine, "Hari Won"))
    assert img("hw_nsfw") not in all_ids(run(engine, "bãi biển"))


def test_Q4_safe_search_can_be_turned_off(engine):
    assert img("hw_nsfw") in all_ids(run(engine, "Hari Won", safe_search=False))


def test_hits_are_hydrated(engine):
    hit = run(engine, "Đen Vâu").groups[0].hits[0]
    assert hit.thumb_url.startswith("https://example.com/img/")
    assert hit.description and hit.source_url
    assert hit.person_ids == ["den_vau"]


def test_top_k_and_total(engine):
    g = run(engine, "Mỹ Tâm", top_k=1).groups[0]
    assert len(g.hits) == 1 and g.total == 4


def test_timings_reported(engine):
    _, timings = engine.search(SearchRequest(query="Đen Vâu"))
    assert set(timings) == {"parse", "retrieve", "hydrate"}


def test_engine_without_index_answers_with_message(tmp_path):
    e = SearchEngine(Settings(index_dir=tmp_path, index_version="none"))
    e.load()
    resp, _ = e.search(SearchRequest(query="Đen Vâu"))
    assert resp.groups == [] and resp.messages == [MSG_NO_INDEX]


def test_thumbs_from_s3_are_presigned(mini_index):
    out, _ = mini_index
    e = SearchEngine(
        Settings(
            index_dir=out,
            index_version="test",
            thumbs_from_s3=True,
            s3_endpoint_url="http://localhost:9000",
            s3_access_key_id="test",
            s3_secret_access_key="test",
        )
    )
    e.load()
    hit = e.search(SearchRequest(query="Đen Vâu"))[0].groups[0].hits[0]
    assert f"thumbs/{hit.image_id[:2]}/{hit.image_id}.webp" in hit.thumb_url
    assert "Signature" in hit.thumb_url
