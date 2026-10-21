"""SCENARIOS X2: truy vấn tìm ảnh nhạy cảm của người thật bị từ chối, kể cả khi viết lách."""

import pytest

from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.retrieval import guard
from imgsearch.retrieval.engine import SearchEngine


@pytest.mark.parametrize(
    "query",
    [
        "Ngọc Trinh khỏa thân",
        "khoa than Ngoc Trinh",  # không dấu
        "Ngọc Trinh k.h.ỏ.a t.h.â.n",  # chữ cách dấu chấm
        "ảnh s e x người nổi tiếng",  # chữ cách khoảng trắng
        "Ngoc Trinh nud3",  # số thay chữ
        "Hari Won ảnh nóng",
        "Ngọc Trinh 18+",
        "naked celebrity",
    ],
)
def test_strict_terms_blocked_with_or_without_person(query):
    assert guard.check(query, has_person=False).blocked


@pytest.mark.parametrize("query", ["Hồ Ngọc Hà bikini", "thi thể Trấn Thành", "Mỹ Tâm gợi cảm"])
def test_borderline_terms_blocked_only_with_person(query):
    assert guard.check(query, has_person=True).blocked
    assert not guard.check(query, has_person=False).blocked  # "bikini" chủ đề chung: cho qua


@pytest.mark.parametrize(
    "query",
    [
        "bác sĩ khoa thận",  # khác dấu với "khỏa thân": không chặn
        "Đen Vâu hát trong concert",
        "Trấn Thành và Hari Won",
        "lô hàng xuất khẩu",
        "U23 Việt Nam",
        "Sơn Tùng M-TP",
    ],
)
def test_normal_queries_pass(query):
    assert not guard.check(query, has_person=True).blocked


def test_engine_refuses_without_results(mini_index):
    out, _ = mini_index
    e = SearchEngine(Settings(_env_file=None, index_dir=out, index_version="test"))
    e.load()
    resp, _ = e.search(SearchRequest(query="Hari Won bikini"))
    assert resp.query_type == "refused" and resp.groups == [] and resp.messages
    resp, _ = e.search(SearchRequest(query="bikini"))  # không có tên người: tìm bình thường
    assert resp.query_type != "refused"
