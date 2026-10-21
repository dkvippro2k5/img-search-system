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
        "hiep dam",  # cụm dài: không dấu vẫn chặn
        "phim heo",
        "địt",  # từ tục có dấu
        "Ngọc Trinh địttt",  # chữ lặp
        "cặc",
        "đéo",
        "djt me",  # teencode không dấu (regex)
        "đcm",
        "vcl Sơn Tùng",
        "loz",
        "fuckkk",
        "pornhub",
    ],
)
def test_strict_terms_blocked_with_or_without_person(query):
    assert guard.check(query, has_person=False).blocked


@pytest.mark.parametrize(
    "query",
    [
        "Hồ Ngọc Hà bikini",
        "thi thể Trấn Thành",
        "Mỹ Tâm gợi cảm",
        "Ngọc Trinh áo tắm",
        "Hồ Ngọc Hà hở hang",  # chỉ khớp khi gõ đúng dấu ("hang" vốn không dấu)
        "Hari Won mát mẻ",
    ],
)
def test_borderline_terms_blocked_only_with_person(query):
    assert guard.check(query, has_person=True).blocked
    assert not guard.check(query, has_person=False).blocked  # "bikini" chủ đề chung: cho qua


@pytest.mark.parametrize(
    "query",
    [
        "cac ca si",  # bỏ dấu trùng "cặc"
        "Trấn Thành deo kinh",  # "đéo"
        "buoi bieu dien",  # "buồi"
        "Mỹ Tâm đi du lịch",  # "đụ", "đĩ"
        "dep nhu hoa",  # "nhũ hoa"
        "Trấn Thành ho hang",  # "hở hang"
        "Hồ Ngọc Hà mat me",  # "mát mẻ"
        "Đàm Vĩnh Hưng màu mè",  # "máu me"
        "mua dam",  # "mua dâm" / mưa dầm
        "người phụ nữ đảm đang",  # "dâm đãng"
        "NSND Thanh Lam tình tứ",  # "làm tình" khi bỏ dấu
        "hoa dâm bụt",
        "Krajan Dick",  # tên person trong dữ liệu
    ],
)
def test_words_equal_to_sensitive_terms_without_marks_pass(query):
    assert not guard.check(query, has_person=True).blocked


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
