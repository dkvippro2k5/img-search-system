"""Trích quan hệ từ tiêu đề. Tiêu đề lấy từ data/golden (bằng chứng thật và bẫy)."""

import pytest

from imgsearch.kg.article_relations import extract_relations, relations_in_title
from imgsearch.retrieval.aliases import AliasIndex, PersonEntry
from imgsearch.retrieval.query_parser import QueryParser

PERSONS = [
    PersonEntry("pnv", "Phạm Nhật Vượng", "businessman"),
    PersonEntry("pnmh", "Phạm Nhật Minh Hoàng", "businessman", ("Minh Hoàng",)),
    PersonEntry("pnqa", "Phạm Nhật Quân Anh", "businessman"),
    PersonEntry("pth", "Phạm Thu Hương", "businessman"),
    PersonEntry("pn", "Phương Nhi", "celeb"),
    PersonEntry("tt", "Trấn Thành", "celeb"),
    PersonEntry("hw", "Hari Won", "celeb"),
    PersonEntry("bk", "Bằng Kiều", "singer"),
    PersonEntry("tuan", "Tuấn Trần", "celeb"),
    PersonEntry("kn", "Khả Như", "celeb"),
    PersonEntry("hang", "Thanh Hằng", "celeb"),
]


@pytest.fixture(scope="module")
def parser():
    return QueryParser(AliasIndex(PERSONS))


@pytest.mark.parametrize(
    ("title", "edge"),
    [
        (
            "Doanh nhân Phạm Nhật Minh Hoàng - con trai thứ 2 của tỷ phú Phạm Nhật Vượng",
            ("pnv", "child", "pnmh"),
        ),
        (
            "Phạm Nhật Minh Hoàng, con trai tỉ phú Phạm Nhật Vượng nắm nhiều vị trí",
            ("pnv", "child", "pnmh"),
        ),
        (
            "Bà Phạm Thu Hương - Người vợ kín tiếng của tỷ phú Phạm Nhật Vượng và những chu…",
            ("pnv", "spouse", "pth"),
        ),
        (
            "Gia thế Phạm Nhật Minh Hoàng - chồng á hậu Phương Nhi - Ngôi sao",
            ("pn", "spouse", "pnmh"),
        ),
        ("Trấn Thành tiết lộ cuộc sống sau 8 năm cưới Hari Won", ("hw", "spouse", "tt")),
        (
            "Vợ chồng Trấn Thành – Hari Won tung bộ ảnh tình tứ ngày Valentine",
            ("tt", "spouse", "hw"),
        ),
        (
            "Hari Won gặp áp lực khi lần đầu làm MC cùng ông xã Trấn Thành",
            ("tt", "spouse", "hw"),
        ),
    ],
)
def test_explicit_evidence(parser, title, edge):
    assert relations_in_title(title, parser) == [edge]


@pytest.mark.parametrize(
    "title",
    [
        "Sự nghiệp thăng trầm của Tuấn Trần – “con trai” Trấn Thành",  # gọi thân mật
        "Khả Như từ bỏ hình tượng quý cô sang chảnh khi đóng vợ Trấn Thành",  # vai diễn
        'Trấn Thành chính thức lộ diện là MC Anh Trai "Say Hi"',  # tên chương trình
        "Trấn Thành tuyên bố 'tình cảm cũng có hạn sử dụng', Hari Won liền phản ứng",  # tuyên bố
        "Hari Won bày tỏ mong muốn có con với Trấn Thành",  # có con với
        "Vợ chồng Trấn Thành đi 'hâm nóng', Bằng Kiều mừng sinh nhật mẹ",  # B không phải vợ A
        "Con trai ông Phạm Nhật Vượng làm Chủ tịch VinFast",  # không nêu tên người con
        "Ông Phạm Nhật Quân Anh làm Chủ tịch VinFast",  # chức vụ, không phải quan hệ
        "Nhan sắc Thanh Hằng thăng hạng sau khi lấy chồng",  # không nêu tên người chồng
    ],
)
def test_traps_produce_no_edge(parser, title):
    assert relations_in_title(title, parser) == []


def test_no_two_step_inference(parser):
    titles = [
        ("Con trai ông Phạm Nhật Vượng làm Chủ tịch VinFast", None),
        ("Ông Phạm Nhật Quân Anh làm Chủ tịch VinFast", None),
    ]
    assert extract_relations(titles, parser) == []


def test_spouse_edges_deduplicated_and_keep_evidence(parser):
    titles = [
        ("Gia thế Phạm Nhật Minh Hoàng - chồng á hậu Phương Nhi - Ngôi sao", "u1"),
        ("Doanh nhân Phạm Nhật Minh Hoàng sắp cưới á hậu Phương Nhi là ai?", "u2"),
    ]
    edges = extract_relations(titles, parser)
    assert {(e.subj_person, e.predicate, e.obj_person) for e in edges} == {("pn", "spouse", "pnmh")}
    assert [e.evidence_url for e in edges] == ["u1", "u2"]
    assert all(e.source == "article" and e.evidence for e in edges)
