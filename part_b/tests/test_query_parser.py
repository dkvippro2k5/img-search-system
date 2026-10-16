"""Test bộ phân tích truy vấn theo mã kịch bản trong capstone/SCENARIOS.md.

Từ điển person ở đây là fixture nhỏ, không phải dữ liệu thật; quan hệ không được kiểm tra
ở tầng này (parser chỉ nhận dạng dạng truy vấn, đáp án do đồ thị tri thức quyết định).
"""

import time

import pytest

from imgsearch.retrieval.aliases import AliasIndex, MatchType, PersonEntry
from imgsearch.retrieval.query_parser import QueryParser, QueryType

PERSONS = [
    PersonEntry("den_vau", "Đen Vâu", "singer", ("Đen",)),
    PersonEntry("ho_ngoc_ha", "Hồ Ngọc Hà", "singer", ("Hà Hồ",)),
    PersonEntry("son_tung_mtp", "Sơn Tùng M-TP", "singer", ("Sơn Tùng", "Sếp Tùng")),
    PersonEntry("son_tung_writer", "Sơn Tùng", "writer"),
    PersonEntry("tran_thanh", "Trấn Thành", "celeb"),
    PersonEntry("hari_won", "Hari Won", "celeb"),
    PersonEntry("pham_nhat_vuong", "Phạm Nhật Vượng", "businessman"),
    PersonEntry("my_tam", "Mỹ Tâm", "singer"),
]


@pytest.fixture(scope="module")
def parser() -> QueryParser:
    return QueryParser(AliasIndex(PERSONS))


def persons(parsed) -> list[str]:
    return [pid for m in parsed.mentions for pid in m.person_ids]


# --- L1: gõ không dấu, L2: sai chính tả, L3: biệt danh ---


@pytest.mark.parametrize("query", ["Đen Vâu", "den vau", "DEN VAU", "ảnh Đen Vâu"])
def test_L1_unaccented_links_same_person(parser, query):
    p = parser.parse(query)
    assert persons(p) == ["den_vau"]
    assert p.suggestions == []


def test_L1_accented_match_is_exact(parser):
    assert parser.parse("Hồ Ngọc Hà").mentions[0].match == MatchType.EXACT
    assert parser.parse("ho ngoc ha").mentions[0].match == MatchType.FOLDED


def test_L2_wrong_diacritic_is_corrected_with_suggestion(parser):
    p = parser.parse("Hồ Ngọc Hạ")
    assert persons(p) == ["ho_ngoc_ha"]
    assert p.suggestions == ["Hồ Ngọc Hà"]


@pytest.mark.parametrize("query", ["Sơn Tùng MTP", "son tung mtp", "Sơn Tùng M-TP"])
def test_L2_stage_name_formatting(parser, query):
    p = parser.parse(query)
    assert persons(p) == ["son_tung_mtp"]
    assert p.ambiguous == []


@pytest.mark.parametrize(
    ("query", "expected", "suggestion"),
    [("Tran Thnh", "tran_thanh", "Trấn Thành"), ("Hari Wonn", "hari_won", "Hari Won")],
)
def test_L2_typo_fuzzy_match(parser, query, expected, suggestion):
    p = parser.parse(query)
    assert persons(p) == [expected]
    assert p.mentions[0].match == MatchType.FUZZY
    assert p.suggestions == [suggestion]
    assert p.unknown_names == []


@pytest.mark.parametrize(
    ("query", "expected"), [("Hà Hồ", "ho_ngoc_ha"), ("Sếp Tùng", "son_tung_mtp")]
)
def test_L3_nickname(parser, query, expected):
    assert persons(parser.parse(query)) == [expected]


def test_L3_single_word_alias_needs_capital(parser):
    assert persons(parser.parse("Đen")) == ["den_vau"]
    assert persons(parser.parse("Đen biểu diễn")) == ["den_vau"]
    # S4: "đen trắng" không phải rapper Đen
    assert persons(parser.parse("ảnh tư liệu đen trắng")) == []


# --- L7, L8: phủ định và đếm số chỉ gắn cờ ---


def test_L7_negation_flagged(parser):
    p = parser.parse("Trấn Thành không đeo kính")
    assert "negation" in p.unsupported
    assert persons(p) == ["tran_thanh"]


@pytest.mark.parametrize("query", ["ảnh ba người", "hai cầu thủ", "3 ca sĩ"])
def test_L8_count_flagged(parser, query):
    assert "count" in parser.parse(query).unsupported


def test_L8_year_is_not_count(parser):
    assert parser.parse("Hồ Ngọc Hà năm 2019").unsupported == []


# --- L9: đầu vào lạ không làm crash ---


@pytest.mark.parametrize("query", ["", "   ", "<script>alert(1)</script>", "!!!", "x" * 2000])
def test_L9_odd_input_does_not_crash(parser, query):
    p = parser.parse(query)
    assert p.type == QueryType.SINGLE
    assert p.relation_path == []


def test_long_query_stays_fast(parser):
    t = time.perf_counter()
    parser.parse(" ".join(["Nguyễn Văn Bình"] * 100))
    assert time.perf_counter() - t < 0.5


# --- E2, E3: trùng tên; E6: người không có trong dữ liệu ---


def test_E2_same_name_is_ambiguous(parser):
    p = parser.parse("Sơn Tùng")
    assert sorted(persons(p)) == ["son_tung_mtp", "son_tung_writer"]
    assert p.ambiguous == ["Sơn Tùng"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [("nhà văn Sơn Tùng", "son_tung_writer"), ("ca sĩ Sơn Tùng", "son_tung_mtp")],
)
def test_E3_occupation_disambiguates(parser, query, expected):
    p = parser.parse(query)
    assert persons(p) == [expected]
    assert p.ambiguous == []


def test_E6_unknown_person(parser):
    p = parser.parse("Nguyễn Văn Bình")
    assert persons(p) == []
    assert p.unknown_names == ["Nguyễn Văn Bình"]


def test_S3_topic_query_has_category_and_no_person(parser):
    p = parser.parse("vận động viên cầu lông")
    assert p.type == QueryType.SINGLE
    assert persons(p) == []
    assert p.intents[0].categories == ["athletes"]
    assert p.unknown_names == []


# --- Multi-intent ---


def test_M1_two_people_each(parser):
    p = parser.parse("Trấn Thành và Hari Won")
    assert p.type == QueryType.MULTI_INTENT
    assert [i.person_ids for i in p.intents] == [["tran_thanh"], ["hari_won"]]
    assert p.co_occurrence is False


@pytest.mark.parametrize(
    "query",
    [
        "ảnh Trấn Thành chụp cùng Hari Won",
        "Trấn Thành và Hari Won chụp chung",
        "tran thanh chup cung hari won",
    ],
)
def test_M2_together_in_frame(parser, query):
    p = parser.parse(query)
    assert p.type == QueryType.MULTI_INTENT
    assert sorted(pid for i in p.intents for pid in i.person_ids) == ["hari_won", "tran_thanh"]
    assert p.co_occurrence is True


def test_M3_one_person_two_contexts(parser):
    p = parser.parse("Mỹ Tâm trên sân khấu và Mỹ Tâm thời thơ ấu")
    assert p.type == QueryType.MULTI_INTENT
    assert [i.person_ids for i in p.intents] == [["my_tam"], ["my_tam"]]
    assert p.intents[1].text == "Mỹ Tâm thời thơ ấu"


def test_M4_two_topics(parser):
    p = parser.parse("ca sĩ nhạc rap và cầu thủ bóng đá")
    assert p.type == QueryType.MULTI_INTENT
    assert [i.categories for i in p.intents] == [["singer"], ["athletes"]]


def test_M4_comma_separates_intents(parser):
    p = parser.parse("Đen Vâu, Mỹ Tâm")
    assert [i.person_ids for i in p.intents] == [["den_vau"], ["my_tam"]]


def test_M7_family_group_is_not_split_into_separate_people(parser):
    p = parser.parse("Gia đình Trấn Thành – Hari Won")
    assert p.co_occurrence is True
    assert sorted(persons(p)) == ["hari_won", "tran_thanh"]
    assert p.relation_path == []


def test_M7_couple_phrase_is_not_relation(parser):
    p = parser.parse("vợ chồng Trấn Thành Hari Won")
    assert p.type == QueryType.MULTI_INTENT
    assert p.co_occurrence is True
    assert p.relation_path == []


def test_attribute_after_and_stays_single_intent(parser):
    p = parser.parse("Hồ Ngọc Hà mặc áo dài và đội nón lá")
    assert p.type == QueryType.SINGLE
    assert p.intents[0].person_ids == ["ho_ngoc_ha"]
    assert p.intents[0].text == "Hồ Ngọc Hà mặc áo dài đội nón lá"


def test_single_person_with_together_marker_is_single(parser):
    p = parser.parse("Phạm Nhật Vượng cùng con trai")
    assert p.type == QueryType.SINGLE
    assert p.co_occurrence is False


# --- Multi-hop ---


@pytest.mark.parametrize(
    ("query", "predicate", "anchor"),
    [
        ("ảnh con trai chủ tịch Phạm Nhật Vượng", "child", "pham_nhat_vuong"),
        ("anh con trai pham nhat vuong", "child", "pham_nhat_vuong"),
        ("con trai thứ hai của tỷ phú Phạm Nhật Vượng", "child", "pham_nhat_vuong"),
        ("ảnh vợ của MC Trấn Thành", "spouse", "tran_thanh"),
        ("ông xã Hari Won", "spouse", "hari_won"),
        ("mẹ của Mỹ Tâm", "parent", "my_tam"),
    ],
)
def test_H1_relation_one_hop(parser, query, predicate, anchor):
    p = parser.parse(query)
    assert p.type == QueryType.MULTI_HOP
    assert p.relation_path == [predicate]
    assert p.anchor_person_ids == [anchor]


def test_H7_ambiguous_anchor_asks_back(parser):
    p = parser.parse("vợ của Sơn Tùng")
    assert p.type == QueryType.MULTI_HOP
    assert sorted(p.anchor_person_ids) == ["son_tung_mtp", "son_tung_writer"]
    assert p.ambiguous == ["Sơn Tùng"]


@pytest.mark.parametrize("query", ["anh trai say hi", "Anh Trai Say Hi", "Đen Vâu Anh Trai Say Hi"])
def test_H8_show_name_is_not_relation(parser, query):
    p = parser.parse(query)
    assert p.type == QueryType.SINGLE
    assert p.relation_path == []
    assert p.unknown_names == []


def test_relation_word_without_person_is_not_multi_hop(parser):
    p = parser.parse("ảnh mẹ và con")
    assert p.type != QueryType.MULTI_HOP


def test_empty_alias_index_still_parses():
    p = QueryParser(AliasIndex([])).parse("Trấn Thành và Hari Won")
    assert p.mentions == []
    assert p.unknown_names == ["Trấn Thành", "Hari Won"]
