"""Gộp entity bằng tên. Ví dụ lấy từ tên entity thật trong dữ liệu (data/golden/README.md)."""

from imgsearch.data.entity_merge import (
    EntityInput,
    derived_aliases,
    merge_entities,
    parse_name,
)


def persons_of(result, entity_id) -> list[str]:
    names = {p.person_id: p.name for p in result.persons}
    return [names[p] for p in result.entity_persons[entity_id]]


def by_name(result, name):
    return next(p for p in result.persons if p.name == name)


def test_parse_strips_prefix_and_reads_paren_alias():
    p = parse_name("Phùng Thanh Độ (Độ Mixi)")
    assert p.words == ["Phùng", "Thanh", "Độ"] and p.clean
    assert p.aliases == [["Độ", "Mixi"]]
    assert parse_name("Tay vợt cầu lông Vũ Thị Trang").words == ["Vũ", "Thị", "Trang"]
    assert parse_name("Võ sĩ Trần Hiếu Ngân - HCB Olympic 2000").words == ["Trần", "Hiếu", "Ngân"]
    assert not parse_name("Tỷ phú Phạm Nhật Vượng sưu tập tranh").clean


def test_paren_descriptor_is_not_alias():
    assert parse_name("Phương Nhi (Á hậu 2 HHVN 2022)").aliases == []
    assert parse_name("Nguyễn Trần Duy Nhất (Muay Thái)").aliases == []  # môn thể thao
    assert parse_name("X (MB Bank CEO)").aliases == []  # viết tắt


def test_E4_fragmented_entities_merge():
    r = merge_entities(
        [
            EntityInput("e1", "Phạm Nhật Vượng", "businessman", 3),
            EntityInput("e2", "Sếp Phạm Nhật Vượng Vingroup quản lý tập đoàn", "businessman", 3),
            EntityInput("e3", "Tỷ phú Phạm Nhật Vượng sưu tập tranh", "businessman", 2),
        ]
    )
    assert len(r.persons) == 1
    assert r.methods == {"e1": "name", "e2": "name_contains", "e3": "name_contains"}


def test_paren_alias_links_streamer_entities():
    r = merge_entities(
        [
            EntityInput("a", "Streamer Độ Mixi (giữ)", "celeb"),
            EntityInput("b", "Phùng Thanh Độ (Độ Mixi)", "celeb"),
            EntityInput("c", "Sinh nhật streamer Độ Mixi fan tổ chức", "celeb"),
            EntityInput("d", "Streamer Độ Mixi và Bomman nhậu reaction", "celeb"),
        ]
    )
    assert len(r.persons) == 1
    assert "Phùng Thanh Độ" in [r.persons[0].name, *r.persons[0].aliases]


def test_same_name_different_occupation_stays_apart_and_suffix_merges():
    r = merge_entities(
        [
            EntityInput("fb1", "Tiền vệ Nguyễn Quang Hải", "athletes", 3),
            EntityInput("fb2", "Quang Hải", "athletes", 2),
            EntityInput("fb3", "Sinh nhật Quang Hải bóng đá tuyển VN", "athletes", 1),
            EntityInput("sg", "Quang Hải (ca sĩ)", "singer", 2),
            EntityInput("sc", "Quang Hải (nhạc trưởng)", "scientist", 2),
        ]
    )
    assert persons_of(r, "fb2") == persons_of(r, "fb1") == persons_of(r, "fb3")
    assert persons_of(r, "fb1") == ["Nguyễn Quang Hải"]
    assert len({r.entity_persons[e][0] for e in ("fb1", "sg", "sc")}) == 3
    assert "Quang Hải" in by_name(r, "Nguyễn Quang Hải").aliases


def test_group_entity_links_both_people():
    r = merge_entities(
        [
            EntityInput("tt", "Trấn Thành (MC)", "celeb"),
            EntityInput("hw", "Hari Won", "singer"),
            EntityInput("both", "Trấn Thành & Hari Won cặp đôi", "celeb"),
            EntityInput("glued", "Thổi nến Trấn Thành Hari Won lãng mạn", "celeb"),
        ]
    )
    assert persons_of(r, "both") == ["Trấn Thành", "Hari Won"]
    assert persons_of(r, "glued") == ["Trấn Thành", "Hari Won"]


def test_E2_stage_name_variants_unify_but_other_son_tung_separate():
    r = merge_entities(
        [
            EntityInput("w", "Sơn Tùng", "writer"),
            EntityInput("s1", "Sơn Tùng M-TP", "singer", 2),
            EntityInput("s2", "Sơn Tùng MTP", "singer"),
            EntityInput("s3", "Sơn Tùng MTP dạo phố với Poodle", "celeb"),
            EntityInput("t", "Tạ Sơn Tùng (Rikkeisoft)", "businessman"),
        ]
    )
    assert persons_of(r, "s1") == persons_of(r, "s2") == persons_of(r, "s3") == ["Sơn Tùng M-TP"]
    assert len({r.entity_persons[e][0] for e in ("w", "s1", "t")}) == 3
    # "Sơn Tùng" là cách gọi của cả ba -> truy vấn "Sơn Tùng" sẽ phải hỏi lại (E2)
    assert "Sơn Tùng" in by_name(r, "Sơn Tùng M-TP").aliases
    assert "Sơn Tùng" in by_name(r, "Tạ Sơn Tùng").aliases


def test_place_name_does_not_link_another_person():
    r = merge_entities(
        [
            EntityInput("s", "Sơn Tùng M-TP", "singer"),
            EntityInput("p", "Vũ Thu Hà (Hà Nội)", "politicer"),
            EntityInput("x", "Sơn Tùng M-TP tốt nghiệp Nhạc viện Hà Nội (tư liệu)", "celeb"),
        ]
    )
    assert persons_of(r, "x") == ["Sơn Tùng M-TP"]
    assert "Hà Nội" not in by_name(r, "Vũ Thu Hà").aliases


def test_fallback_takes_longest_capitalized_run():
    r = merge_entities([EntityInput("v", "Vận động viên Ánh Viên và đồng đội bơi lội", "athletes")])
    assert r.persons[0].name == "Ánh Viên"
    assert r.methods["v"] == "name_fallback"


def test_derived_aliases():
    assert derived_aliases("Trần Khả Như".split()) == ["Khả Như"]
    assert derived_aliases("Sơn Tùng M-TP".split()) == ["Sơn Tùng"]
    assert derived_aliases("Vũ Thị Trang".split()) == []  # không gọi "Thị Trang"
    assert derived_aliases("Lý Hoàng Nam Davis Cup".split()) == []  # không phải tên người
    assert derived_aliases("Hari Won".split()) == []


# --- lỗi gặp khi chạy trên 8k entity thật ---


def test_show_name_in_parens_does_not_chain_people():
    r = merge_entities(
        [
            EntityInput("a", "Anh Tú (The Voice)", "singer"),
            EntityInput("b", "Minh Hằng (The Voice)", "singer"),
            EntityInput("c", "Nguyễn Văn Bình (Next Top Model)", "celeb"),
            EntityInput("d", "Trần Thị Lan (Next Top Model)", "celeb"),
        ]
    )
    assert len({tuple(v) for v in r.entity_persons.values()}) == 4


def test_names_differing_only_in_tone_marks_stay_apart():
    r = merge_entities(
        [EntityInput("a", "Tuấn Hưng", "singer"), EntityInput("b", "Trần Tuấn Hùng", "singer")]
    )
    assert r.entity_persons["a"] != r.entity_persons["b"]


def test_one_word_name_does_not_absorb_descriptive_entities():
    r = merge_entities(
        [
            EntityInput("a", "Nghệ", "singer"),
            EntityInput("b", "Nghệ nhân Ba Tu (đờn kìm)", "singer"),
            EntityInput("c", "Nghệ nhân Kray Sức", "singer"),
        ]
    )
    assert len({tuple(v) for v in r.entity_persons.values()}) == 3


def test_hyphenated_academic_title_is_stripped():
    assert parse_name("GS-TS Trần Quang Hải").words == ["Trần", "Quang", "Hải"]
    assert parse_name("Sơn Tùng M-TP").words == ["Sơn", "Tùng", "M-TP"]
