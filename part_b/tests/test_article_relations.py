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
    # tên lấy từ tiêu đề thật trong data/raw (bẫy ở quy mô 80k)
    PersonEntry("bao", "Bảo", "singer"),
    PersonEntry("pd", "Phương Dung", "singer"),
    PersonEntry("hcm", "Hồ Chí Minh", "historic"),
    PersonEntry("ntt", "Nguyễn Đức Trung", "singer"),
    PersonEntry("nhp", "Nguyễn Hạnh Phúc", "politicer", ("Hạnh Phúc",)),
    PersonEntry("hm", "Hòa Minzy", "singer", ("Hoà Minzy",)),
    PersonEntry("ncd", "Ngô Chí Dũng", "businessman"),
    PersonEntry("nkh", "Ngô Kiến Huy", "singer"),
    PersonEntry("tth", "Thanh Thảo", "singer"),
    PersonEntry("cl", "Cẩm Ly", "singer"),
    PersonEntry("mv", "Minh Vy", "singer"),
    PersonEntry("kng", "Kiều Nga", "singer"),
    PersonEntry("ep", "Elvis Phương", "singer"),
    PersonEntry("na", "Nguyễn Ánh", "historic"),
    PersonEntry("dp", "Đức Phúc", "singer"),
    PersonEntry("phn", "Phi Nhung", "singer"),
    PersonEntry("blm", "Bùi Lê Mận", "singer"),
    PersonEntry("cnh", "Châu Ngọc Hà", "celeb"),
    PersonEntry("ql", "Quang Lê", "singer"),
    PersonEntry("bt", "Bình Tinh", "singer"),
    PersonEntry("ktl", "Kim Tử Long", "singer"),
    PersonEntry("hcv", "HCV SEA Games", "athletes"),
    PersonEntry("nvl", "Nguyễn Văn Lai", "athletes"),
    PersonEntry("cc", "Chính Chu", "businessman"),
    PersonEntry("khv", "Khắc Việt", "singer"),
    PersonEntry("khh", "Khắc Hưng", "singer"),
    PersonEntry("trinh", "Trinh Trinh", "celeb"),
    PersonEntry("sl", "Sara Lưu", "singer"),
    PersonEntry("dk", "Đăng Khôi", "singer"),
    PersonEntry("cn", "Chinh Nhân", "singer"),
    PersonEntry("pduy", "Phạm Duy", "singer"),
    PersonEntry("tng", "Tuấn Ngọc", "singer"),
    PersonEntry("vql", "Vân Quang Long", "singer"),
    PersonEntry("dkp", "Đinh Kiến Phong", "singer"),
    PersonEntry("ltt", "Lê Thu Trang", "celeb", ("Thu Trang",)),
    PersonEntry("cdl", "Cường Đô La", "businessman"),
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
        ("Ca sĩ Kiều Nga - em gái Elvis Phương qua đời", ("ep", "sibling", "kng")),
        ("Khắc Hưng - Em trai Khắc Việt: Nhạc sĩ Việt đầu tiên", ("khv", "sibling", "khh")),
        ("Trinh Trinh làm vợ 3 Kim Tử Long", ("ktl", "spouse", "trinh")),
        ("NSND Kiều Nga, em gái danh ca Elvis Phương giờ ra sao?", ("ep", "sibling", "kng")),
        (
            "Ca sĩ Cẩm Ly tiết lộ cuộc sống hôn nhân với ông xã Minh Vy - Báo Người Lao Động",
            ("mv", "spouse", "cl"),
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
        # --- bẫy gặp trên dữ liệu thật ---
        "Ca sĩ Phương Dung về Úc chịu tang chồng - Báo Người Lao Động",  # "Báo" ≠ "Bảo"
        "Ca sĩ Phương Dung về Úc chịu tang chồng | Báo Người Lao Động",
        "Nhạc sĩ Nguyễn Đức Trung: Tự hào là người con của TP. Hồ Chí Minh",  # địa danh
        "Hoà Minzy khoe con trai hạnh phúc đón sinh nhật bên 'bố Cương'",  # chữ thường
        "Ông Ngô Chí Dũng cùng vợ ông Phạm Nhật Vượng vào danh sách tỷ phú Forbes",
        "Ngô Kiến Huy sau 11 năm có con riêng với em gái ca sĩ Thanh Thảo",
        "Nguyễn Quang chưa bao giờ áp lực khi là con trai của Nguyễn Ánh 9",  # khớp mờ
        'Nhịp Showbiz: Đức Phúc bị "vạ lây"; Em trai Phi Nhung viết tâm thư',  # khác mệnh đề
        'Sao Mai Bùi Lê Mận: "Tôi muốn lúc nào cũng là "của lạ" của chồng"',  # câu nói
        "MC Kỳ Duyên, Quang Lê dự đám cưới của Châu Ngọc Hà",  # đám cưới ≠ cưới
        "Ông xã Bình Tinh nói lí do không thể mời bà xã, Kim Tử Long đóng MV",  # qua dấu phẩy
        "Hari Won trải lòng về việc khó sinh con, Trấn Thành nhắn nhủ xúc động",
        "Vợ Nguyễn Văn Lai bế con, âu yếm ngắm chồng nhận HCV SEA Games 31",  # không phải người
        "Chính Chu - tỷ phú Mỹ gốc Việt trước tin đồn ly hôn em gái Cẩm Ly",  # em gái Cẩm Ly
        "Sara Lưu ngượng ngùng khi đóng cảnh yêu đương với em trai Đăng Khôi",
        "Bình Tinh khóc nức nở nhìn con trai cố nghệ sĩ Chinh Nhân ca diễn",
        # đúng sự thật nhưng cú pháp giống hai câu trên: bỏ để ưu tiên độ chính xác
        "Khắc Việt từng bật khóc khi đánh em trai Khắc Hưng",
        'Thái Hiền: "Ái nữ" nhạc sĩ Phạm Duy, chị vợ Tuấn Ngọc',  # họ hàng bên vợ
        "Vợ của Vân Quang Long cưới ca sĩ Đinh Kiến Phong sau 4 năm",  # X là người sở hữu
        "Cuộc sống sang chảnh của Đàm Thu Trang trước khi làm vợ Cường Đô La",  # tên cụt
        "Linh Lan tổ chức đám cưới với ca sĩ Đinh Kiến Phong, con gái Vân Quang Long",
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
