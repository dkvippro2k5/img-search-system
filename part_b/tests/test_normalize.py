from imgsearch.text.normalize import fold, normalize, strip_diacritics


def test_strip_diacritics_handles_d_bar():
    assert strip_diacritics("Đen Vâu") == "Den Vau"
    assert strip_diacritics("đường") == "duong"


def test_fold_matches_unaccented_query():
    assert fold("Hồ Ngọc Hà") == fold("ho ngoc ha") == "ho ngoc ha"


def test_normalize_keeps_diacritics_and_drops_punctuation():
    assert normalize("  Sơn Tùng M-TP!! ") == "sơn tùng m tp"


def test_normalize_unifies_unicode_forms():
    decomposed = "Hà".replace("à", "à")
    assert normalize(decomposed) == normalize("Hà")
