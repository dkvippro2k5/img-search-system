from imgsearch.config import Settings
from imgsearch.retrieval.engine import SearchEngine


def test_engine_loads_index_and_person_aliases(mini_index):
    out, _ = mini_index
    engine = SearchEngine(Settings(index_dir=out, index_version="test"))
    engine.load()
    info = engine.info()
    assert info["index_loaded"] is True
    assert info["n_images"] == 26
    assert info["n_persons"] == 12
    parsed = engine.parser.parse("Hà Hồ")
    assert parsed.mentions[0].person_ids == ("ho_ngoc_ha",)


def test_engine_without_index_still_starts(tmp_path):
    engine = SearchEngine(Settings(index_dir=tmp_path, index_version="none"))
    engine.load()
    assert engine.ready
    assert engine.info()["index_loaded"] is False
