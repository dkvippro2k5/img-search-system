"""Truy xuất dense (retrieval/dense.py) và gộp với BM25 trong engine, bằng encoder giả.

Vector "thị giác" giả của mỗi ảnh mini sinh từ khái niệm trong caption; encoder giả hiểu từ
đồng nghĩa. "gala" không có trong caption/tiêu đề nào: BM25 trượt, dense phải tìm được ảnh sân khấu.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import MINI, img
from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.data.build_meta import build_index
from imgsearch.data.dataset import read_jsonl_dir
from imgsearch.retrieval.dense import (
    DenseIndex,
    OnnxTextEncoder,
    align_vectors,
    canonicalize,
    min_cosine,
)
from imgsearch.retrieval.engine import SearchEngine

CONCEPTS = {
    "stage": ["sân khấu", "biểu diễn", "concert", "gala", "liveshow", "trình diễn"],
    "aodai": ["áo dài"],
    "badminton": ["cầu lông"],
    "redcarpet": ["thảm đỏ"],
    "studio": ["phòng thu"],
    "beach": ["bãi biển", "đồ bơi"],
}
AXES = list(CONCEPTS)


def concept_vec(text: str) -> np.ndarray:
    t = text.lower()
    v = np.array([any(w in t for w in CONCEPTS[a]) for a in AXES] + [0.05], dtype=np.float32)
    return v / np.linalg.norm(v)


class FakeEncoder:
    name = "fake"
    dim = len(AXES) + 1

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls += texts
        return np.stack([concept_vec(t) for t in texts])


def write_features(path: Path, skip: set[str] = frozenset()) -> None:
    path.mkdir(parents=True, exist_ok=True)
    images = [i for i in read_jsonl_dir(MINI).images if i.image_id not in skip]
    np.save(path / "image_embeddings.npy", np.stack([concept_vec(i.caption or "") for i in images]))
    (path / "image_embeddings.ids.txt").write_text(
        "\n".join(i.image_id for i in images), encoding="utf-8"
    )
    (path / "image_embeddings.json").write_text(json.dumps({"model": "fake"}), encoding="utf-8")


@pytest.fixture(scope="module")
def dense_index(tmp_path_factory):
    root = tmp_path_factory.mktemp("dense")
    write_features(root / "features", skip={img("mt3")})  # một ảnh chưa có vector
    manifest = build_index(MINI, root / "index", "d", root / "features")
    return root / "index", manifest


def engine(index_dir, encoder=None, **kw) -> SearchEngine:
    s = Settings(_env_file=None, index_dir=index_dir, index_version="d", dense_model="fake", **kw)
    e = SearchEngine(s, text_encoder=encoder or FakeEncoder())
    e.load()
    return e


def ids(resp, group=0) -> list[str]:
    return [h.image_id for h in resp.groups[group].hits] if resp.groups else []


def search(e, q, **kw):
    return e.search(SearchRequest(query=q, **kw))[0]


# --- đơn vị ---


def test_canonicalize_like_open_clip():
    assert canonicalize("Hari Won mặc váy cưới!  ") == "hari won mặc váy cưới"
    assert canonicalize("ảnh_có chữ &amp; VINGROUP") == "ảnh có chữ  vingroup".replace("  ", " ")


def test_align_vectors_by_vec_row():
    vecs, has = align_vectors(["b", "a", "c"], np.eye(2, dtype=np.float32), ["a", "b"])
    assert has.tolist() == [True, True, False]
    assert vecs[0].tolist() == [0, 1] and vecs[1].tolist() == [1, 0] and not vecs[2].any()


def test_dense_search_filters_and_skips_missing_vectors():
    vecs = np.array([[1, 0], [0.9, 0.1], [1, 0], [0, 1]], dtype=np.float32)
    has = np.array([True, True, False, True])
    enc = type("E", (), {"dim": 2, "encode": lambda self, t: np.array([[1.0, 0.0]])})()
    allowed = {("p",): np.array([1, 2, 3])}
    d = DenseIndex(
        vecs, has, ["a", "b", "c", "d"], enc, lambda p, c: allowed[tuple(p)] if p else None
    )
    assert [i for i, _ in d.search("x")] == ["a", "b", "d"]  # c không có vector
    assert [i for i, _ in d.search("x", person_ids=["p"])] == ["b", "d"]
    assert [i for i, _ in d.search("x", limit=1)] == ["a"]
    assert d.search("   ") == []
    with pytest.raises(ValueError, match="chiều"):
        DenseIndex(vecs, has, list("abcd"), type("E", (), {"dim": 3})(), lambda p, c: None)


def test_min_cosine_from_siglip_calibration():
    meta = {"logit_scale": 100.0, "logit_bias": -15.0}
    c = min_cosine(meta, 0.5)
    assert c == pytest.approx(0.15)  # sigmoid(100·0,15 − 15) = 0,5
    assert min_cosine({}, 0.1) is None and min_cosine(meta, 0) is None


def test_min_score_gates_the_whole_channel():
    vecs = np.array([[1, 0], [0.6, 0.8]], dtype=np.float32)
    enc = type("E", (), {"dim": 2, "encode": lambda self, t: np.array([[1.0, 0.0]])})()
    d = DenseIndex(vecs, np.array([True, True]), ["a", "b"], enc, lambda p, c: None, 0.9)
    assert [i for i, _ in d.search("x")] == ["a", "b"]  # ảnh tốt nhất qua cổng: giữ cả danh sách
    d.min_score = 1.5
    assert d.search("x") == []
    d.allowed_rows = lambda p, c: np.array([0, 1])
    assert [i for i, _ in d.search("x", person_ids=["p"])] == [
        "a",
        "b",
    ]  # trong một người: không cắt


def test_build_index_writes_dense(dense_index):
    index_dir, manifest = dense_index
    n = manifest["stats"]["images_indexed"]
    assert manifest["dense"]["model"] == "fake" and manifest["dense"]["images_with_vector"] == n - 1
    assert (index_dir / "vd" / "dense.npz").exists() and "dense.npz" in manifest["files"]


def test_build_skips_dense_when_no_image_matches(tmp_path):
    feats = tmp_path / "f"
    feats.mkdir()
    np.save(feats / "image_embeddings.npy", np.ones((1, 3), dtype=np.float32))
    (feats / "image_embeddings.ids.txt").write_text("không-có-trong-index", encoding="utf-8")
    (feats / "image_embeddings.json").write_text('{"model": "fake"}', encoding="utf-8")
    manifest = build_index(MINI, tmp_path / "index", "x", feats)
    assert "skipped" in manifest["dense"] and not (tmp_path / "index/vx/dense.npz").exists()


def test_build_without_features_has_no_dense(mini_index):
    out, manifest = mini_index
    assert manifest["dense"] == {} and not (out / "vtest" / "dense.npz").exists()


# --- engine ---


def test_dense_finds_what_bm25_misses(dense_index):
    e = engine(dense_index[0])
    got = ids(search(e, "gala"))
    stage = {img(k) for k in ("den1", "hnh2", "stm1", "hw2", "mt1")}
    assert search(engine(dense_index[0], dense_enabled=False), "gala").groups == []
    assert set(got[:5]) == stage


def test_S1_person_plus_description_reorders_but_keeps_all_images(dense_index):
    e = engine(dense_index[0])
    got = ids(search(e, "Hồ Ngọc Hà gala"))
    assert got[0] == img("hnh2") and set(got) == {img("hnh1"), img("hnh2")}  # hnh3 là bản trùng


def test_dense_gets_description_without_the_name(dense_index):
    enc = FakeEncoder()
    e = engine(dense_index[0], enc)
    search(e, "Hồ Ngọc Hà gala")
    assert enc.calls == ["gala"]
    enc.calls.clear()
    search(e, "Hồ Ngọc Hà")  # chỉ có tên: không gọi dense
    assert enc.calls == []


def test_Q4_safety_filter_applies_to_dense_hits(dense_index):
    e = engine(dense_index[0], allow_unsafe_search=True)
    assert img("hw_nsfw") not in ids(search(e, "bãi biển"))
    assert img("hw_nsfw") in ids(search(e, "bãi biển", safe_search=False))


def test_dense_off_when_index_built_with_another_model(dense_index):
    e = engine(dense_index[0])
    assert e.dense is not None and e.info()["dense"] == "fake"
    s = Settings(_env_file=None, index_dir=dense_index[0], index_version="d", dense_model="x")
    other = SearchEngine(s, text_encoder=FakeEncoder())
    other.load()
    assert other.dense is None


def test_real_onnx_encoder_if_present():
    """Chỉ chạy khi đã xuất model (scripts/export_text_encoder.py); không có thì bỏ qua."""
    model_dir = Path(__file__).resolve().parents[2] / "models" / "siglip2-b32"
    if not (model_dir / "text.onnx").exists():
        pytest.skip("chưa xuất model siglip2-b32")
    enc = OnnxTextEncoder(model_dir, "text.onnx")
    v = enc.encode(["ca sĩ trên sân khấu", "ca si tren san khau"])
    assert v.shape == (2, enc.dim) and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-3)
    assert enc.encode_query("áo dài") is enc.encode_query("áo dài")  # cache
