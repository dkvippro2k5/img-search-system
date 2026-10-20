"""Kênh dense văn bản (P3): MentorClient, nạp text_dense.npz, gộp RRF, lỗi mạng -> bỏ kênh."""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from conftest import img
from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.eval import pooling
from imgsearch.retrieval.engine import MSG_NO_RESULT, SearchEngine
from imgsearch.retrieval.mentor import QUERY_INSTRUCT, MentorClient, MentorTextEncoder
from imgsearch.retrieval.store import MetaStore

MODEL = "qwen3-embedding-4b"
DIM = 4


class FakeEncoder:
    """Truy vấn "kênh văn bản" -> trục 0 (trùng vector ảnh mục tiêu); "mạng lỗi" -> None."""

    dim, name = DIM, MODEL

    def __init__(self):
        self.calls = []

    def encode_query(self, text):
        self.calls.append(text)
        if "mạng lỗi" in text:
            return None
        return np.eye(DIM, dtype=np.float32)[0]


@pytest.fixture
def text_index(mini_index, tmp_path):
    """Bản sao index mini + text_dense.npz: ảnh mt3 nằm đúng trục 0, ảnh khác trục 1."""
    out, _ = mini_index
    target = tmp_path / "vtext"
    shutil.copytree(out / "vtest", target)
    ids = MetaStore(target / "meta.sqlite").image_ids_by_row()
    vecs = np.tile(np.eye(DIM, dtype=np.float16)[1], (len(ids), 1))
    vecs[ids.index(img("mt3"))] = np.eye(DIM, dtype=np.float16)[0]
    np.savez(target / "text_dense.npz", vecs=vecs, has_vec=np.ones(len(ids), bool))
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["text_dense"] = {"model": MODEL, "dim": DIM}
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def make_engine(index_dir, encoder=None, **kw) -> SearchEngine:
    kw = {"text_dense_enabled": True, "text_dense_weight": 2.0, **kw}
    s = Settings(_env_file=None, index_dir=index_dir, index_version="text", **kw)
    e = SearchEngine(s, query_text_encoder=encoder or FakeEncoder())
    e.load()
    return e


def run(engine, query):
    return engine.search(SearchRequest(query=query))[0]


def test_text_channel_loads_and_ranks_target_first(text_index):
    e = make_engine(text_index)
    assert e.info()["text_dense"] == MODEL
    # S1: trong ảnh Mỹ Tâm, BM25 xếp mt1 ("sân khấu") đầu; kênh văn bản (trọng số 2) đẩy mt3 lên
    g = run(e, "Mỹ Tâm kênh văn bản trên sân khấu").groups[0]
    assert [h.image_id for h in g.hits][0] == img("mt3")
    assert {h.image_id for h in g.hits} >= {img("mt1"), img("mt2"), img("mt3"), img("mt4")}


def test_text_channel_alone_answers_query_bm25_misses(text_index):
    resp = run(make_engine(text_index), "zzqq xxww")  # BM25 không khớp gì
    assert resp.groups and resp.groups[0].hits[0].image_id == img("mt3")


def test_mentor_failure_falls_back_to_other_channels(text_index):
    e = make_engine(text_index)
    with_text = [h.image_id for h in run(e, "Mỹ Tâm mạng lỗi trên sân khấu").groups[0].hits]
    e.text_dense = None
    without = [h.image_id for h in run(e, "Mỹ Tâm mạng lỗi trên sân khấu").groups[0].hits]
    assert with_text == without  # encoder trả None -> như không có kênh
    assert (
        MSG_NO_RESULT
        in make_engine(text_index).search(SearchRequest(query="zzqq mạng lỗi"))[0].messages
    )


def test_name_only_query_does_not_call_mentor(text_index):
    enc = FakeEncoder()
    e = make_engine(text_index, encoder=enc)
    run(e, "Mỹ Tâm")
    assert enc.calls == []


def test_text_channel_disabled_on_model_mismatch_or_flag(text_index):
    assert SearchEngine(Settings(_env_file=None)).settings.text_dense_weight == 0.5  # mặc định
    assert make_engine(text_index, text_embed_model="khac").text_dense is None
    assert make_engine(text_index, text_dense_enabled=False).text_dense is None
    # không encoder giả, không key -> tắt, không lỗi
    s = Settings(
        _env_file=None, index_dir=text_index, index_version="text", text_dense_enabled=True
    )
    e = SearchEngine(s)
    e.load()
    assert e.text_dense is None and e.ready


def test_pooling_config_toggles_text_channel(text_index):
    e = make_engine(text_index)
    td = e.text_dense
    assert "w2+t0.5" not in pooling.available(e)  # mini không có dense ảnh
    with pooling.config(e, "bm25"):
        assert e.text_dense is None
    assert e.text_dense is td


# --- MentorClient với server giả ---


class _Fake(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.headers["Authorization"] == "Bearer k"
        if self.path == "/v1/embeddings":
            data = [
                {"index": i, "embedding": [3.0, 4.0] + [0.0] * 6}
                for i, _ in enumerate(body["input"])
            ][::-1]  # trả ngược thứ tự: client phải sắp lại theo index
            out = {"data": data}
        else:
            out = {
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.1},
                ]
            }
        raw = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture(scope="module")
def fake_server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_mentor_client_parses_embeddings_and_rerank(fake_server):
    c = MentorClient(
        "k", f"{fake_server}/v1/embeddings", MODEL, f"{fake_server}/v1/rerank", timeout=5
    )
    v = c.embed(["a", "b"])
    assert v.shape == (2, 8) and np.allclose(v[:, :2], [0.6, 0.8])
    assert c.embed_documents(["x"] * 5, batch=2, workers=2).shape == (5, 8)
    assert c.rerank("q", ["d0", "d1"]) == [0.1, 0.9]
    q = c.embed_query("truy vấn")
    assert c.embed_query("truy vấn") is q  # cache
    enc = MentorTextEncoder(c, 2, MODEL)
    assert np.allclose(enc.encode_query("truy vấn"), [0.6, 0.8])
    assert QUERY_INSTRUCT.startswith("Instruct:")


def test_mentor_client_network_error_returns_none():
    c = MentorClient("k", "http://127.0.0.1:9/v1/embeddings", MODEL, "http://127.0.0.1:9/r", 0.5)
    assert c.embed_query("x") is None
    assert c.rerank("q", ["d"]) is None
    assert c.rerank("q", []) == []


def test_prefetch_sends_all_intents_at_once(fake_server):
    c = MentorClient("k", f"{fake_server}/v1/embeddings", MODEL, timeout=5)
    enc = MentorTextEncoder(c, 2, MODEL)
    enc.prefetch(["vế một", "vế hai", "vế một"])
    assert set(enc._inflight) == {"vế một", "vế hai"}
    assert np.allclose(enc.encode_query("vế hai"), [0.6, 0.8])
    assert set(enc._inflight) == {"vế một"}
    enc.encode_query("vế một")
    enc.prefetch(["vế một"])  # đã có trong cache: không gửi lại
    assert enc._inflight == {}


def test_engine_prefetch_is_reused_by_search(text_index, fake_server):
    c = MentorClient("k", f"{fake_server}/v1/embeddings", MODEL, timeout=5, cache_size=0)
    enc = MentorTextEncoder(c, DIM, MODEL)
    e = make_engine(text_index, encoder=enc)
    calls = []
    real = c.embed_query
    c.embed_query = lambda t: calls.append(t) or real(t)
    e.prefetch("ca sĩ biểu diễn trên sân khấu")
    assert set(enc._inflight) == {"ca sĩ biểu diễn trên sân khấu"}
    run(e, "ca sĩ biểu diễn trên sân khấu")
    assert calls == ["ca sĩ biểu diễn trên sân khấu"] and enc._inflight == {}
    e.prefetch("Mỹ Tâm")  # chỉ có tên: không gọi mentor
    assert enc._inflight == {}
