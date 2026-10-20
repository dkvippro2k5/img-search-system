import pytest
from fastapi.testclient import TestClient

from imgsearch.api.main import app
from imgsearch.config import get_settings


@pytest.fixture(scope="module")
def client(mini_index):
    # Dùng index mini dựng riêng cho test, không phụ thuộc ../data trên máy đang chạy.
    out, _ = mini_index
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INDEX_DIR", str(out))
        mp.setenv("INDEX_VERSION", "test")
        get_settings.cache_clear()
        with TestClient(app) as c:
            yield c
    get_settings.cache_clear()


def test_healthz_ready(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["n_images"] == 27


def test_search_returns_server_timing(client):
    resp = client.post("/search", json={"query": "Đen Vâu"})
    assert resp.status_code == 200
    assert "total;dur=" in resp.headers["Server-Timing"]


@pytest.mark.parametrize("query", ["", "   ", "x" * 2000])
def test_search_rejects_invalid_query(client, query):
    # SCENARIOS L9: rỗng / quá dài -> lỗi hợp lệ, không crash
    assert client.post("/search", json={"query": query}).status_code == 422


def test_search_does_not_execute_markup(client):
    resp = client.post("/search", json={"query": "<script>alert(1)</script>"})
    assert resp.status_code == 200
    assert resp.json()["query"] == "<script>alert(1)</script>"


def test_search_returns_groups_from_index(client):
    body = client.post("/search", json={"query": "Trấn Thành và Hari Won"}).json()
    assert body["query_type"] == "multi_intent"
    assert [g["kind"] for g in body["groups"]] == ["intent", "intent", "co_occurrence"]
    assert all(g["hits"] for g in body["groups"])


def test_search_relation_has_reasoning(client):
    body = client.post("/search", json={"query": "vợ của Trấn Thành"}).json()
    assert body["reasoning"][0]["evidence"]
    assert body["groups"][0]["kind"] == "relation_target"


def test_metrics_counts_requests_empty_and_latency(client):
    before = client.get("/metrics").json()
    client.post("/search", json={"query": "Đen Vâu"})
    client.post("/search", json={"query": "con của Mỹ Tâm"})  # H3: không bằng chứng, không ảnh
    client.post("/search", json={"query": ""})  # 422: không tính là request đã xử lý
    m = client.get("/metrics").json()
    req, lat = m["requests"], m["latency_ms"]
    assert req["total"] - before["requests"]["total"] == 2
    assert req["empty"] - before["requests"]["empty"] == 1
    assert req["with_messages"] - before["requests"]["with_messages"] >= 1
    assert sum(lat["histogram"].values()) == req["total"]
    assert lat["p50"] is not None and lat["p95"] >= lat["p50"]
    assert {"parse", "total"} <= set(lat["step_avg"])
    assert m["index"]["n_images"] == 27 and m["mentor"] is None  # index mini: kênh văn bản tắt
