import pytest
from fastapi.testclient import TestClient

from imgsearch.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_healthz_ready(client):
    assert client.get("/healthz").status_code == 200


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
