"""Bản public (X4): mật khẩu dùng chung, giới hạn tần suất."""

import base64

import pytest
from fastapi.testclient import TestClient

from imgsearch.api import main
from imgsearch.api.access import RateLimiter
from imgsearch.config import get_settings


def basic(password: str, user: str = "hoidong") -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


@pytest.fixture(scope="module")
def client(mini_index):
    out, _ = mini_index
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INDEX_DIR", str(out))
        mp.setenv("INDEX_VERSION", "test")
        mp.setenv("DEMO_PASSWORD", "mật-khẩu")
        get_settings.cache_clear()
        with TestClient(main.app) as c:
            yield c
    get_settings.cache_clear()


def test_password_required_except_healthz(client):
    assert client.get("/healthz").status_code == 200  # healthcheck của Docker không có mật khẩu
    r = client.get("/")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    for wrong in ({}, basic("sai"), {"Authorization": "Basic !!!"}):
        assert client.post("/search", json={"query": "Đen Vâu"}, headers=wrong).status_code == 401
    ok = client.post("/search", json={"query": "Đen Vâu"}, headers=basic("mật-khẩu"))
    assert ok.status_code == 200 and ok.json()["groups"]
    assert client.get("/metrics", headers=basic("mật-khẩu")).status_code == 200


def test_rate_limit_per_ip(client, monkeypatch):
    monkeypatch.setattr(main, "_limiter", RateLimiter(2))
    auth = basic("mật-khẩu")
    send = lambda ip: client.post(  # noqa: E731
        "/search", json={"query": "Đen Vâu"}, headers={**auth, "CF-Connecting-IP": ip}
    )
    assert [send("1.1.1.1").status_code for _ in range(3)] == [200, 200, 429]
    assert send("2.2.2.2").status_code == 200  # IP khác không bị ảnh hưởng


def test_rate_limiter_window():
    lim = RateLimiter(1)
    assert lim.allow("a", now=0) and not lim.allow("a", now=30) and lim.allow("a", now=61)
