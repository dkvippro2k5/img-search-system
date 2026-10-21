import json

import pytest
from fastapi.testclient import TestClient

from conftest import img
from imgsearch.api.main import app
from imgsearch.config import get_settings

DEN1, DEN2 = img("den1"), img("den2")
FILE_SHA = "ab" + "0" * 62  # sha256 nội dung tải về, khác image_id (CDN nén lại ảnh)


@pytest.fixture(scope="module")
def client(mini_index, tmp_path_factory):
    out, _ = mini_index
    data = tmp_path_factory.mktemp("data")
    fetched = data / "raw" / "fetched"
    (fetched / "images" / "ab").mkdir(parents=True)
    (fetched / "images" / "ab" / f"{FILE_SHA}.jpg").write_bytes(b"\xff\xd8jpeg")
    # đường dẫn trong nhật ký tương đối với cwd lúc tải, dấu \ của Windows
    rec = {"url": "https://example.com/img/den1.jpg", "status": "ok"}
    rec["path"] = f"..\\data\\raw\\fetched\\images\\ab\\{FILE_SHA}.jpg"
    (fetched / "fetch_log.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("INDEX_DIR", str(out))
        mp.setenv("INDEX_VERSION", "test")
        mp.setenv("DEMO_PASSWORD", "")  # .env của máy có thể đặt mật khẩu bản public
        mp.setenv("DATA_DIR", str(data))
        mp.setenv("LOCAL_IMAGES", "true")
        get_settings.cache_clear()
        with TestClient(app) as c:
            yield c
    get_settings.cache_clear()


def test_thumb_urls_point_to_local_endpoint(client):
    body = client.post("/search", json={"query": "Đen Vâu"}).json()
    assert {h["thumb_url"] for h in body["groups"][0]["hits"]} >= {f"/img/{DEN1}"}


def test_local_image_served_from_disk_else_redirect(client):
    resp = client.get(f"/img/{DEN1}")
    assert resp.status_code == 200 and resp.content == b"\xff\xd8jpeg"
    resp = client.get(f"/img/{DEN2}", follow_redirects=False)  # chưa tải: về image_url gốc
    assert resp.status_code == 302 and resp.headers["location"].startswith("https://example.com/")


@pytest.mark.parametrize("bad", ["../../etc/passwd", "abc", "f" * 64])
def test_local_image_rejects_bad_or_unknown_ids(client, bad):
    assert client.get(f"/img/{bad}", follow_redirects=False).status_code == 404
