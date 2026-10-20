"""Tải ảnh thiếu sha256 (data/fetch.py), với hàm get giả: không cần mạng."""

import hashlib
import io
import json

import pytest

from imgsearch.data.fetch import (
    ascii_url,
    fetch_all,
    fetch_one,
    image_files,
    read_log,
    select_all_targets,
    select_targets,
)

PIL = pytest.importorskip("PIL.Image")


def png_bytes(w=3, h=2) -> bytes:
    buf = io.BytesIO()
    PIL.new("RGB", (w, h)).save(buf, "PNG")
    return buf.getvalue()


class Resp:
    def __init__(self, status=200, body=b"", ctype="image/png"):
        self.status_code, self._body, self.headers = status, body, {"Content-Type": ctype}

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i : i + n]

    def close(self):
        pass


class FakeGet:
    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, url, headers, timeout, stream):
        self.calls.append((url, headers))
        r = self.table[url]
        if isinstance(r, Exception):
            raise r
        return r


class ReadTimeout(Exception):  # tên giống requests.ReadTimeout
    pass


def test_select_targets_skips_known_and_duplicate_urls():
    rows = [
        {"image_url": "a", "sha256": "s1"},
        {"image_url": "a", "sha256": ""},  # cùng URL với ảnh đã có sha: merge tự điền
        {"image_url": "b", "sha256": "", "source_url": "page"},
        {"image_url": "b", "sha256": ""},
        {"image_url": "", "sha256": ""},
    ]
    assert select_targets(rows) == [("b", "page")]


def test_select_all_targets_dedupes_urls():
    rows = [
        {"image_url": "a", "sha256": "s1", "source_url": "p"},
        {"image_url": "a", "sha256": ""},
        {"image_url": "b", "sha256": "s2"},
        {"image_url": "", "sha256": "s3"},
    ]
    assert select_all_targets(rows) == [("a", "p"), ("b", None)]


def test_fetch_ok_saves_by_content_hash(tmp_path):
    body = png_bytes(3, 2)
    get = FakeGet({"https://x.vn/Kh%C3%A1nh.png": Resp(body=body)})
    res = fetch_one("https://x.vn/Khánh.png", "https://báo.vn/bài", tmp_path, get)
    sha = hashlib.sha256(body).hexdigest()
    assert (res.status, res.sha256, res.width, res.height) == ("ok", sha, 3, 2)
    assert (tmp_path / sha[:2] / f"{sha}.png").read_bytes() == body
    url, headers = get.calls[0]
    assert url == "https://x.vn/Kh%C3%A1nh.png"  # header HTTP chỉ nhận latin-1
    assert headers["Referer"].isascii()


@pytest.mark.parametrize(
    ("resp", "status"),
    [
        (Resp(404), "http_404"),
        (Resp(body=b"<html>", ctype="text/html"), "not_image"),
        (Resp(body=b"not an image", ctype="image/jpeg"), "not_image"),
        (ReadTimeout("slow"), "timeout"),
        (ConnectionError("dns"), "error"),
    ],
)
def test_fetch_failures(tmp_path, resp, status):
    res = fetch_one("http://u", None, tmp_path, FakeGet({"http://u": resp}))
    assert res.status == status and res.sha256 is None


def test_fetch_all_resumes_and_retries_transient(tmp_path):
    log = tmp_path / "log.jsonl"
    body = png_bytes()
    urls = [("http://ok", None), ("http://gone", None), ("http://slow", None)]
    first = FakeGet(
        {"http://ok": Resp(body=body), "http://gone": Resp(404), "http://slow": ReadTimeout("x")}
    )
    counts = fetch_all(urls, log, tmp_path, first, 2)
    assert counts == {"skipped_done": 0, "ok": 1, "http_404": 1, "timeout": 1}

    second = FakeGet({"http://slow": Resp(body=body)})  # chỉ URL lỗi tạm được thử lại
    counts = fetch_all(urls, log, tmp_path, second, 2)
    assert counts == {"skipped_done": 2, "ok": 1}
    assert [u for u, _ in second.calls] == ["http://slow"]
    assert read_log(log)["http://slow"]["status"] == "ok"
    assert len(log.read_text(encoding="utf-8").splitlines()) == 4
    assert all(json.loads(line)["fetched_at"] for line in log.read_text().splitlines())


def test_relative_url_is_final_failure(tmp_path):
    res = fetch_one("/api/entities/local-image/26377e2d8f44a938.jpg", None, tmp_path, FakeGet({}))
    assert res.status == "not_http" and res.final


def test_ascii_url_keeps_reserved_chars():
    assert ascii_url("https://a.vn/x?q=1&b=%20") == "https://a.vn/x?q=1&b=%20"


def test_image_files_maps_ids_to_downloaded_files():
    images = [
        {"image_id": "s1", "image_url": "u1"},
        {"image_id": "s2", "image_url": "u2"},
        {"image_id": "s3", "image_url": None},
    ]
    log = {"u1": {"status": "ok", "path": "p1"}, "u2": {"status": "http_404", "path": None}}
    assert image_files(images, log) == [("s1", "p1")]
