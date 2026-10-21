import shutil
from urllib.parse import urlparse

import pytest

from conftest import img
from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.data import thumbs
from imgsearch.retrieval.engine import SearchEngine
from imgsearch.storage import s3 as s3mod

DEN1, DEN2 = img("den1"), img("den2")


def test_make_thumb_resizes_and_skips_broken(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    src = tmp_path / "a.jpg"
    Image.new("RGB", (1200, 800), "red").save(src)
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    out = tmp_path / "thumbs"
    files = {DEN1: src, DEN2: tmp_path / "broken.jpg"}
    assert thumbs.build_thumbs(files, out, workers=2) == [DEN1]
    with Image.open(thumbs.thumb_path(out, DEN1)) as im:
        assert im.format == "WEBP" and max(im.size) == thumbs.THUMB_SIZE
    src.unlink()  # đã có thumbnail: chạy lại không cần file nguồn
    assert thumbs.build_thumbs({DEN1: src}, out) == [DEN1]


def test_thumb_list_roundtrip(tmp_path):
    thumbs.write_list(tmp_path / "t.txt", [DEN1, DEN2])
    assert thumbs.read_list(tmp_path / "t.txt") == {DEN1, DEN2}
    assert thumbs.read_list(tmp_path / "missing.txt") == set()


class FakeS3:
    def __init__(self, have):
        self.have, self.uploads = have, []

    def get_paginator(self, name):
        contents = [{"Key": k, "Size": v} for k, v in self.have.items()]

        class Pager:
            def paginate(self, Bucket, Prefix):  # noqa: N803 — tên tham số của boto3
                yield {"Contents": [c for c in contents if c["Key"].startswith(Prefix)]}

        return Pager()

    def upload_file(self, filename, bucket, key, ExtraArgs=None):  # noqa: N803
        self.uploads.append((key, (ExtraArgs or {}).get("ContentType")))

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.uploads.append((Key, Body))


def test_upload_thumbs_skips_existing_same_size(tmp_path):
    for i in (DEN1, DEN2):
        p = thumbs.thumb_path(tmp_path, i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"webp")
    fake = FakeS3({s3mod.thumb_key(DEN1): 4})
    settings = Settings(_env_file=None, s3_bucket="b")
    assert s3mod.upload_thumbs(settings, tmp_path, s3=fake) == 1
    assert fake.uploads == [(s3mod.thumb_key(DEN2), "image/webp")]


def test_upload_index_lists_only_thumbs_on_s3(tmp_path):
    folder = tmp_path / "vtest"
    folder.mkdir()
    (folder / "meta.sqlite").write_bytes(b"db")
    thumbs.write_list(folder / thumbs.LIST_FILE, [DEN1, DEN2])
    fake = FakeS3({s3mod.thumb_key(DEN1): 4})  # DEN2 chưa đẩy thumbnail (lỗi giữa chừng)
    settings = Settings(_env_file=None, s3_bucket="b", index_version="test")
    assert s3mod.upload_index(settings, tmp_path, s3=fake) == 2
    assert ("index/vtest/thumbs.txt", f"{DEN1}\n".encode()) in fake.uploads


def test_presign_uses_public_endpoint():
    kw = {"s3_access_key_id": "k", "s3_secret_access_key": "s", "s3_bucket": "b"}
    s = Settings(
        _env_file=None,
        s3_endpoint_url="http://minio:9000",
        s3_public_endpoint_url="http://localhost:9000",
        **kw,
    )
    url = s3mod.presigned_thumb_url(s, DEN1, s3mod.client(s, public=True))
    assert urlparse(url).netloc == "localhost:9000" and "X-Amz-Signature" in url
    r2 = Settings(_env_file=None, s3_endpoint_url="https://acc.r2.cloudflarestorage.com", **kw)
    assert "r2.cloudflarestorage.com" in s3mod.client(r2, public=True).meta.endpoint_url


def test_engine_presigns_listed_thumbs_and_hotlinks_the_rest(mini_index, tmp_path):
    out, _ = mini_index
    shutil.copytree(out / "vtest", tmp_path / "vtest")
    thumbs.write_list(tmp_path / "vtest" / thumbs.LIST_FILE, [DEN1])
    s = Settings(
        _env_file=None,
        index_dir=tmp_path,
        index_version="test",
        thumbs_from_s3=True,
        s3_endpoint_url="http://minio:9000",
        s3_public_endpoint_url="http://localhost:9000",
        s3_access_key_id="k",
        s3_secret_access_key="s",
    )
    e = SearchEngine(s)
    e.load()
    hits = e.search(SearchRequest(query="Đen Vâu"))[0].groups[0].hits
    urls = {h.image_id: h.thumb_url for h in hits}
    assert urlparse(urls[DEN1]).netloc == "localhost:9000"
    assert urls[DEN2].startswith("https://example.com/")  # không có thumbnail: hotlink
    # ô xem ảnh dùng ảnh gốc, không phóng to thumbnail 320 px
    assert all(h.image_url and h.image_url.startswith("https://example.com/") for h in hits)
    # cùng cửa sổ nửa hạn: URL giữ nguyên để trình duyệt cache được
    again = e.search(SearchRequest(query="Đen Vâu"))[0].groups[0].hits
    assert {h.image_id: h.thumb_url for h in again}[DEN1] == urls[DEN1]
