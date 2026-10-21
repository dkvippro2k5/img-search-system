"""Tải index từ S3 (storage/s3.py) và REQUIRE_INDEX (engine.load) với client S3 giả."""

import shutil

import pytest

from imgsearch.config import Settings
from imgsearch.retrieval.engine import SearchEngine
from imgsearch.storage import s3 as s3mod


class FakeS3:
    """Đủ cho download_index: paginator list_objects_v2 + download_file, đọc từ thư mục."""

    def __init__(self, root, prefix):
        self.root, self.prefix, self.downloads = root, prefix, []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        files = sorted(p for p in self.root.iterdir() if p.is_file())
        contents = [{"Key": self.prefix + p.name, "Size": p.stat().st_size} for p in files]

        class Pager:
            def paginate(self, Bucket, Prefix):  # noqa: N803 — tên tham số của boto3
                yield {"Contents": [c for c in contents if c["Key"].startswith(Prefix)]}

        return Pager()

    def download_file(self, bucket, key, filename):
        self.downloads.append(key)
        shutil.copyfile(self.root / key[len(self.prefix) :], filename)


def test_download_index_copies_then_skips_unchanged(mini_index, tmp_path):
    out, _ = mini_index
    settings = Settings(_env_file=None, index_dir=tmp_path, index_version="test")
    fake = FakeS3(out / "vtest", s3mod.index_prefix("test"))
    n = s3mod.download_index(settings, s3=fake)
    assert (
        n == len(list((out / "vtest").iterdir())) and (tmp_path / "vtest" / "meta.sqlite").exists()
    )
    assert not list((tmp_path / "vtest").glob("*.part"))
    assert s3mod.download_index(settings, s3=fake) == 0  # đã có, cùng kích thước


def test_download_index_empty_prefix_raises(tmp_path):
    settings = Settings(_env_file=None, index_dir=tmp_path, index_version="none")
    with pytest.raises(FileNotFoundError):
        s3mod.download_index(settings, s3=FakeS3(tmp_path, "index/vother/"))


def test_engine_downloads_index_from_s3_on_load(mini_index, tmp_path, monkeypatch):
    out, _ = mini_index
    fake = FakeS3(out / "vtest", s3mod.index_prefix("test"))
    monkeypatch.setattr(s3mod, "client", lambda settings: fake)
    e = SearchEngine(
        Settings(_env_file=None, index_dir=tmp_path, index_version="test", index_from_s3=True)
    )
    e.load()
    assert e.ready and e.store is not None and fake.downloads


def test_require_index_keeps_engine_not_ready(tmp_path, monkeypatch):
    def broken(settings):
        raise ConnectionError("không có mạng")

    monkeypatch.setattr(s3mod, "client", broken)
    settings = Settings(
        _env_file=None,
        index_dir=tmp_path,
        index_version="missing",
        index_from_s3=True,
        require_index=True,
    )
    e = SearchEngine(settings)
    e.load()  # lỗi tải không làm sập khởi động
    assert not e.ready
    # không bắt buộc thì vẫn chạy với index rỗng như trước
    e2 = SearchEngine(Settings(_env_file=None, index_dir=tmp_path, index_version="missing"))
    e2.load()
    assert e2.ready and e2.store is None


def test_download_model_to_models_dir(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("text.onnx", "tokenizer.json", "encoder.json"):
        (src / name).write_bytes(b"x")
    models = tmp_path / "models"
    settings = Settings(_env_file=None, models_dir=models, dense_model="siglip2-b32")
    fake = FakeS3(src, s3mod.model_prefix("siglip2-b32"))
    assert s3mod.download_model(settings, s3=fake) == 3
    assert (models / "siglip2-b32" / "text.onnx").exists()
    assert s3mod.download_model(settings, s3=fake) == 0  # đã có, cùng kích thước


def test_engine_model_download_failure_does_not_block_start(mini_index, tmp_path, monkeypatch):
    def broken(settings):
        raise ConnectionError("không có mạng")

    monkeypatch.setattr(s3mod, "client", broken)
    out, _ = mini_index
    e = SearchEngine(
        Settings(_env_file=None, index_dir=out, index_version="test", models_from_s3=True)
    )
    e.load()
    assert e.ready and e.dense is None  # không có model: chạy BM25


def test_upload_model_skips_existing(tmp_path):
    src = tmp_path / "siglip2-b32"
    src.mkdir()
    (src / "text.onnx").write_bytes(b"changed")
    (src / "encoder.json").write_bytes(b"{}")
    prefix = s3mod.model_prefix("siglip2-b32")
    have = [{"Key": prefix + "text.onnx", "Size": 2}, {"Key": prefix + "encoder.json", "Size": 2}]

    class Fake:
        uploads: list[str] = []

        def get_paginator(self, name):
            class Pager:
                def paginate(self, Bucket, Prefix):  # noqa: N803 — tên tham số của boto3
                    yield {"Contents": have}

            return Pager()

        def upload_file(self, filename, bucket, key):
            self.uploads.append(key)

    fake = Fake()
    assert s3mod.upload_model(Settings(_env_file=None), src, s3=fake) == 1  # encoder.json đã có
    assert fake.uploads == [prefix + "text.onnx"]
