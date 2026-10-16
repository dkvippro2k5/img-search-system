import hashlib
from pathlib import Path

import pytest

from imgsearch.data.build_meta import build_index
from imgsearch.retrieval.store import MetaStore

MINI = Path(__file__).parent / "fixtures" / "mini"


def img(key: str) -> str:
    """image_id của ảnh `key` trong bộ dữ liệu giả mini."""
    return hashlib.sha256(f"mini:{key}".encode()).hexdigest()


@pytest.fixture(scope="session")
def mini_index(tmp_path_factory) -> tuple[Path, dict]:
    """Dựng index từ bộ mini một lần cho cả phiên test. Trả (thư mục gốc index, manifest)."""
    out = tmp_path_factory.mktemp("index")
    manifest = build_index(MINI, out, "test")
    return out, manifest


@pytest.fixture(scope="session")
def mini_store(mini_index) -> MetaStore:
    out, _ = mini_index
    return MetaStore(out / "vtest" / "meta.sqlite")
