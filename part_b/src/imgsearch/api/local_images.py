"""Ảnh đã tải trên máy cho demo cục bộ (LOCAL_IMAGES=true): image_id -> file trong
data/raw/fetched/images/, tra qua nhật ký tải theo image_url (data/fetch.py).

Khóa theo URL chứ không theo tên file: tên file là sha256 của nội dung tải về, có thể khác
sha256 trong CSV khi CDN nén lại ảnh.
"""

import logging
from pathlib import Path

from imgsearch.data.fetch import read_log
from imgsearch.retrieval.store import MetaStore

log = logging.getLogger(__name__)


class LocalImages:
    def __init__(self, data_dir: Path, store: MetaStore):
        fetched = Path(data_dir) / "raw" / "fetched"
        images_dir = fetched / "images"
        fetch_log = read_log(fetched / "fetch_log.jsonl")
        self.files: dict[str, Path] = {}
        self.urls: dict[str, str] = {}
        for image_id, url in store.conn.execute("SELECT image_id, image_url FROM image"):
            self.urls[image_id] = url or ""
            rec = fetch_log.get(url or "")
            if rec and rec.get("status") == "ok" and rec.get("path"):
                # đường dẫn trong nhật ký tương đối với cwd lúc tải: chỉ lấy tên file
                name = rec["path"].replace("\\", "/").rsplit("/", 1)[-1]
                self.files[image_id] = images_dir / name[:2] / name
        log.info("Ảnh cục bộ: %d/%d ảnh có file", len(self.files), len(self.urls))

    def path(self, image_id: str) -> Path | None:
        p = self.files.get(image_id)
        return p if p is not None and p.is_file() else None

    def url(self, image_id: str) -> str | None:
        u = self.urls.get(image_id)
        return u if u and u.startswith(("http://", "https://")) else None
