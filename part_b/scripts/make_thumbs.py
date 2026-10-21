"""Tạo thumbnail WebP cho mọi ảnh của index -> ../data/thumbs/, và danh sách ảnh có thumbnail
-> ../data/index/v{ver}/thumbs.txt (đi cùng index lên S3; engine presign ảnh có trong danh sách,
ảnh còn lại hotlink image_url).

    python -X utf8 scripts/make_thumbs.py --index-version people     # ~10 phút, 8 luồng
    python -m imgsearch.storage.s3 --src ../data/index --thumbs ../data/thumbs

Nguồn là ảnh đã tải (scripts/fetch_missing_images.py --all). Chạy lại thì bỏ thumbnail đã có.
Cần extra "index" (Pillow).
"""

import argparse
import logging
import time
from pathlib import Path

from imgsearch.api.local_images import LocalImages
from imgsearch.config import Settings
from imgsearch.data.thumbs import LIST_FILE, build_thumbs, write_list
from imgsearch.retrieval.store import MetaStore


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--data-dir", type=Path, default=Path("../data"))
    ap.add_argument("--out", type=Path, default=Path("../data/thumbs"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    settings = Settings(_env_file=None, index_dir=args.index_dir, index_version=args.index_version)
    store = MetaStore(settings.index_path / "meta.sqlite")
    local = LocalImages(args.data_dir, store)
    files = {i: p for i, p in local.files.items() if p.is_file()}
    t = time.perf_counter()
    ids = build_thumbs(files, args.out, args.workers)
    write_list(settings.index_path / LIST_FILE, ids)
    logging.info(
        "%d/%d ảnh có thumbnail (%d ảnh của index không có file nguồn), %.0f s -> %s",
        len(ids),
        len(files),
        store.count_images() - len(files),
        time.perf_counter() - t,
        settings.index_path / LIST_FILE,
    )


if __name__ == "__main__":
    main()
