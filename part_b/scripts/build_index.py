"""Dựng index -> {out}/v{version}/: meta.sqlite (metadata, KG), lexical.npz (BM25), manifest.json.

Đầu vào là thư mục JSONL (định dạng ở imgsearch/data/dataset.py), mặc định ../data/merged.
Thử với dữ liệu giả:
    python scripts/build_index.py --input tests/fixtures/mini --version dev
"""

import argparse
import json
from pathlib import Path

from imgsearch.config import get_settings
from imgsearch.data.build_meta import build_index


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=settings.data_dir / "merged")
    ap.add_argument("--out", type=Path, default=settings.index_dir)
    ap.add_argument("--version", default=settings.index_version)
    ap.add_argument(
        "--features",
        type=Path,
        default=settings.data_dir / "features",
        help="thư mục có image_embeddings.npy/.ids.txt/.json (không có thì bỏ qua dense)",
    )
    args = ap.parse_args(argv)

    manifest = build_index(args.input, args.out, args.version, args.features)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
