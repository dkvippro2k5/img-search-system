"""Tải ảnh thiếu sha256 trong data/raw/people-*.csv -> data/raw/fetched/ (chạy lại được).

    python scripts/fetch_missing_images.py                 # tải
    python scripts/fetch_missing_images.py --dry-run       # chỉ đếm
Sau đó chạy scripts/merge_data.py: nó đọc nhật ký để điền sha256.
"""

import argparse
import json
import time
from pathlib import Path

from imgsearch.config import get_settings
from imgsearch.data.fetch import fetch_all, select_targets
from imgsearch.data.merge import RAW_GLOB, raw_files, read_raw


def main(argv: list[str] | None = None) -> None:
    data = get_settings().data_dir
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=data / "raw")
    ap.add_argument("--glob", default=RAW_GLOB)
    ap.add_argument("--out", type=Path, default=data / "raw" / "fetched")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="chỉ tải N URL đầu (thử)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files = raw_files(args.raw, args.glob)
    rows = read_raw(files)
    targets = select_targets(rows)[: args.limit]
    print(f"{len(files)} file, {len(rows)} dòng, {len(targets)} URL cần tải")
    if args.dry_run:
        return

    import requests  # extra "index"

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    t0, n = time.perf_counter(), 0

    def progress(_res) -> None:
        nonlocal n
        n += 1
        if n % 200 == 0:
            print(f"  {n}/{len(targets)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    counts = fetch_all(
        targets,
        args.out / "fetch_log.jsonl",
        args.out / "images",
        session.get,
        workers=args.workers,
        on_result=progress,
    )
    counts["seconds"] = round(time.perf_counter() - t0, 1)
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
