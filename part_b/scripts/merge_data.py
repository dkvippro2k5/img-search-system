"""Bước 1: hợp nhất CSV ảnh người đã duyệt -> thư mục JSONL (định dạng ở imgsearch/data/dataset.py).

    python scripts/merge_data.py        # ../data/raw/people-*.csv -> ../data/merged/people
    python scripts/build_index.py --input ../data/merged/people --version people
Đọc nhật ký tải ảnh (scripts/fetch_missing_images.py) nếu có, để điền sha256 cho dòng thiếu.
"""

import argparse
import json
import time
from pathlib import Path

from imgsearch.config import get_settings
from imgsearch.data.dataset import write_jsonl_dir
from imgsearch.data.fetch import read_log
from imgsearch.data.merge import RAW_GLOB, merge, raw_files, read_raw


def main(argv: list[str] | None = None) -> None:
    data = get_settings().data_dir
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, default=data / "raw")
    ap.add_argument("--glob", default=RAW_GLOB)
    ap.add_argument("--fetch-log", type=Path, default=data / "raw" / "fetched" / "fetch_log.jsonl")
    ap.add_argument("--out", type=Path, default=data / "merged" / "people")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    files = raw_files(args.raw, args.glob)
    if not files:
        raise SystemExit(f"không có file {args.glob} trong {args.raw}")
    ds, report = merge(read_raw(files), read_log(args.fetch_log))
    write_jsonl_dir(ds, args.out)
    report.update(
        files=[f.name for f in files], out=str(args.out), seconds=round(time.perf_counter() - t0, 1)
    )
    (args.out / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
