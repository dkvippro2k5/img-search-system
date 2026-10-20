"""Chuyển bộ golden sang thư mục JSONL để dựng index (xem imgsearch/data/golden.py).

python scripts/golden_to_dataset.py --mode auto     # hệ thống thật: chỉ đọc metadata.json
python scripts/golden_to_dataset.py --mode oracle   # chẩn đoán: person + quan hệ từ gold/
python scripts/build_index.py --input ../data/merged/golden-auto --version golden-auto
python -m imgsearch.eval.run_eval --index-version golden-auto
"""

import argparse
import json
from pathlib import Path

from imgsearch.data.dataset import write_jsonl_dir
from imgsearch.data.golden import MODES, to_dataset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=Path("../data/golden"))
    ap.add_argument("--mode", choices=MODES, default="auto")
    ap.add_argument("--no-caption", action="store_true", help="bỏ caption (giống dữ liệu thật)")
    ap.add_argument(
        "--out", type=Path, default=None, help="mặc định ../data/merged/golden-<mode>[-nocap]"
    )
    args = ap.parse_args(argv)

    ds, report = to_dataset(args.golden, args.mode, with_caption=not args.no_caption)
    suffix = "-nocap" if args.no_caption else ""
    out = args.out or Path("../data/merged") / f"golden-{args.mode}{suffix}"
    write_jsonl_dir(ds, out)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
