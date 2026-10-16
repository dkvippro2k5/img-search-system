"""Đọc bench/results/summary.json của k6 và sinh reports/perf.md.

TODO: ghi kèm phần cứng (CPU, RAM, OS, Docker/WSL2, giới hạn container), p50/p95/p99 theo
mức VU và loại truy vấn, QPS, tỷ lệ lỗi, QPS tối đa mà p95 < 3 s.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # part_b/
SUMMARY = ROOT / "bench" / "results" / "summary.json"
REPORT = ROOT.parent / "reports" / "perf.md"


def main() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    dur = summary["metrics"]["http_req_duration"]
    lines = [
        "# Báo cáo hiệu năng",
        "",
        "| Chỉ số | Giá trị (ms) |",
        "|---|---|",
        *(f"| {k} | {v:.1f} |" for k, v in dur.items() if isinstance(v, (int, float))),
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {REPORT}")


if __name__ == "__main__":
    main()
