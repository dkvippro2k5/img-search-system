"""Chạy benchmark HTTP đầy đủ (PLAN 6) rồi sinh ../reports/perf.md. Thay cho `make bench`.

python bench/run_bench.py                      # nocache (chính) rồi cache
python bench/run_bench.py --modes nocache      # chỉ một chế độ
python bench/make_report.py                    # chỉ dựng lại báo cáo từ bench/results/

Mỗi chế độ: dựng lại container api (QUERY_CACHE_SIZE=0 khi nocache), đo thời gian tới khi
/healthz 200 (R3), chạy k6 (compose profile bench, 1/4/8/16 VU x 60 s) trong lúc lấy mẫu
`docker stats` của api, rồi chuyển kết quả vào bench/results/<chế độ>/.
Cần: Docker Desktop đang chạy, `docker compose up -d --build` đã chạy được một lần (MinIO có
index), part_b/.env có INDEX_VERSION.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # part_b/
RESULTS = ROOT / "bench" / "results"
HEALTH = "http://localhost:8000/healthz"


def compose(*args: str, env: dict | None = None, check: bool = True) -> str:
    out = subprocess.run(
        ["docker", "compose", *args],
        cwd=ROOT,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if check and out.returncode != 0:
        raise SystemExit(f"docker compose {' '.join(args)} lỗi:\n{out.stderr[-2000:]}")
    return out.stdout


def wait_healthy(timeout: float = 600) -> dict:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            with urllib.request.urlopen(HEALTH, timeout=2) as r:
                if r.status == 200:
                    return {"seconds": round(time.perf_counter() - t0, 1), **json.load(r)}
        except OSError:
            pass
        time.sleep(0.5)
    raise SystemExit(f"api không healthy sau {timeout} s (xem: docker compose logs api)")


def container_id() -> str:
    return compose("ps", "-q", "api").strip()


class StatsSampler(threading.Thread):
    """`docker stats --no-stream` của container api mỗi ~1 s -> danh sách mẫu."""

    def __init__(self, cid: str):
        super().__init__(daemon=True)
        self.cid, self.samples, self.stop = cid, [], threading.Event()

    def run(self):
        fmt = "{{.CPUPerc}};{{.MemUsage}}"
        while not self.stop.is_set():
            out = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", fmt, self.cid],
                capture_output=True,
                text=True,
            ).stdout.strip()
            if out:
                cpu, mem = out.split(";")
                self.samples.append({"t": time.time(), "cpu": cpu, "mem": mem.split("/")[0]})
            self.stop.wait(1.0)


def run_mode(mode: str, extra: dict[str, str] | None = None, name: str | None = None) -> None:
    env = {"QUERY_CACHE_SIZE": "0" if mode == "nocache" else "1024", **(extra or {})}
    print(f"[{mode}] dựng lại api (QUERY_CACHE_SIZE={env['QUERY_CACHE_SIZE']})")
    compose("up", "-d", "--force-recreate", "--no-deps", "api", env=env)
    health = wait_healthy()
    print(f"[{mode}] healthy sau {health['seconds']} s: {health}")

    out_dir = RESULTS / (name or mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "raw.json"):
        (RESULTS / name).unlink(missing_ok=True)
    sampler = StatsSampler(container_id())
    sampler.start()
    t0 = time.time()
    print(f"[{mode}] k6 chạy khoảng 5 phút...")
    log = compose("--profile", "bench", "run", "--rm", "bench", env=env, check=False)
    sampler.stop.set()
    sampler.join()
    # Lời gọi mentor thất bại (429 do giới hạn 100 request/phút mỗi key, timeout): engine bỏ
    # kênh văn bản cho request đó, nên số đo của kênh này phụ thuộc máy mentor.
    api_log = compose("logs", "api", "--no-log-prefix", check=False)
    mentor_errors = {
        "rate_limited": api_log.count("HTTP 429"),
        "total": api_log.count("Embedding mentor lỗi"),
    }
    for name in ("summary.json", "raw.json"):
        if not (RESULTS / name).exists():
            raise SystemExit(f"k6 không ghi {name}:\n{log[-3000:]}")
        shutil.move(RESULTS / name, out_dir / name)
    (out_dir / "k6.log").write_text(log, encoding="utf-8")
    (out_dir / "stats.json").write_text(json.dumps(sampler.samples), encoding="utf-8")
    (out_dir / "run.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "started": t0,
                "health": health,
                "env": env,
                "mentor_errors": mentor_errors,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[{mode}] xong -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", default="nocache,cache")
    ap.add_argument("--env", action="append", default=[], help="KEY=VAL cho api, lặp được")
    ap.add_argument("--name", default=None, help="hậu tố thư mục kết quả, ví dụ c1")
    args = ap.parse_args(argv)
    extra = dict(e.split("=", 1) for e in args.env)
    for mode in args.modes.split(","):
        run_mode(mode, extra, f"{mode}-{args.name}" if args.name else None)
    subprocess.run(
        [sys.executable, str(ROOT / "bench" / "make_report.py")],
        check=True,
        env={**os.environ, "PYTHONUTF8": "1"},
    )


if __name__ == "__main__":
    main()
