"""Đọc bench/results/<chế độ>/ (bench/run_bench.py) và sinh ../reports/perf.md (PLAN 6).

Mỗi chế độ (nocache: tắt cache vector truy vấn, số chính; cache: có cache): p50/p95/p99 theo
mức VU và theo loại truy vấn, QPS, tỷ lệ lỗi, QPS tối đa mà p95 < 3 s, thời gian xử lý trong
server (Server-Timing), CPU/RAM của container, thời gian khởi động tới healthy. Kèm phần cứng.
Không có cache kết quả ở API: "cache" chỉ là cache vector truy vấn dense.
"""

import json
import os
import platform
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # part_b/
RESULTS = ROOT / "bench" / "results"
REPORT = ROOT.parent / "reports" / "perf.md"
STAGES = ("vu1", "vu4", "vu8", "vu16")
STAGE_SECONDS = 60
SLO_MS = 3000
QTYPES = ("single", "multi_intent", "multi_hop")


def pct(values: list[float], p: float) -> float:
    """Phân vị kiểu nearest-rank."""
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, max(0, round(p / 100 * len(s) + 0.5) - 1))]


def load_points(path: Path) -> dict:
    """raw.json của k6 (JSON lines) -> {(metric, scenario, qtype): [giá trị]}."""
    out = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            if '"Point"' not in line:
                continue
            r = json.loads(line)
            m = r["metric"]
            if m not in ("http_req_duration", "http_req_failed", "server_ms"):
                continue
            tags = r["data"].get("tags") or {}
            scen = tags.get("scenario", "")
            if scen not in STAGES:
                continue  # bỏ làm nóng
            v = r["data"]["value"]
            out[(m, scen, "all")].append(v)
            out[(m, scen, tags.get("qtype", "?"))].append(v)
    return out


def _mem_mb(s: str) -> float:
    m = re.match(r"([\d.]+)\s*([KMG]i?B)", s.strip())
    if not m:
        return float("nan")
    n, unit = float(m.group(1)), m.group(2)[0]
    return n * {"K": 1 / 1024, "M": 1, "G": 1024}[unit]


def hardware() -> list[str]:
    cpu = platform.processor()
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    ram = ""
    if platform.system() == "Windows":
        import ctypes

        class MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                *[(f"f{i}", ctypes.c_ulonglong) for i in range(6)],
            ]

        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            ram = f"{st.ullTotalPhys / 2**30:.1f} GB"
    lines = [
        f"- Máy: {cpu}, {os.cpu_count()} luồng logic, RAM {ram or '?'}, "
        f"{platform.system()} {platform.release()} ({platform.version()})"
    ]
    info = subprocess.run(
        [
            "docker",
            "info",
            "--format",
            "{{.ServerVersion}};{{.OperatingSystem}};{{.KernelVersion}};{{.NCPU}};{{.MemTotal}}",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if info and not info.startswith(";"):
        ver, osname, kernel, ncpu, mem = info.split(";")
        lines.append(
            f"- Docker {ver} trên {osname} (kernel {kernel}): VM có {ncpu} CPU, "
            f"{int(mem) / 2**30:.1f} GB RAM"
        )
    lines.append(
        "- Container api: giới hạn 4 CPU, 4 GB RAM (docker-compose.yml), ONNX Runtime 4 luồng, "
        "uvicorn 1 worker; k6 chạy trong container cùng máy (cạnh tranh CPU với api)."
    )
    return lines


def mode_section(mode: str, d: Path) -> tuple[list[str], dict]:
    pts = load_points(d / "raw.json")
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    stats = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    rows, best = [], None
    lines = [
        f"### Chế độ `{mode}`",
        "",
        f"Khởi động tới khi /healthz 200 (tải index từ MinIO + nạp model): "
        f"**{run['health']['seconds']} s** [Đã đo].",
        "",
        "| VU | Số request | QPS | Lỗi | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) "
        "| Server p50 | Server p95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for st in STAGES:
        dur = pts[("http_req_duration", st, "all")]
        if not dur:
            continue
        fail = pts[("http_req_failed", st, "all")]
        srv = pts[("server_ms", st, "all")]
        qps = len(dur) / STAGE_SECONDS
        p95 = pct(dur, 95)
        err = sum(fail) / len(fail) if fail else 0.0
        rows.append({"stage": st, "qps": qps, "p95": p95, "err": err})
        if p95 < SLO_MS and err < 0.01:
            best = max(best or 0.0, qps)
        lines.append(
            f"| {st[2:]} | {len(dur)} | {qps:.1f} | {100 * err:.2f}% | {pct(dur, 50):.0f} | "
            f"{p95:.0f} | {pct(dur, 99):.0f} | {max(dur):.0f} | {pct(srv, 50):.0f} | "
            f"{pct(srv, 95):.0f} |"
        )
    lines += [
        "",
        "p95 theo loại truy vấn (ms):",
        "",
        "| VU | " + " | ".join(QTYPES) + " |",
        "|---|" + "---|" * len(QTYPES),
    ]
    for st in STAGES:
        if pts[("http_req_duration", st, "all")]:
            cells = [f"{pct(pts[('http_req_duration', st, q)], 95):.0f}" for q in QTYPES]
            lines.append(f"| {st[2:]} | " + " | ".join(cells) + " |")
    cpu = [float(s["cpu"].rstrip("%")) for s in stats if s["cpu"].rstrip("%")]
    mem = [_mem_mb(s["mem"]) for s in stats]
    errs = run.get("mentor_errors")
    if errs is not None and run["health"].get("text_dense"):
        lines += [
            "",
            f"Lời gọi mentor thất bại trong lần đo này: {errs['total']} (trong đó "
            f"{errs['rate_limited']} lần HTTP 429 do giới hạn request); request đó chạy không "
            "có kênh văn bản [Đã đo].",
        ]
    if cpu:
        lines += [
            "",
            f"Container api trong lúc chạy k6: CPU trung bình {statistics.mean(cpu):.0f}%, "
            f"cao nhất {max(cpu):.0f}% (100% = 1 lõi; giới hạn 400%); RAM cao nhất "
            f"{max(mem):.0f} MB [Đã đo].",
        ]
    return lines, {"best_qps": best, "rows": rows}


def main() -> None:
    modes = sorted(
        (d.name for d in RESULTS.iterdir() if (d / "raw.json").exists()),
        key=lambda m: (m.split("-")[0] != "nocache", m),
    )
    if not modes:
        raise SystemExit(f"Chưa có kết quả trong {RESULTS}: chạy bench/run_bench.py")
    body, summary = [], {}
    for m in modes:
        lines, s = mode_section(m, RESULTS / m)
        body += lines + [""]
        summary[m] = s
    head = [
        "# Báo cáo hiệu năng",
        "",
        "Sinh tự động bởi `part_b/bench/make_report.py` từ kết quả k6 (`bench/run_bench.py`). "
        "Mọi số trong file là **[Đã đo]** trên phần cứng ghi dưới đây.",
        "",
        "## Phần cứng và cấu hình",
        "",
        *hardware(),
        "",
        "## Kịch bản",
        "",
        "- `POST /search` qua HTTP, top_k mặc định (30 ảnh mỗi nhóm), lọc an toàn bật.",
        "- 100 truy vấn lấy từ test set (`testsets/bench_queries.json`): 50% single, 30% "
        "multi-intent, 20% multi-hop; mỗi request chọn ngẫu nhiên một truy vấn.",
        "- Làm nóng 50 request (không tính), rồi 1, 4, 8, 16 VU, mỗi mức 60 s, VU gửi liên tục "
        "không nghỉ. Phân vị tính trên mọi request của mức đó.",
        "- `nocache`: tắt cache vector truy vấn dense (`QUERY_CACHE_SIZE=0`), mỗi request đều "
        "chạy text encoder ONNX: đây là số chính. `cache`: bật cache (truy vấn lặp lại). API "
        "không có cache kết quả.",
        "- Hậu tố `-cN`: `SEARCH_CONCURRENCY=N` (số truy vấn được tính toán cùng lúc, còn lại "
        "xếp hàng; 0 = không giới hạn). Không hậu tố = mặc định của docker-compose (2).",
        "- Không hậu tố: cấu hình mặc định, **có** kênh dense văn bản qwen3 qua API mentor (trọng "
        "số 0,5, mỗi truy vấn có phần mô tả gọi mạng một lần). `-notext`: tắt kênh này "
        "(`TEXT_DENSE_ENABLED=false`); các biến thể `-cN` cũng đo khi chưa có kênh văn bản.",
        "- Kênh văn bản gọi API của mentor qua đường hầm: key bị giới hạn **100 request/phút** "
        "(header `x-litellm-key-rpm-limit`, đo 20/10), vượt thì HTTP 429 và engine bỏ kênh cho "
        "request đó (HTTP vẫn 200). Latency kênh này phụ thuộc tải máy mentor, nên hai lần đo "
        "liên tiếp có thể chênh nhau; dưới tải lớn, chất lượng thực tế gần với `-notext`.",
        "- Server p50/p95: thời gian trong server theo header `Server-Timing` total (gồm thời gian "
        "chờ semaphore, không gồm mạng).",
        "",
        "## Tóm tắt",
        "",
    ]
    for m, s in summary.items():
        best, rows = s["best_qps"], s["rows"]
        ok = [r for r in rows if r["p95"] < SLO_MS and r["err"] < 0.01]
        at = max(ok, key=lambda r: r["qps"])["stage"][2:] if ok else None
        peak = max(rows, key=lambda r: r["qps"]) if rows else None
        last = rows[-1] if rows else None
        if peak is None:
            trend = ""
        elif last["qps"] < 0.85 * peak["qps"]:
            trend = (
                f"Thông lượng giảm khi tăng tải ({peak['qps']:.1f} QPS ở {peak['stage'][2:]} VU "
                f"-> {last['qps']:.1f} ở {last['stage'][2:]} VU): các request tranh luồng CPU."
            )
        elif peak is last:
            trend = "QPS còn tăng ở mức VU cao nhất đã thử: chưa chạm trần."
        else:
            trend = (
                f"Thông lượng giữ ổn định khi tăng tải ({last['qps']:.1f} QPS ở "
                f"{last['stage'][2:]} VU): đã bão hòa CPU, thêm VU chỉ tăng thời gian xếp hàng."
            )
        head.append(
            f"- `{m}`: QPS cao nhất mà p95 < 3 s và lỗi < 1%: "
            + (f"**{best:.1f}** (ở {at} VU)" if best is not None else "không đạt ở mức nào")
            + f" [Đã đo]. {trend}"
        )
    head += ["", "## Kết quả", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(head + body) + "\n", encoding="utf-8")
    print(f"Đã ghi {REPORT}")


if __name__ == "__main__":
    main()
