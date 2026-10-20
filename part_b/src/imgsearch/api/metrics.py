"""Số đo vận hành cho /metrics: đếm trong bộ nhớ của tiến trình, mất khi khởi động lại.

Latency lấy từ chính các bước của Server-Timing (PLAN 4.6). Phân vị tính trên `window` request
gần nhất; histogram và bộ đếm tính từ lúc khởi động.
"""

import threading
import time
from collections import Counter, deque

from imgsearch.api.schemas import SearchResponse

BUCKETS_MS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000)
_LOWER = (0, *BUCKETS_MS[:-1])
_BUCKET_LABELS = [
    *(f"{lo}-{hi}" for lo, hi in zip(_LOWER, BUCKETS_MS, strict=True)),
    f">{BUCKETS_MS[-1]}",
]


def _quantile(sorted_ms: list[float], q: float) -> float | None:
    if not sorted_ms:
        return None
    return sorted_ms[min(len(sorted_ms) - 1, int(q * len(sorted_ms)))]


class Metrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.errors = 0
        self.empty = 0  # không nhóm nào có ảnh (tên lạ, không bằng chứng quan hệ, P1)
        self.with_messages = 0
        self.by_type: Counter[str] = Counter()
        self.buckets = [0] * (len(BUCKETS_MS) + 1)  # phần tử cuối: > 5000 ms
        self.step_sum_ms: Counter[str] = Counter()
        self.recent: deque[float] = deque(maxlen=window)

    def observe(self, result: SearchResponse, timings: dict[str, float]) -> None:
        total = timings.get("total", 0.0)
        idx = next((i for i, b in enumerate(BUCKETS_MS) if total <= b), len(BUCKETS_MS))
        with self._lock:
            self.requests += 1
            self.by_type[result.query_type] += 1
            self.empty += not any(g.hits for g in result.groups)
            self.with_messages += bool(result.messages)
            self.buckets[idx] += 1
            self.step_sum_ms.update(timings)
            self.recent.append(total)

    def observe_error(self) -> None:
        with self._lock:
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            recent = sorted(self.recent)
            n = self.requests
            return {
                "uptime_s": round(time.time() - self.started, 1),
                "requests": {
                    "total": n,
                    "errors": self.errors,
                    "empty": self.empty,
                    "with_messages": self.with_messages,
                    "by_query_type": dict(self.by_type),
                },
                "latency_ms": {
                    "window": len(recent),
                    "p50": _quantile(recent, 0.50),
                    "p95": _quantile(recent, 0.95),
                    "p99": _quantile(recent, 0.99),
                    "max": recent[-1] if recent else None,
                    # số request trong từng khoảng (không cộng dồn): "0-10", "10-25"... ">5000"
                    "histogram": dict(zip(_BUCKET_LABELS, self.buckets, strict=True)),
                    "step_avg": {k: round(v / n, 2) for k, v in self.step_sum_ms.items()},
                },
            }
