"""Pooling cho truy vấn cần chấm tay (PLAN 5.1): gộp top-k của nhiều cấu hình hệ thống.

Cấu hình chỉ khác ở kênh dense; cùng một engine, đổi tại chỗ (engine không cache kết quả).
"""

from collections.abc import Iterator
from contextlib import contextmanager

from imgsearch.api.schemas import SearchRequest
from imgsearch.eval.run_eval import flatten
from imgsearch.retrieval.engine import SearchEngine

# tên -> trọng số dense (None: tắt dense, chỉ BM25). w8 gần như chỉ dense.
CONFIGS: dict[str, float | None] = {"bm25": None, "w2": 2.0, "w8": 8.0}


@contextmanager
def config(engine: SearchEngine, name: str) -> Iterator[SearchEngine]:
    weight = CONFIGS[name]
    dense, settings = engine.dense, engine.settings
    if weight is None:
        engine.dense = None
    else:
        if dense is None:
            raise RuntimeError(f"cấu hình {name} cần dense nhưng engine không có dense")
        engine.settings = settings.model_copy(update={"dense_weight": weight})
    try:
        yield engine
    finally:
        engine.dense, engine.settings = dense, settings


def ranked(engine: SearchEngine, query: str, k: int) -> list[str]:
    """sha256 đầy đủ, xen kẽ các nhóm như lưới hiển thị, cắt k."""
    resp, _ = engine.search(SearchRequest(query=query, top_k=k))
    order = flatten(resp, short=None)
    return order[:k]


def pool(engine: SearchEngine, query: str, depth: int = 20) -> dict[str, dict[str, int]]:
    """image_id -> {cấu hình: hạng (1..depth)}."""
    out: dict[str, dict[str, int]] = {}
    for name in CONFIGS:
        with config(engine, name):
            for rank, image_id in enumerate(ranked(engine, query, depth), start=1):
                out.setdefault(image_id, {})[name] = rank
    return out
