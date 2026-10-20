"""Pooling cho truy vấn cần chấm tay (PLAN 5.1): gộp top-k của nhiều cấu hình hệ thống.

Cấu hình chỉ khác ở kênh dense; cùng một engine, đổi tại chỗ (engine không cache kết quả).
"""

from collections.abc import Iterator
from contextlib import contextmanager

from imgsearch.api.schemas import SearchRequest
from imgsearch.config import Settings
from imgsearch.eval.run_eval import flatten
from imgsearch.retrieval.engine import SearchEngine

# tên -> (trọng số dense ảnh, trọng số dense văn bản); None = tắt kênh đó. BM25 luôn bật (1).
# w8 gần như chỉ ảnh; "+t0.5": thêm kênh văn bản qwen3 của mentor trọng số 0,5 (P3: dev
# tốt nhất trong 0,5 / 1 / 2; golden thì 2 tốt nhất).
CONFIGS: dict[str, tuple[float | None, float | None]] = {
    "bm25": (None, None),
    "w2": (2.0, None),
    "w8": (8.0, None),
    "w2+t0.5": (2.0, 0.5),
}
MENTOR_FIELDS = (
    "mentor_api_key",
    "text_embed_url",
    "text_embed_model",
    "rerank_url",
    "rerank_model",
    "mentor_timeout_seconds",
)


def eval_settings(**kw) -> Settings:
    """Settings cho chấm/pooling: bỏ qua .env (kết quả không phụ thuộc cấu hình máy) trừ phần
    dịch vụ mentor (key, URL) để kênh văn bản chạy được."""
    env = Settings()
    mentor = {f: getattr(env, f) for f in MENTOR_FIELDS}
    # chấm luôn nạp kênh văn bản nếu có (để so cấu hình), dù mặc định runtime tắt
    return Settings(_env_file=None, text_dense_enabled=True, **mentor, **kw)


def available(engine: SearchEngine) -> list[str]:
    return [
        n
        for n, (img, txt) in CONFIGS.items()
        if (img is None or engine.dense) and (txt is None or engine.text_dense)
    ]


@contextmanager
def config(engine: SearchEngine, name: str) -> Iterator[SearchEngine]:
    img, txt = CONFIGS[name]
    dense, text_dense, settings = engine.dense, engine.text_dense, engine.settings
    if (img is not None and dense is None) or (txt is not None and text_dense is None):
        raise RuntimeError(f"cấu hình {name} cần kênh mà engine không có")
    update = {}
    if img is None:
        engine.dense = None
    else:
        update["dense_weight"] = img
    if txt is None:
        engine.text_dense = None
    else:
        update["text_dense_weight"] = txt
    engine.settings = settings.model_copy(update=update)
    try:
        yield engine
    finally:
        engine.dense, engine.text_dense, engine.settings = dense, text_dense, settings


def ranked(engine: SearchEngine, query: str, k: int) -> list[str]:
    """sha256 đầy đủ, xen kẽ các nhóm như lưới hiển thị, cắt k."""
    resp, _ = engine.search(SearchRequest(query=query, top_k=k))
    order = flatten(resp, short=None)
    return order[:k]


def pool(
    engine: SearchEngine, query: str, depth: int = 20, configs: list[str] | None = None
) -> dict[str, dict[str, int]]:
    """image_id -> {cấu hình: hạng (1..depth)}."""
    out: dict[str, dict[str, int]] = {}
    for name in configs or available(engine):
        with config(engine, name):
            for rank, image_id in enumerate(ranked(engine, query, depth), start=1):
                out.setdefault(image_id, {})[name] = rank
    return out
