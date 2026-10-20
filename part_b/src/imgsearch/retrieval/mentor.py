"""Client cho dịch vụ của mentor (API kiểu OpenAI): embedding văn bản qwen3 và reranker jina.

Dịch vụ chạy qua đường hầm tạm thời: mọi lời gọi trong đường /search có timeout ngắn, lỗi thì
trả None để engine quay về BM25 + dense ảnh (như quy tắc LLM fallback trong CLAUDE.md).
Chỉ gửi văn bản (truy vấn, tiêu đề, tên); không gửi ảnh. Key đọc từ .env, không ghi vào code.
"""

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

log = logging.getLogger(__name__)

# Qwen3-Embedding: truy vấn có câu chỉ dẫn, tài liệu thì không (model card). Chỉ dẫn tiếng Anh.
QUERY_INSTRUCT = (
    "Instruct: Given a Vietnamese image search query, retrieve news titles and descriptions "
    "of matching photos\nQuery: "
)


class MentorError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None):
        super().__init__(msg)
        self.status = status  # mã HTTP nếu server trả lỗi (429: vượt giới hạn request của key)


class MentorClient:
    def __init__(
        self,
        api_key: str,
        embed_url: str | None = None,
        embed_model: str = "qwen3-embedding-4b",
        rerank_url: str | None = None,
        rerank_model: str = "jina-reranker-v2",
        timeout: float = 1.0,
        dim: int | None = None,
        cache_size: int = 1024,
    ):
        self.api_key, self.timeout, self.dim = api_key, timeout, dim
        self.embed_url, self.embed_model = embed_url, embed_model
        self.rerank_url, self.rerank_model = rerank_url, rerank_model
        self.cache_size = cache_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        # Đếm lời gọi embedding truy vấn (không tính trúng cache) cho /metrics.
        self.stats = {"calls": 0, "errors": 0, "rate_limited": 0}
        self._stats_lock = threading.Lock()

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self.stats[key] += 1

    def _post(self, url: str, body: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise MentorError(f"HTTP {e.code}: {e.read()[:200]!r}", e.code) from e
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise MentorError(str(e)) from e

    # --- embedding ---

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        """Cắt Matryoshka (server không nhận `dimensions`) rồi chuẩn hóa L2."""
        if self.dim:
            vecs = vecs[:, : self.dim]
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)

    def embed(self, texts: list[str], timeout: float | None = None) -> np.ndarray:
        body = {"model": self.embed_model, "input": texts}
        r = self._post(self.embed_url, body, timeout or self.timeout)
        data = sorted(r["data"], key=lambda d: d["index"])
        return self._normalize(np.asarray([d["embedding"] for d in data], dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray | None:
        """Vector truy vấn (có chỉ dẫn), cache LRU. Lỗi/timeout -> None."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        self._count("calls")
        try:
            vec = self.embed([QUERY_INSTRUCT + text])[0]
        except MentorError as e:
            self._count("rate_limited" if e.status == 429 else "errors")
            log.warning("Embedding mentor lỗi (%s): bỏ kênh văn bản", e)
            return None
        if self.cache_size:
            self._cache[text] = vec
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return vec

    def embed_documents(
        self, texts: list[str], batch: int = 64, workers: int = 4, timeout: float = 120
    ) -> np.ndarray:
        """Mã hóa offline nhiều tài liệu (song song theo lô, thử lại khi lỗi tạm)."""

        def one(i: int) -> np.ndarray:
            for attempt in range(5):
                try:
                    return self.embed(texts[i : i + batch], timeout=timeout)
                except MentorError as e:
                    if attempt == 4:
                        raise
                    log.warning("Lô %d lỗi (%s), thử lại", i, e)
                    time.sleep(2**attempt)
            raise AssertionError

        with ThreadPoolExecutor(workers) as pool:
            parts = list(pool.map(one, range(0, len(texts), batch)))
        return np.concatenate(parts) if parts else np.zeros((0, self.dim or 0), np.float32)

    # --- rerank ---

    def rerank(
        self, query: str, documents: list[str], timeout: float | None = None
    ) -> list[float] | None:
        """Điểm liên quan theo đúng thứ tự `documents`. Lỗi/timeout -> None."""
        if not documents:
            return []
        body = {"model": self.rerank_model, "query": query, "documents": documents}
        try:
            r = self._post(self.rerank_url, body, timeout or self.timeout)
        except MentorError as e:
            log.warning("Rerank mentor lỗi (%s): giữ thứ tự cũ", e)
            return None
        scores = [0.0] * len(documents)
        for x in r["results"]:
            scores[x["index"]] = float(x["relevance_score"])
        return scores


class MentorTextEncoder:
    """Bộ mã hóa truy vấn cho DenseIndex của kênh văn bản. encode_query trả None khi mentor lỗi:
    DenseIndex khi đó trả rỗng và engine chạy tiếp với BM25 + dense ảnh."""

    def __init__(self, client: MentorClient, dim: int, name: str):
        self.client, self.dim, self.name = client, dim, name
        self._inflight: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mentor-embed")

    def prefetch(self, texts: list[str]) -> None:
        """Gửi trước embedding của mọi vế truy vấn cùng lúc (multi-intent không cộng dồn
        latency mạng); encode_query sau đó chờ lời gọi đang bay thay vì gọi lại."""
        with self._lock:
            if len(self._inflight) > 64:  # vế không bao giờ được dùng (truy vấn lỗi giữa chừng)
                self._inflight.clear()
            for t in dict.fromkeys(texts):
                if t not in self._inflight and t not in self.client._cache:
                    self._inflight[t] = self._pool.submit(self.client.embed_query, t)

    def encode_query(self, text: str) -> np.ndarray | None:
        with self._lock:
            fut = self._inflight.pop(text, None)
        vec = fut.result() if fut is not None else self.client.embed_query(text)
        if vec is None:
            return None
        vec = vec[: self.dim]
        return vec / max(float(np.linalg.norm(vec)), 1e-12)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.client.embed([QUERY_INSTRUCT + t for t in texts])[:, : self.dim]
