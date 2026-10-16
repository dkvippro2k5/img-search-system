"""Điều phối truy xuất lai (H2): phân tích -> liên kết person -> BM25 + dense -> RRF.

Luồng xử lý xem PLAN.md 4.4. Mỗi bước ghi thời gian vào `timings` (ms).
"""

import time

from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.config import Settings
from imgsearch.retrieval.query_parser import parse


class SearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.n_images = 0

    def load(self) -> None:
        """TODO: tải index/v{INDEX_VERSION}/ (meta.sqlite, text.faiss, manifest.json)."""
        self.ready = True

    def info(self) -> dict:
        return {"index_version": self.settings.index_version, "n_images": self.n_images}

    def search(self, req: SearchRequest) -> tuple[SearchResponse, dict[str, float]]:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        parsed = parse(req.query)
        timings["parse"] = (time.perf_counter() - t) * 1000

        # TODO: retrieve (lexical + dense), fusion, multi-intent / multi-hop, dedupe, lọc an toàn.
        return (
            SearchResponse(query=req.query, query_type=parsed.type, groups=[]),
            timings,
        )
