"""Điều phối truy xuất lai (H2): phân tích -> liên kết person -> BM25 + dense -> RRF.

Luồng xử lý xem PLAN.md 4.4. Mỗi bước ghi thời gian vào `timings` (ms).
"""

import logging
import time

from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.config import Settings
from imgsearch.retrieval.aliases import AliasIndex
from imgsearch.retrieval.lexical import LexicalIndex
from imgsearch.retrieval.query_parser import QueryParser
from imgsearch.retrieval.store import MetaStore

log = logging.getLogger(__name__)


class SearchEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False
        self.n_images = 0
        self.store: MetaStore | None = None
        self.lexical: LexicalIndex | None = None
        self.parser = QueryParser(AliasIndex([]))

    def load(self) -> None:
        """Nạp index/v{INDEX_VERSION}/meta.sqlite nếu có và dựng từ điển person.

        TODO: tải từ S3 khi INDEX_FROM_S3; nạp text.faiss; bắt buộc có index khi chạy Docker.
        """
        meta = self.settings.index_path / "meta.sqlite"
        if meta.exists():
            self.store = MetaStore(meta)
            self.lexical = LexicalIndex.load(self.settings.index_path / "lexical.npz", self.store)
            self.parser = QueryParser(AliasIndex(self.store.persons()))
            self.n_images = self.store.count_images()
            log.info("Đã nạp %s: %d ảnh, %d person", meta, self.n_images, len(self.parser.aliases))
        else:
            log.warning("Không thấy %s, chạy với index rỗng", meta)
        self.ready = True

    def info(self) -> dict:
        return {
            "index_version": self.settings.index_version,
            "index_loaded": self.store is not None,
            "n_images": self.n_images,
            "n_persons": len(self.parser.aliases),
        }

    def search(self, req: SearchRequest) -> tuple[SearchResponse, dict[str, float]]:
        timings: dict[str, float] = {}

        t = time.perf_counter()
        parsed = self.parser.parse(req.query)
        timings["parse"] = (time.perf_counter() - t) * 1000

        # TODO (bước 2): lexical + dense, RRF, multi-intent / multi-hop, dedupe, lọc an toàn.
        return (
            SearchResponse(query=req.query, query_type=parsed.type, groups=[]),
            timings,
        )
