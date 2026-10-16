"""Truy xuất từ khóa: BM25 (retrieval/bm25.py) trên tên person + tiêu đề bài báo + caption.

So khớp dấu theo từng token, giống bộ phân tích truy vấn: token có dấu chỉ khớp đúng dấu
("chùa" không khớp "chưa"), token không dấu khớp dạng bỏ dấu ("chua" khớp cả hai).
Ảnh khớp nhiều token xếp cao hơn.
"""

from collections.abc import Collection
from pathlib import Path

import numpy as np

from imgsearch.retrieval.bm25 import BM25Index, query_terms
from imgsearch.retrieval.store import MetaStore


class LexicalIndex:
    def __init__(self, bm25: BM25Index, store: MetaStore):
        self.bm25 = bm25
        self.store = store
        self.image_ids = store.image_ids_by_row()  # vec_row -> image_id
        if len(self.image_ids) != bm25.n_docs:
            raise ValueError(
                f"lexical.npz có {bm25.n_docs} tài liệu, meta.sqlite có {len(self.image_ids)} ảnh"
            )

    @classmethod
    def load(cls, path: str | Path, store: MetaStore) -> "LexicalIndex":
        return cls(BM25Index.load(path), store)

    def search(
        self, text: str, limit: int = 200, person_ids: Collection[str] | None = None
    ) -> list[tuple[str, float]]:
        """Trả [(image_id, điểm)] giảm dần. `person_ids`: chỉ lấy ảnh có ít nhất một person này."""
        terms = query_terms(text)
        if not terms:
            return []
        allowed = None
        if person_ids:
            allowed = np.array(self.store.vec_rows_of(person_ids), dtype=np.int64)
            if len(allowed) == 0:
                return []
        return [(self.image_ids[d], s) for d, s in self.bm25.search(terms, limit, allowed)]
