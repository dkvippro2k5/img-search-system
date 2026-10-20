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
        self.category_rows = {
            c: np.array(sorted(rows), dtype=np.int64)
            for c, rows in store.vec_rows_by_category().items()
        }

    @classmethod
    def load(cls, path: str | Path, store: MetaStore) -> "LexicalIndex":
        return cls(BM25Index.load(path), store)

    def search(
        self,
        text: str,
        limit: int = 200,
        person_ids: Collection[str] | None = None,
        categories: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Trả [(image_id, điểm)] giảm dần.

        `person_ids`: chỉ lấy ảnh có ít nhất một person này. Nếu không có `person_ids`,
        `categories` giới hạn trong ảnh của person thuộc các nghề đó (S3).
        """
        terms = query_terms(text)
        if not terms:
            return []
        allowed = self.allowed_rows(person_ids, categories)
        if allowed is not None and len(allowed) == 0:
            return []
        return [(self.image_ids[d], s) for d, s in self.bm25.search(terms, limit, allowed)]

    def allowed_rows(
        self,
        person_ids: Collection[str] | None = None,
        categories: Collection[str] | None = None,
    ) -> np.ndarray | None:
        """vec_row được phép: ảnh của `person_ids`, nếu không thì ảnh của người thuộc
        `categories`; None = mọi ảnh. Dùng chung cho BM25 và dense."""
        if person_ids:
            return np.array(self.store.vec_rows_of(person_ids), dtype=np.int64)
        if categories:
            parts = [self.category_rows[c] for c in categories if c in self.category_rows]
            return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)
        return None
