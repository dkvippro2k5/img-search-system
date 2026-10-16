"""Đọc meta.sqlite (chỉ đọc). Mỗi luồng một kết nối vì FastAPI chạy endpoint đồng bộ
trong threadpool."""

import json
import sqlite3
import threading
from collections.abc import Collection
from pathlib import Path

from imgsearch.retrieval.aliases import PersonEntry


class MetaStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._local = threading.local()

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def persons(self) -> list[PersonEntry]:
        rows = self.conn.execute("SELECT person_id, name, category, aliases FROM person")
        return [
            PersonEntry(
                r["person_id"], r["name"], r["category"], tuple(json.loads(r["aliases"] or "[]"))
            )
            for r in rows
        ]

    def count_images(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM image").fetchone()[0]

    def image_ids_by_row(self) -> list[str]:
        """image_id theo vec_row (0..N-1): dùng chung cho BM25 và FAISS."""
        return [r[0] for r in self.conn.execute("SELECT image_id FROM image ORDER BY vec_row")]

    def vec_rows_of(self, person_ids: Collection[str]) -> list[int]:
        if not person_ids:
            return []
        marks = ",".join("?" * len(person_ids))
        return [
            r[0]
            for r in self.conn.execute(
                f"SELECT DISTINCT i.vec_row FROM image_person ip JOIN image i USING (image_id) "
                f"WHERE ip.person_id IN ({marks}) ORDER BY i.vec_row",
                list(person_ids),
            )
        ]

    def images_of(self, person_ids: Collection[str], limit: int | None = None) -> list[str]:
        """Ảnh của các person, theo thứ tự vec_row (TODO: xếp theo điểm chất lượng)."""
        if not person_ids:
            return []
        marks = ",".join("?" * len(person_ids))
        sql = (
            f"SELECT DISTINCT i.image_id FROM image_person ip JOIN image i USING (image_id) "
            f"WHERE ip.person_id IN ({marks}) ORDER BY i.vec_row"
        )
        params: list = list(person_ids)
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [r[0] for r in self.conn.execute(sql, params)]

    def persons_of(self, image_ids: Collection[str]) -> dict[str, list[str]]:
        if not image_ids:
            return {}
        marks = ",".join("?" * len(image_ids))
        out: dict[str, list[str]] = {i: [] for i in image_ids}
        for r in self.conn.execute(
            f"SELECT image_id, person_id FROM image_person WHERE image_id IN ({marks})",
            list(image_ids),
        ):
            out[r[0]].append(r[1])
        return out

    def get_images(self, image_ids: Collection[str]) -> dict[str, dict]:
        if not image_ids:
            return {}
        marks = ",".join("?" * len(image_ids))
        rows = self.conn.execute(
            f"SELECT * FROM image WHERE image_id IN ({marks})", list(image_ids)
        )
        return {r["image_id"]: dict(r) for r in rows}
