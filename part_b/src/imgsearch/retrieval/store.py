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

    def vec_rows_by_category(self) -> dict[str, list[int]]:
        """category -> vec_row của ảnh có person thuộc nghề đó. Gọi một lần lúc nạp:
        truy vấn mỗi lần mất ~100 ms trên 80k ảnh [Đã đo, 17/10]."""
        out: dict[str, list[int]] = {}
        for category, row in self.conn.execute(
            "SELECT DISTINCT p.category, i.vec_row FROM image_person ip "
            "JOIN image i USING (image_id) JOIN person p USING (person_id) "
            "WHERE p.category IS NOT NULL"
        ):
            out.setdefault(category, []).append(row)
        return out

    def unsafe_image_ids(self, nsfw_block_score: float, unsafe_flags: Collection[str]) -> set[str]:
        """Ảnh bị ẩn khi bật lọc an toàn (Q4): điểm NSFW cao hoặc mang cờ nhạy cảm."""
        marks = ",".join("?" * len(unsafe_flags)) or "NULL"
        return {
            r[0]
            for r in self.conn.execute(
                f"SELECT image_id FROM image WHERE nsfw_score >= ? OR sensitive_flag IN ({marks})",
                [nsfw_block_score, *unsafe_flags],
            )
        }

    def relations_touching(self, person_id: str) -> list[dict]:
        """Mọi cạnh có person ở một đầu (cả hai chiều); multi_hop.py chọn chiều cần dùng."""
        rows = self.conn.execute(
            "SELECT * FROM relation WHERE subj_person = ? OR obj_person = ?",
            [person_id, person_id],
        )
        return [dict(r) for r in rows]

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
