"""Truy xuất dense: mã hóa truy vấn bằng ONNX Runtime (không torch), so cosine với vector ảnh.

Kênh hiện có: ảnh SigLIP2 (bench/dense_bakeoff.py: trên golden, khi không có caption, BM25 +
ảnh nâng nDCG@10 từ 0,41 lên 0,64–0,69). Vector ảnh tính offline (scripts/encode_images.py),
căn theo vec_row lúc dựng index (`dense.npz`: `vecs` float16 N×d, `has_vec` bool N; ảnh chưa có
vector là hàng 0 và không bao giờ được trả). Tìm chính xác bằng numpy trên tập hàng được phép
(lọc person / nghề), không cần FAISS ở 90k ảnh (nhân ma trận ~ vài ms).

Model ở {MODELS_DIR}/{tên}/: tokenizer.json, text.onnx (fp32), encoder.json
(scripts/export_text_encoder.py). Tiền xử lý văn bản phải giống hệt lúc xuất model
(open_clip "canonicalize": bỏ dấu câu, chữ thường); export kiểm tra id token khớp open_clip.
"""

import html
import json
import math
import string
from collections import OrderedDict
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Protocol

import numpy as np

# Câu kiểm tra khi xuất model: có dấu, không dấu, dấu câu, chữ hoa.
CHECK_TEXTS = [
    "ca sĩ biểu diễn trên sân khấu",
    "Hari Won mặc váy cưới!",
    "van dong vien cau long",
    "ảnh có chữ VINGROUP",
]
_PUNCT = str.maketrans("", "", string.punctuation)
QUERY_CACHE_SIZE = 1024


def canonicalize(text: str) -> str:
    """Như open_clip clean="canonicalize" (bỏ ftfy: đầu vào đã là Unicode NFC sạch)."""
    text = html.unescape(html.unescape(text)).strip().replace("_", " ")
    return " ".join(text.translate(_PUNCT).lower().split())


class TextEncoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class OnnxTextEncoder:
    """Text encoder ONNX. encoder.json: context_length, pad_id, eos_id, dim."""

    def __init__(self, model_dir: str | Path, onnx_file: str = "text.onnx", threads: int = 0):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.meta = json.loads((model_dir / "encoder.json").read_text(encoding="utf-8"))
        self.name = self.meta["name"]
        self.dim = int(self.meta["dim"])
        self.ctx = int(self.meta["context_length"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / onnx_file), opts, providers=["CPUExecutionProvider"]
        )
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def token_ids(self, texts: list[str]) -> np.ndarray:
        eos, pad = self.meta.get("eos_id"), self.meta["pad_id"]
        out = np.full((len(texts), self.ctx), pad, dtype=np.int64)
        for i, enc in enumerate(self.tokenizer.encode_batch([canonicalize(t) for t in texts])):
            ids = list(enc.ids)
            if eos is not None and (not ids or ids[-1] != eos):
                ids.append(eos)
            if len(ids) > self.ctx:  # cắt nhưng giữ EOS ở cuối, như HF truncation
                ids = ids[: self.ctx - 1] + ([eos] if eos is not None else [ids[self.ctx - 1]])
            out[i, : len(ids)] = ids
        return out

    def encode(self, texts: list[str]) -> np.ndarray:
        (vecs,) = self.session.run(None, {"input_ids": self.token_ids(texts)})
        return np.asarray(vecs, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Có cache LRU: cùng truy vấn (nhiều vế, nhiều người trùng tên) chỉ mã hóa một lần."""
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        vec = self.encode([text])[0]
        self._cache[text] = vec
        if len(self._cache) > QUERY_CACHE_SIZE:
            self._cache.popitem(last=False)
        return vec


RowFilter = Callable[[Collection[str] | None, Collection[str] | None], np.ndarray | None]


class DenseIndex:
    """Vector ảnh theo vec_row + encoder truy vấn. `allowed_rows(person_ids, categories)` dùng
    chung với BM25 (LexicalIndex.allowed_rows) để hai kênh lọc giống hệt nhau."""

    def __init__(
        self,
        vecs: np.ndarray,
        has_vec: np.ndarray,
        image_ids: list[str],
        encoder: TextEncoder,
        allowed_rows: RowFilter,
        min_score: float | None = None,
    ):
        """`min_score`: cosine tối thiểu (xem min_cosine); None = không lọc."""
        if len(vecs) != len(image_ids) or len(has_vec) != len(image_ids):
            raise ValueError(f"dense.npz có {len(vecs)} hàng, index có {len(image_ids)} ảnh")
        if vecs.shape[1] != encoder.dim:
            raise ValueError(f"vector ảnh {vecs.shape[1]} chiều, encoder {encoder.dim} chiều")
        # float32 một lần lúc nạp (90k × 768 ≈ 276 MB): nhân cả ma trận mỗi truy vấn rồi mới
        # lấy hàng được phép, tránh chép ma trận con.
        self.vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        self.has_vec = has_vec.astype(bool)
        self.rows_with_vec = np.flatnonzero(self.has_vec)
        self.image_ids = image_ids
        self.encoder = encoder
        self.allowed_rows = allowed_rows
        self.min_score = min_score

    @classmethod
    def load(cls, path, image_ids, encoder, allowed_rows, min_score=None) -> "DenseIndex":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["vecs"], z["has_vec"], image_ids, encoder, allowed_rows, min_score)

    @property
    def coverage(self) -> float:
        return len(self.rows_with_vec) / max(len(self.image_ids), 1)

    def search(
        self,
        text: str,
        limit: int = 200,
        person_ids: Collection[str] | None = None,
        categories: Collection[str] | None = None,
    ) -> list[tuple[str, float]]:
        """[(image_id, cosine)] giảm dần, chỉ trong ảnh có vector và được phép."""
        if not text.strip():
            return []
        allowed = self.allowed_rows(person_ids, categories)
        rows = self.rows_with_vec if allowed is None else allowed[self.has_vec[allowed]]
        if len(rows) == 0:
            return []
        encode = getattr(self.encoder, "encode_query", None)
        q = encode(text) if encode else self.encoder.encode([text])[0]
        sims = (self.vecs @ q.astype(np.float32))[rows]
        # Ngưỡng là CỔNG, không cắt từng ảnh: ảnh khớp nhất vẫn dưới ngưỡng thì truy vấn không có
        # nghĩa với model ("xyz abc") -> bỏ cả kênh. Cắt từng ảnh làm danh sách ngắn, RRF chỉ còn
        # BM25 (golden: q12 0,72 -> 0,39). Trong ảnh của một người không xét (chỉ xếp lại, S1).
        if self.min_score is not None and not person_ids and sims.max() < self.min_score:
            return []
        if len(rows) > limit:
            top = np.argpartition(-sims, limit)[:limit]
            top = top[np.argsort(-sims[top], kind="stable")]
        else:
            top = np.argsort(-sims, kind="stable")
        return [(self.image_ids[int(rows[i])], float(sims[i])) for i in top]


def min_cosine(meta: dict, min_prob: float) -> float | None:
    """Cosine ứng với xác suất khớp `min_prob` của SigLIP: sigmoid(scale·cos + bias) = min_prob.
    Model không có logit_bias (CLIP) thì None."""
    scale, bias = meta.get("logit_scale"), meta.get("logit_bias")
    if scale is None or bias is None or not 0 < min_prob < 1:
        return None
    return (math.log(min_prob / (1 - min_prob)) - bias) / scale


def align_vectors(
    image_ids_by_row: list[str], emb: np.ndarray, emb_ids: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Căn embedding (thứ tự tùy ý, khóa image_id) theo vec_row của index -> (vecs f16, has_vec)."""
    pos = {i: k for k, i in enumerate(emb_ids)}
    vecs = np.zeros((len(image_ids_by_row), emb.shape[1]), dtype=np.float16)
    has = np.zeros(len(image_ids_by_row), dtype=bool)
    for row, image_id in enumerate(image_ids_by_row):
        k = pos.get(image_id)
        if k is not None:
            vecs[row] = emb[k]
            has[row] = True
    return vecs, has
