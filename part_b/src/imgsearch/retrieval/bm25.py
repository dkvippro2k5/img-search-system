"""BM25 trong bộ nhớ bằng numpy.

Thay cho SQLite FTS5: FTS5 phải tính bm25 cho mọi tài liệu khớp trước khi lấy top-k, với
80k ảnh và từ phổ biến mất 100-150 ms/truy vấn [Đã đo, 17/10]. Ở đây điểm của từng cặp
(term, tài liệu) được tính sẵn lúc build; truy vấn chỉ cộng các posting rồi lấy top-k.

Mỗi từ được lưu hai term: "a:<có dấu>" và "f:<không dấu>". Token truy vấn có dấu tìm theo
"a:", không dấu tìm theo "f:" (giống quy tắc so dấu của bộ phân tích truy vấn).
Trọng số trường (tên người > tiêu đề, caption) làm bằng cách lặp term (xấp xỉ BM25F).
"""

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from imgsearch.retrieval.lexicon import STOPWORDS
from imgsearch.retrieval.query_parser import tokenize

K1 = 1.2
B = 0.75
FIELD_REPEAT = {"names": 3, "description": 1, "caption": 1}


def doc_terms(names: str, description: str, caption: str) -> list[str]:
    terms: list[str] = []
    for text, repeat in (
        (names, FIELD_REPEAT["names"]),
        (description, FIELD_REPEAT["description"]),
        (caption, FIELD_REPEAT["caption"]),
    ):
        for t in tokenize(text or ""):
            if t.is_word and t.fold not in STOPWORDS:
                terms += [f"a:{t.norm}", f"f:{t.fold}"] * repeat
    return terms


def query_terms(text: str) -> list[str]:
    terms = [
        f"a:{t.norm}" if t.has_marks else f"f:{t.fold}"
        for t in tokenize(text)
        if t.is_word and t.fold not in STOPWORDS
    ]
    return list(dict.fromkeys(terms))


class BM25Index:
    def __init__(
        self,
        terms: Sequence[str],
        offsets: np.ndarray,
        docs: np.ndarray,
        scores: np.ndarray,
        n_docs: int,
    ):
        self.vocab = {t: i for i, t in enumerate(terms)}
        self.offsets = offsets  # int64, len = V + 1
        self.docs = docs  # int32, posting: chỉ số tài liệu
        self.scores = scores  # float32, posting: điểm BM25 tính sẵn
        self.n_docs = n_docs

    @classmethod
    def build(cls, corpus: Iterable[list[str]]) -> "BM25Index":
        """corpus[i] là danh sách term của tài liệu i (i = vec_row)."""
        tfs = [Counter(doc) for doc in corpus]
        n_docs = len(tfs)
        lengths = np.array([sum(tf.values()) for tf in tfs], dtype=np.float64)
        avgdl = float(lengths.mean()) if n_docs and lengths.mean() > 0 else 1.0

        df: Counter = Counter()
        for tf in tfs:
            df.update(tf.keys())
        idf = {t: math.log(1 + (n_docs - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for doc_id, tf in enumerate(tfs):
            norm = K1 * (1 - B + B * lengths[doc_id] / avgdl)
            for t, f in tf.items():
                postings[t].append((doc_id, idf[t] * f * (K1 + 1) / (f + norm)))

        terms = sorted(postings)
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(postings[t]) for t in terms])
        docs = np.empty(offsets[-1], dtype=np.int32)
        scores = np.empty(offsets[-1], dtype=np.float32)
        for i, t in enumerate(terms):
            pairs = postings[t]
            docs[offsets[i] : offsets[i + 1]] = [d for d, _ in pairs]
            scores[offsets[i] : offsets[i + 1]] = [s for _, s in pairs]
        return cls(terms, offsets, docs, scores, n_docs)

    def save(self, path: str | Path) -> None:
        terms = sorted(self.vocab, key=self.vocab.get)
        with open(path, "wb") as fh:
            np.savez(
                fh,
                terms=np.array(terms, dtype=str),
                offsets=self.offsets,
                docs=self.docs,
                scores=self.scores,
                n_docs=np.array(self.n_docs),
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with np.load(path, allow_pickle=False) as z:
            return cls(z["terms"].tolist(), z["offsets"], z["docs"], z["scores"], int(z["n_docs"]))

    def search(
        self, terms: Sequence[str], limit: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Top-k [(doc_id, điểm)] giảm dần. `allowed`: chỉ xét các doc_id này."""
        acc = np.zeros(self.n_docs, dtype=np.float32)
        for t in terms:
            i = self.vocab.get(t)
            if i is None:
                continue
            s, e = self.offsets[i], self.offsets[i + 1]
            acc[self.docs[s:e]] += self.scores[s:e]  # doc_id không lặp trong một posting

        cand = np.flatnonzero(acc) if allowed is None else allowed[acc[allowed] > 0]
        if len(cand) > limit:
            cand = cand[np.argpartition(-acc[cand], limit)[:limit]]
        cand = cand[np.argsort(-acc[cand], kind="stable")]
        return [(int(d), float(acc[d])) for d in cand]
