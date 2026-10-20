"""Mã hóa văn bản của mọi ảnh trong index bằng qwen3-embedding của mentor (P3, kênh văn bản).

python -X utf8 scripts/encode_texts.py --index-version people
python scripts/build_index.py ... (tự đọc ../data/features/text_embeddings.*), hoặc
python -X utf8 scripts/encode_texts.py --index-version people --write-index   # gắn vào index có sẵn

Văn bản = đúng văn bản BM25 của ảnh (build_meta.lexical_texts: tên + biệt danh person, tiêu đề,
tên entity, từ khóa, caption). Chỉ gửi văn bản, không gửi ảnh. Chia shard
../data/features/text_shards/ (chạy lại thì bỏ qua shard đã có), rồi gộp thành
../data/features/text_embeddings.{npy,ids.txt,json} (float16, đủ chiều; cắt chiều lúc nạp).
"""

import argparse
import json
import logging
import sqlite3
import time
from pathlib import Path

import numpy as np

from imgsearch.config import Settings
from imgsearch.data.build_meta import add_text_dense, lexical_texts
from imgsearch.retrieval.mentor import MentorClient

SHARD = 2048


def doc_text(parts: tuple[str, str, str]) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())[:1000]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--features", type=Path, default=Path("../data/features"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--write-index", action="store_true", help="dựng text_dense.npz vào index")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    s = Settings()
    if not (s.mentor_api_key and s.text_embed_url):
        raise SystemExit("Thiếu MENTOR_API_KEY / TEXT_EMBED_URL trong .env")
    client = MentorClient(s.mentor_api_key, s.text_embed_url, s.text_embed_model)

    meta = args.index_dir / f"v{args.index_version}" / "meta.sqlite"
    conn = sqlite3.connect(f"file:{meta}?mode=ro", uri=True)
    ids = [r[0] for r in conn.execute("SELECT image_id FROM image ORDER BY vec_row")]
    conn.close()
    texts = [doc_text(t) for t in lexical_texts(meta)]
    shards = args.features / "text_shards"
    shards.mkdir(parents=True, exist_ok=True)

    t0, done = time.perf_counter(), 0
    for start in range(0, len(texts), SHARD):
        path = shards / f"{start:07d}.npy"
        if path.exists():
            continue
        vecs = client.embed_documents(
            texts[start : start + SHARD], batch=args.batch, workers=args.workers
        )
        np.save(path, vecs.astype(np.float16))
        done += len(vecs)
        rate = done / (time.perf_counter() - t0)
        left = (len(texts) - start - SHARD) / rate if rate else 0
        n = min(start + SHARD, len(texts))
        logging.info("%d/%d (%.0f văn bản/s, còn ~%.0f phút)", n, len(texts), rate, left / 60)

    emb = np.concatenate([np.load(shards / f"{i:07d}.npy") for i in range(0, len(texts), SHARD)])
    base = args.features / "text_embeddings"
    np.save(base.with_suffix(".npy"), emb)
    base.with_suffix(".ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    info = {
        "model": s.text_embed_model,
        "dim": int(emb.shape[1]),
        "n": len(ids),
        "text": "build_meta.lexical_texts nối bằng dấu cách, cắt 1000 ký tự",
        "index_version": args.index_version,
        "created_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    base.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=2), "utf-8")
    logging.info("Đã ghi %s.npy: %s", base, emb.shape)
    if args.write_index:
        logging.info("text_dense: %s", add_text_dense(meta.parent, args.features))


if __name__ == "__main__":
    main()
