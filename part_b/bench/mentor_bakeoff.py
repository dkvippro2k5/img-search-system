"""Bake-off kênh văn bản của mentor trên bộ golden (P3): qwen3 dense văn bản, jina rerank.

python -X utf8 bench/mentor_bakeoff.py      # cần .env có MENTOR_API_KEY + URL; extra "embed"

So với hệ thống hiện tại (bm25 + 2·ảnh SigLIP2) trên 11 truy vấn bối cảnh của golden, hai chế độ
caption / no_caption như bench/dense_bakeoff.py. Chỉ gửi văn bản golden (tên entity + tiêu đề
+ caption) và truy vấn; không gửi ảnh. Golden là tập phát triển, nhỏ: chỉ để chọn hướng; số
quyết định đo trên dev của test set thật (eval/run_testset.py). Ghi ../reports/mentor_bakeoff.md.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense_bakeoff import (  # noqa: E402
    SHORT,
    TOP,
    caption_of,
    e5_encoder,
    encode_images,
    load_rows,
    rank_dense,
    score,
    text_encoder,
)

from imgsearch.config import Settings  # noqa: E402
from imgsearch.eval.run_eval import load_gold_queries  # noqa: E402
from imgsearch.retrieval.aliases import AliasIndex  # noqa: E402
from imgsearch.retrieval.bm25 import BM25Index, doc_terms, query_terms  # noqa: E402
from imgsearch.retrieval.fusion import rrf  # noqa: E402
from imgsearch.retrieval.mentor import QUERY_INSTRUCT, MentorClient  # noqa: E402
from imgsearch.retrieval.query_parser import QueryParser  # noqa: E402
from imgsearch.retrieval.store import MetaStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
IMG = "siglip2-b32"
DIMS = (2560, 1024, 512)  # cắt Matryoshka phía client


def fuse(lists: list[list[int]], weights: list[float]) -> list[int]:
    return [d for d, _ in rrf(lists, weights=weights)]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=ROOT.parent / "data" / "golden")
    ap.add_argument("--index", type=Path, default=ROOT.parent / "data/index/vgolden-auto")
    ap.add_argument("--cache", type=Path, default=ROOT.parent / "data/features/bakeoff")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "reports" / "mentor_bakeoff.md")
    args = ap.parse_args(argv)

    s = Settings()
    if not (s.mentor_api_key and s.text_embed_url and s.rerank_url):
        raise SystemExit("Thiếu MENTOR_API_KEY / TEXT_EMBED_URL / RERANK_URL trong .env")
    mentor = MentorClient(
        s.mentor_api_key, s.text_embed_url, s.text_embed_model, s.rerank_url, s.rerank_model
    )

    rows = load_rows(args.golden)
    ids = [r["sha256"][:SHORT] for r in rows]
    store = MetaStore(args.index / "meta.sqlite")
    parser = QueryParser(AliasIndex(store.persons()))
    queries = [
        q
        for q in load_gold_queries(args.golden / "gold" / "queries.json")
        if q.type == "single" and q.relevant and not parser.parse(q.query).mentions
    ]
    print(f"{len(rows)} ảnh, {len(queries)} truy vấn bối cảnh")

    def allowed_for(q) -> np.ndarray:
        cats = {c for it in parser.parse(q.query).intents for c in it.categories}
        if not cats:
            return np.arange(len(rows))
        return np.array([i for i, r in enumerate(rows) if r.get("category") in cats])

    img_vecs, _ = encode_images(IMG, rows, args.golden, args.cache)
    img_q = text_encoder(IMG)([q.query for q in queries])
    e5 = e5_encoder()
    e5_q = e5([f"query: {q.query}" for q in queries])

    t = time.perf_counter()
    qwen_q_full = mentor.embed([QUERY_INSTRUCT + q.query for q in queries], timeout=60)
    q_ms = (time.perf_counter() - t) * 1000 / len(queries)

    results: dict[str, dict[str, list[dict]]] = {}
    per_query: dict[str, dict[str, float]] = {}
    rerank_ms = []
    for mode in ("caption", "no_caption"):
        texts = [
            " ".join(
                p
                for p in (
                    r["entity_name"],
                    r.get("source_title") or "",
                    caption_of(r) if mode == "caption" else "",
                )
                if p
            )
            for r in rows
        ]
        cache = args.cache / f"qwen3_{mode}.npy"
        if cache.exists():
            qwen_docs_full = np.load(cache)
        else:
            qwen_docs_full = mentor.embed_documents(texts, batch=64, workers=2)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, qwen_docs_full)
        bm25 = BM25Index.build(
            doc_terms(
                r["entity_name"],
                r.get("source_title") or "",
                caption_of(r) if mode == "caption" else "",
            )
            for r in rows
        )
        e5_docs = e5([f"passage: {t}" for t in texts])

        def trunc(v: np.ndarray, d: int) -> np.ndarray:
            v = v[..., :d]
            return v / np.linalg.norm(v, axis=-1, keepdims=True)

        for qi, q in enumerate(queries):
            allowed = allowed_for(q)
            base = {
                "bm25": [d for d, _ in bm25.search(query_terms(q.query), TOP, allowed)],
                "img": rank_dense(img_vecs, img_q[qi], allowed),
                "e5": rank_dense(e5_docs, e5_q[qi], allowed),
            }
            for d in DIMS:
                base[f"qwen{d}"] = rank_dense(
                    trunc(qwen_docs_full, d), trunc(qwen_q_full[qi], d), allowed
                )
            combos = {
                "bm25": base["bm25"],
                "img": base["img"],
                "qwen2560": base["qwen2560"],
                "e5": base["e5"],
                "bm25+2img (hiện tại)": fuse([base["bm25"], base["img"]], [1, 2]),
                "bm25+qwen2560": fuse([base["bm25"], base["qwen2560"]], [1, 1]),
                "bm25+e5+2img": fuse([base["bm25"], base["e5"], base["img"]], [1, 1, 2]),
            }
            for d in DIMS:
                combos[f"bm25+qwen{d}+2img"] = fuse(
                    [base["bm25"], base[f"qwen{d}"], base["img"]], [1, 1, 2]
                )
            combos["bm25+2qwen2560+2img"] = fuse(
                [base["bm25"], base["qwen2560"], base["img"]], [1, 2, 2]
            )
            # Rerank top 50 của hai hệ thống; "+rrf" = gộp thứ hạng rerank với thứ hạng gốc
            # (reranker chỉ đọc chữ, không để nó một mình quyết định ảnh không có tiêu đề khớp).
            for name in ("bm25+2img (hiện tại)", "bm25+qwen2560+2img"):
                cands = combos[name][:TOP]
                t = time.perf_counter()
                sc = mentor.rerank(q.query, [texts[d] for d in cands], timeout=60)
                rerank_ms.append((time.perf_counter() - t) * 1000)
                if sc is None:
                    continue
                order = [cands[i] for i in np.argsort(-np.asarray(sc), kind="stable")]
                combos[f"rerank({name})"] = order
                combos[f"rerank+rrf({name})"] = fuse([cands, order], [1, 1])
            for method, docs in combos.items():
                sc = score([ids[d] for d in docs[:TOP]], q)
                results.setdefault(mode, {}).setdefault(method, []).append(sc)
                per_query.setdefault(f"{mode}|{method}", {})[q.id] = sc["ndcg10"]

    lines = [
        "# Kênh văn bản của mentor trên bộ golden (P3)",
        "",
        f"{len(rows)} ảnh, {len(queries)} truy vấn bối cảnh ({', '.join(q.id for q in queries)}). "
        "Golden nhỏ và là tập phát triển: chỉ để chọn hướng, chênh lệch nhỏ là nhiễu. "
        "Văn bản ảnh = tên entity + tiêu đề (+ caption ở chế độ `caption`).",
        "",
        f"- qwen3-embedding-4b qua API mentor: {q_ms:.0f} ms/truy vấn khi gửi theo lô "
        f"{len(queries)} câu [Đã đo]. Server không nhận `dimensions`: 1024/512 là cắt phía client.",
        f"- jina-reranker-v2, 50 văn bản: p50 {np.median(rerank_ms):.0f} ms, max "
        f"{max(rerank_ms):.0f} ms [Đã đo].",
        "- RRF có trọng số: `bm25+qwen+2img` = BM25 (1) + qwen (1) + ảnh SigLIP2 (2).",
    ]
    for mode, by_method in results.items():
        lines += [
            "",
            f"## Chế độ `{mode}`",
            "",
            "| Phương pháp | P@R | R@10 | R@50 | nDCG@10 |",
            "|---|---|---|---|---|",
        ]
        ordered = sorted(by_method.items(), key=lambda kv: -np.mean([x["ndcg10"] for x in kv[1]]))
        for method, ss in ordered:
            m = {k: np.mean([x[k] for x in ss]) for k in ss[0]}
            lines.append(
                f"| {method} | {m['p_r']:.2f} | {m['r10']:.2f} | {m['r50']:.2f} "
                f"| {m['ndcg10']:.2f} |"
            )
    shown = [
        "bm25+2img (hiện tại)",
        "qwen2560",
        "bm25+qwen2560+2img",
        "rerank(bm25+2img (hiện tại))",
        "rerank+rrf(bm25+2img (hiện tại))",
    ]
    lines += ["", "## nDCG@10 từng truy vấn (no_caption)", ""]
    lines.append("| Truy vấn | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---|" * len(shown))
    for q in queries:
        vals = [per_query.get(f"no_caption|{m}", {}).get(q.id) for m in shown]
        cells = ["—" if v is None else f"{v:.2f}" for v in vals]
        lines.append(f"| {q.id} {q.query} | " + " | ".join(cells) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
