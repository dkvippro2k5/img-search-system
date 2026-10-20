"""So sánh truy xuất dense trên bộ golden (bước 3): BM25 / dense văn bản / dense ảnh / RRF.

Chạy offline, cần extra "embed" (torch, open_clip, transformers):
    python -X utf8 bench/dense_bakeoff.py                     # mọi model
    python -X utf8 bench/dense_bakeoff.py --models siglip2-b16
Ghi ../reports/dense_bakeoff.md. Embedding cache ở ../data/features/bakeoff/ (không commit).

Đánh giá trên các truy vấn single KHÔNG nêu tên người (bối cảnh, q12–q22): đây là chỗ dense
có tác dụng; truy vấn có tên đã lọc theo person. Hai chế độ:
- caption: văn bản ảnh gồm tên entity + tiêu đề + caption VLM (như golden; chỉ 100 ảnh golden
  có caption);
- no_caption: bỏ caption, giống dữ liệu thật hiện tại (CSV không có caption).
Chỉ đọc metadata.json + images/; gold/ chỉ dùng để chấm.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from imgsearch.eval.metrics import ndcg_at_k, recall_at_k
from imgsearch.eval.run_eval import SHORT, load_gold_queries
from imgsearch.retrieval.aliases import AliasIndex
from imgsearch.retrieval.bm25 import BM25Index, doc_terms, query_terms
from imgsearch.retrieval.fusion import rrf
from imgsearch.retrieval.query_parser import QueryParser
from imgsearch.retrieval.store import MetaStore

ROOT = Path(__file__).resolve().parents[1]
IMAGE_MODELS = {
    # tên ngắn: (open_clip model, pretrained)
    "xlmr-b32": ("xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k"),
    "siglip-i18n-b16": ("ViT-B-16-SigLIP-i18n-256", "webli"),
    "siglip2-b32": ("ViT-B-32-SigLIP2-256", "webli"),
    "siglip2-b16": ("ViT-B-16-SigLIP2-256", "webli"),
}
TEXT_MODEL = "intfloat/multilingual-e5-small"
TOP = 50


def load_rows(golden: Path) -> list[dict]:
    rows = json.loads((golden / "metadata.json").read_text(encoding="utf-8"))
    return [r for r in rows if r.get("review_status") == "approved"]


def caption_of(r: dict) -> str:
    return " ".join(p for p in [r.get("caption_vi") or "", " ".join(r.get("tags") or [])] if p)


# --- encoders ---


def encode_images(
    name: str, rows: list[dict], golden: Path, cache: Path
) -> tuple[np.ndarray, dict]:
    path = cache / f"img_{name}.npy"
    if path.exists():
        return np.load(path), {"cached": True}
    import open_clip
    import torch
    from PIL import Image

    arch, pretrained = IMAGE_MODELS[name]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval()
    vecs, t0 = [], time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = [
                preprocess(Image.open(golden / r["file"]).convert("RGB")) for r in rows[i : i + 32]
            ]
            v = model.encode_image(torch.stack(batch))
            vecs.append(torch.nn.functional.normalize(v, dim=-1).numpy())
    secs = time.perf_counter() - t0
    out = np.concatenate(vecs).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return out, {"cached": False, "images_per_s": round(len(rows) / secs, 1)}


def text_encoder(name: str):
    """Hàm (list[str]) -> vector chuẩn hóa, cùng không gian với ảnh của model `name`."""
    import open_clip
    import torch

    arch, pretrained = IMAGE_MODELS[name]
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval()
    tok = open_clip.get_tokenizer(arch)

    def enc(texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            v = model.encode_text(tok(texts))
            return torch.nn.functional.normalize(v, dim=-1).numpy()

    return enc


def e5_encoder():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).eval()

    def enc(texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                b = tok(
                    texts[i : i + 64],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                h = model(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1)
                out.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        return np.concatenate(out)

    return enc


# --- đánh giá ---


def rank_dense(doc_vecs: np.ndarray, q_vec: np.ndarray, allowed: np.ndarray) -> list[int]:
    sims = doc_vecs[allowed] @ q_vec
    return [int(allowed[i]) for i in np.argsort(-sims)[:TOP]]


def score(ranked_ids: list[str], q) -> dict:
    ranked = [i for i in ranked_ids if i not in q.unjudged]
    rel = {i for i, g in q.relevant.items() if g > 0}
    r = min(len(rel), 10)
    return {
        "p_r": sum(i in rel for i in ranked[:r]) / r,
        "r10": recall_at_k(ranked, rel, 10),
        "r50": recall_at_k(ranked, rel, 50),
        "ndcg10": ndcg_at_k(ranked, q.relevant, 10),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--golden", type=Path, default=ROOT.parent / "data" / "golden")
    ap.add_argument("--index", type=Path, default=ROOT.parent / "data/index/vgolden-auto")
    ap.add_argument("--cache", type=Path, default=ROOT.parent / "data/features/bakeoff")
    ap.add_argument("--models", nargs="*", default=list(IMAGE_MODELS))
    ap.add_argument("--out", type=Path, default=ROOT.parent / "reports" / "dense_bakeoff.md")
    args = ap.parse_args(argv)

    rows = load_rows(args.golden)
    ids = [r["sha256"][:SHORT] for r in rows]
    store = MetaStore(args.index / "meta.sqlite")  # chỉ để parser nhận tên người / nghề
    parser = QueryParser(AliasIndex(store.persons()))
    queries = [
        q
        for q in load_gold_queries(args.golden / "gold" / "queries.json")
        if q.type == "single" and q.relevant and not parser.parse(q.query).mentions
    ]
    print(f"{len(rows)} ảnh, {len(queries)} truy vấn bối cảnh: {[q.id for q in queries]}")

    def allowed_for(q) -> np.ndarray:
        cats = {c for it in parser.parse(q.query).intents for c in it.categories}
        if not cats:
            return np.arange(len(rows))
        return np.array([i for i, r in enumerate(rows) if r.get("category") in cats])

    img_vecs, img_info, img_q = {}, {}, {}
    for name in args.models:
        img_vecs[name], img_info[name] = encode_images(name, rows, args.golden, args.cache)
        enc = text_encoder(name)
        t0 = time.perf_counter()
        img_q[name] = enc([q.query for q in queries])
        img_info[name]["text_ms_per_query"] = round(
            (time.perf_counter() - t0) * 1000 / len(queries), 1
        )
        print(name, img_info[name])
    e5 = e5_encoder()
    e5_q = e5([f"query: {q.query}" for q in queries])

    results: dict[str, dict[str, list[dict]]] = {}
    per_query: dict[str, dict[str, float]] = {}
    for mode in ("caption", "no_caption"):
        texts = [
            (
                r["entity_name"],
                r.get("source_title") or "",
                caption_of(r) if mode == "caption" else "",
            )
            for r in rows
        ]
        bm25 = BM25Index.build(doc_terms(*t) for t in texts)
        e5_docs = e5([f"passage: {' '.join(p for p in t if p)}" for t in texts])
        for qi, q in enumerate(queries):
            allowed = allowed_for(q)
            lists = {
                "bm25": [d for d, _ in bm25.search(query_terms(q.query), TOP, allowed)],
                "e5": rank_dense(e5_docs, e5_q[qi], allowed),
            }
            for name in args.models:
                lists[f"img:{name}"] = rank_dense(img_vecs[name], img_q[name][qi], allowed)
            combos = {**lists, "bm25+e5": [d for d, _ in rrf([lists["bm25"], lists["e5"]])]}
            for name in args.models:
                img = lists[f"img:{name}"]
                combos[f"bm25+img:{name}"] = [d for d, _ in rrf([lists["bm25"], img])]
                for w in (2.0, 3.0):
                    combos[f"bm25+{w:g}img:{name}"] = [
                        d for d, _ in rrf([lists["bm25"], img], weights=[1, w])
                    ]
                combos[f"bm25+e5+img:{name}"] = [
                    d for d, _ in rrf([lists["bm25"], lists["e5"], img])
                ]
                for w in (2.0, 3.0):  # ảnh nặng hơn: hai danh sách văn bản sai không lấn át
                    combos[f"bm25+e5+{w:g}img:{name}"] = [
                        d for d, _ in rrf([lists["bm25"], lists["e5"], img], weights=[1, 1, w])
                    ]
            for method, docs in combos.items():
                s = score([ids[d] for d in docs[:TOP]], q)
                results.setdefault(mode, {}).setdefault(method, []).append(s)
                per_query.setdefault(f"{mode}|{method}", {})[q.id] = s["ndcg10"]

    lines = [
        "# So sánh dense trên bộ golden (bước 3)",
        "",
        f"{len(rows)} ảnh approved, {len(queries)} truy vấn bối cảnh "
        f"({', '.join(q.id for q in queries)}). "
        "Bộ nhỏ, golden là tập phát triển: chỉ dùng để chọn hướng, chênh lệch nhỏ là nhiễu.",
        "",
        "## Model ảnh",
        "",
        "| Model | Ảnh/s (CPU, batch 32) | Mã hóa truy vấn (torch, ms) |",
        "|---|---|---|",
    ]
    for name in args.models:
        info = img_info[name]
        lines.append(
            f"| {name} ({IMAGE_MODELS[name][0]}) | {info.get('images_per_s', 'cache')} | "
            f"{info['text_ms_per_query']} |"
        )
    for mode, by_method in results.items():
        lines += [
            "",
            f"## Chế độ `{mode}`",
            "",
            "| Phương pháp | P@R | R@10 | R@50 | nDCG@10 |",
            "|---|---|---|---|---|",
        ]
        ordered = sorted(by_method.items(), key=lambda kv: -np.mean([s["ndcg10"] for s in kv[1]]))
        for method, ss in ordered:
            m = {k: np.mean([s[k] for s in ss]) for k in ss[0]}
            lines.append(
                f"| {method} | {m['p_r']:.2f} | {m['r10']:.2f} | {m['r50']:.2f} "
                f"| {m['ndcg10']:.2f} |"
            )
    lines += ["", "## nDCG@10 từng truy vấn (no_caption)", ""]
    shown = [
        "bm25",
        "e5",
        *[f"img:{n}" for n in args.models],
        "bm25+e5",
        *[f"bm25+e5+img:{n}" for n in args.models],
    ]
    lines.append("| Truy vấn | " + " | ".join(shown) + " |")
    lines.append("|---|" + "---|" * len(shown))
    for q in queries:
        vals = [per_query[f"no_caption|{m}"][q.id] for m in shown]
        lines.append(f"| {q.id} {q.query} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
