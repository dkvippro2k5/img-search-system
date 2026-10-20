"""Dựng testsets/queries.v1.jsonl = truy vấn sinh tự động (imgsearch/eval/testgen.py, đọc
meta.sqlite) + truy vấn viết tay (testsets/manual.v1.jsonl: bối cảnh, multi-hop, phủ định).

python scripts/make_testset.py --index-version people

Cùng seed + cùng index -> cùng test set. Không sửa tay queries.v1.jsonl: sửa manual hoặc
generator rồi chạy lại.
"""

import argparse
import collections
import json
import random
import sqlite3
from pathlib import Path

from imgsearch.eval.testgen import Generator
from imgsearch.eval.testset import load_testset, write_testset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--manual", type=Path, default=Path("testsets/manual.v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("testsets/queries.v1.jsonl"))
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--bench-out", type=Path, default=Path("testsets/bench_queries.json"))
    args = ap.parse_args(argv)

    db = args.index_dir / f"v{args.index_version}" / "meta.sqlite"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    auto = Generator(conn, seed=args.seed).run()
    manual = load_testset(args.manual)
    ids = collections.Counter(q.id for q in auto + manual)
    if dup := [i for i, n in ids.items() if n > 1]:
        raise SystemExit(f"id bị lặp giữa auto và manual: {dup}")

    queries = auto + manual
    write_testset(args.out, queries)
    count = collections.Counter((q.type, q.split) for q in queries)
    for t in ("single", "multi_intent", "multi_hop"):
        print(f"{t:13s} dev {count[(t, 'dev')]:3d}  test {count[(t, 'test')]:3d}")
    print(f"{len(queries)} truy vấn ({len(auto)} tự động, {len(manual)} viết tay) -> {args.out}")

    bench = bench_queries(queries, random.Random(args.seed))
    rows = ",\n".join(json.dumps(b, ensure_ascii=False) for b in bench)
    args.bench_out.write_text(f"[\n{rows}\n]\n", encoding="utf-8")
    print(f"{len(bench)} truy vấn benchmark -> {args.bench_out}")


def bench_queries(queries, rng: random.Random, n: int = 100) -> list[dict]:
    """Truy vấn cho k6 (PLAN 6): 50% single, 30% multi-intent, 20% multi-hop, lấy từ test set
    (cả hai split: đây là đo tốc độ, không phải chất lượng)."""
    out = []
    for qtype, share in (("single", 0.5), ("multi_intent", 0.3), ("multi_hop", 0.2)):
        pool = sorted({q.query for q in queries if q.type == qtype})
        picked = rng.sample(pool, min(len(pool), int(n * share)))
        out += [{"type": qtype, "query": t} for t in picked]
    return out


if __name__ == "__main__":
    main()
