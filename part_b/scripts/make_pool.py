"""Gộp top-20 của các cấu hình (imgsearch/eval/pooling.py) cho truy vấn gt="pool" -> file pool.

python scripts/make_pool.py --index-version people
python scripts/label_server.py            # chấm 0/1/2 trên trình duyệt -> labels.v1.jsonl

Chạy lại được: truy vấn đã có trong file pool chỉ được thêm ảnh mới (nhãn cũ giữ nguyên).
"""

import argparse
import json
from pathlib import Path

from imgsearch.eval.pooling import available, eval_settings, pool
from imgsearch.eval.testset import load_testset
from imgsearch.retrieval.engine import SearchEngine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--testset", type=Path, default=Path("testsets/manual.v1.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("testsets/pool.v1.jsonl"))
    ap.add_argument("--index-dir", type=Path, default=Path("../data/index"))
    ap.add_argument("--index-version", required=True)
    ap.add_argument("--depth", type=int, default=20)
    args = ap.parse_args(argv)

    engine = SearchEngine(eval_settings(index_dir=args.index_dir, index_version=args.index_version))
    engine.load()
    if engine.store is None or engine.dense is None:
        raise SystemExit("cần index có dense.npz và model dense để gộp đủ cấu hình")

    names = {pid: p.name for pid, p in engine.parser.aliases.persons.items()}
    old = {}
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            old[r["qid"]] = r
    queries = [q for q in load_testset(args.testset) if q.gt == "pool" and not q.intent_refs]
    rows, sizes = [], []
    for q in queries:
        found = pool(engine, q.query, args.depth)
        info = engine.store.get_images(list(found))
        persons = engine.store.persons_of(list(found))
        prev = {i["image_id"]: i for i in old.get(q.id, {}).get("images", [])}
        images = [
            {
                "image_id": image_id,
                "description": info.get(image_id, {}).get("description"),
                "image_url": info.get(image_id, {}).get("image_url"),
                "persons": [names.get(p, p) for p in persons.get(image_id, [])],
                "ranks": ranks,
            }
            for image_id, ranks in found.items()
        ]
        images += [i for k, i in prev.items() if k not in found]  # không bỏ ảnh đã gộp trước
        rows.append({"qid": q.id, "query": q.query, "scenario": q.scenario, "images": images})
        sizes.append(len(images))
        print(f"{q.id} {len(images):3d} ảnh  {q.query}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(
        f"{len(rows)} truy vấn, {sum(sizes)} ảnh cần chấm "
        f"(cấu hình {', '.join(available(engine))}, "
        f"độ sâu {args.depth}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
