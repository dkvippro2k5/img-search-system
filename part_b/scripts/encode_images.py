"""Mã hóa ảnh bằng model ảnh–văn bản (open_clip) -> data/features/image_embeddings.* (bước 3).

    python -X utf8 scripts/encode_images.py                          # ảnh thật (merged/people)
    python -X utf8 scripts/encode_images.py --golden ../data/golden \\
        --out ../data/features/golden                                # bộ golden
Sau đó: build_index.py --features <out>. Chạy tiếp được: ghi từng shard vào {out}/shards/,
lần sau bỏ qua ảnh đã có. Ảnh hỏng / không mở được thì bỏ và đếm.
Đầu ra (định dạng ở ../data/README.md): image_embeddings.npy (float16, N×d, đã chuẩn hóa),
image_embeddings.ids.txt (image_id mỗi dòng), image_embeddings.json (model, thời gian).
Cần extra "embed" (torch, open_clip). Không dùng ở runtime API.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from imgsearch.config import get_settings
from imgsearch.data.fetch import image_files, read_log

MODELS = {"siglip2-b32": ("ViT-B-32-SigLIP2-256", "webli")}
SHARD = 2048


def targets_from_merged(merged: Path, fetch_log: Path) -> list[tuple[str, str]]:
    with (merged / "images.jsonl").open(encoding="utf-8") as fh:
        images = [json.loads(line) for line in fh if line.strip()]
    approved = [i for i in images if i.get("status") == "approved"]
    return image_files(approved, read_log(fetch_log))


def targets_from_golden(golden: Path) -> list[tuple[str, str]]:
    rows = json.loads((golden / "metadata.json").read_text(encoding="utf-8"))
    return [(r["sha256"], str(golden / r["file"])) for r in rows if r.get("file")]


def main(argv: list[str] | None = None) -> None:
    import open_clip
    import torch
    from PIL import Image

    data = get_settings().data_dir
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="siglip2-b32", choices=list(MODELS))
    ap.add_argument("--merged", type=Path, default=data / "merged" / "people")
    ap.add_argument("--fetch-log", type=Path, default=data / "raw/fetched/fetch_log.jsonl")
    ap.add_argument("--golden", type=Path, default=None, help="mã hóa bộ golden thay vì merged")
    ap.add_argument("--out", type=Path, default=data / "features")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=0, help="số luồng torch (0: mặc định)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)

    targets = (
        targets_from_golden(args.golden)
        if args.golden
        else targets_from_merged(args.merged, args.fetch_log)
    )
    targets = list(dict(targets).items())[: args.limit]  # bỏ trùng image_id
    shard_dir = args.out / "shards" / args.model
    shard_dir.mkdir(parents=True, exist_ok=True)
    done = set()
    for ids_file in shard_dir.glob("*.ids.txt"):
        done |= set(ids_file.read_text(encoding="utf-8").split())
    todo = [t for t in targets if t[0] not in done]
    print(f"{len(targets)} ảnh có file, {len(done)} đã mã hóa, còn {len(todo)}", flush=True)

    arch, pretrained = MODELS[args.model]
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval()

    def load(t: tuple[str, str]):
        try:
            im = Image.open(t[1])
            im.draft("RGB", (512, 512))  # JPEG: giải mã sẵn ở độ phân giải nhỏ, nhanh hơn nhiều
            return t[0], preprocess(im.convert("RGB"))
        except Exception:
            return t[0], None

    t0, n_bad, shard_no = time.perf_counter(), 0, len(list(shard_dir.glob("*.npy")))
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(todo), SHARD):
            loaded = list(pool.map(load, todo[start : start + SHARD]))
            ok = [(i, x) for i, x in loaded if x is not None]
            n_bad += len(loaded) - len(ok)
            vecs = []
            with torch.inference_mode():
                for b in range(0, len(ok), args.batch):
                    x = torch.stack([x for _, x in ok[b : b + args.batch]])
                    v = torch.nn.functional.normalize(model.encode_image(x), dim=-1)
                    vecs.append(v.numpy().astype(np.float16))
            if ok:
                name = f"part-{shard_no:04d}"
                np.save(shard_dir / f"{name}.npy", np.concatenate(vecs))
                (shard_dir / f"{name}.ids.txt").write_text(
                    "\n".join(i for i, _ in ok), encoding="utf-8"
                )
                shard_no += 1
            n = start + len(loaded)
            rate = n / (time.perf_counter() - t0)
            eta = (len(todo) - n) / rate / 60 if rate else 0
            print(f"  {n}/{len(todo)}  {rate:.1f} ảnh/s  còn ~{eta:.0f} phút", flush=True)

    # Gộp mọi shard thành file đầu ra.
    parts = sorted(shard_dir.glob("*.npy"))
    emb = np.concatenate([np.load(p) for p in parts]) if parts else np.zeros((0, 0), np.float16)
    ids = [i for p in parts for i in p.with_suffix(".ids.txt").read_text(encoding="utf-8").split()]
    base = args.out / "image_embeddings"
    np.save(base.with_suffix(".npy"), emb)
    base.with_suffix(".ids.txt").write_text("\n".join(ids), encoding="utf-8")
    info = {
        "model": args.model,
        "open_clip": [arch, pretrained],
        "n": len(ids),
        "dim": int(emb.shape[1]) if len(emb) else 0,
        "bad_images_this_run": n_bad,
        "seconds_this_run": round(time.perf_counter() - t0, 1),
    }
    base.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
