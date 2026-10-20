"""Xuất text encoder của model ảnh–văn bản (open_clip) sang ONNX cho runtime API (không torch).

    python -X utf8 scripts/export_text_encoder.py                  # siglip2-b32 -> ../models/
Ghi {out}/{name}/: text.onnx (fp32), tokenizer.json, encoder.json (tên model, độ dài ngữ cảnh,
pad/eos id, chiều vector). Không lượng tử hóa: int8 động lệch quá nhiều với SigLIP2 (cosine với
fp32 chỉ 0,83–0,94, đo 19/10/2026). Kiểm tra ngay sau khi xuất:
id token của `tokenizers` khớp open_clip, vector ONNX gần trùng vector torch.
Cần extra "embed" (torch, open_clip, transformers, onnx).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from imgsearch.retrieval.dense import CHECK_TEXTS, OnnxTextEncoder

MODELS = {"siglip2-b32": ("ViT-B-32-SigLIP2-256", "webli")}


def main(argv: list[str] | None = None) -> None:
    import open_clip
    import torch

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default="siglip2-b32", choices=list(MODELS))
    ap.add_argument("--out", type=Path, default=Path("../models"))
    args = ap.parse_args(argv)
    arch, pretrained = MODELS[args.name]
    target = args.out / args.name
    target.mkdir(parents=True, exist_ok=True)

    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval()
    tok = open_clip.get_tokenizer(arch)
    tok.tokenizer.backend_tokenizer.save(str(target / "tokenizer.json"))
    ctx = tok.context_length
    meta = {
        "name": args.name,
        "open_clip": [arch, pretrained],
        "context_length": ctx,
        "pad_id": tok.tokenizer.pad_token_id,
        "eos_id": tok.tokenizer.eos_token_id,
    }
    with torch.no_grad():
        meta["dim"] = int(model.encode_text(tok(["x"])).shape[-1])
    if getattr(model, "logit_bias", None) is not None:  # SigLIP: p = sigmoid(scale * cos + bias)
        meta["logit_scale"] = float(model.logit_scale.exp())
        meta["logit_bias"] = float(model.logit_bias)
    (target / "encoder.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            return torch.nn.functional.normalize(self.m.encode_text(input_ids), dim=-1)

    fp32 = target / "text.onnx"
    torch.onnx.export(
        Text(model),
        (tok(["xin chào"]),),
        str(fp32),
        input_names=["input_ids"],
        output_names=["embedding"],
        dynamic_axes={"input_ids": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    # Kiểm tra: tokenizer và vector.
    with torch.no_grad():
        ref_ids = tok(CHECK_TEXTS).numpy()
        ref = Text(model)(torch.from_numpy(ref_ids)).numpy()
    enc = OnnxTextEncoder(target, onnx_file=fp32.name)
    assert (enc.token_ids(CHECK_TEXTS) == ref_ids).all(), "id token lệch open_clip"
    cos = (enc.encode(CHECK_TEXTS) * ref).sum(1)
    print(f"{fp32.name}: {fp32.stat().st_size / 1e6:.0f} MB, cos với torch min {cos.min():.4f}")
    assert cos.min() > 0.999, cos
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
