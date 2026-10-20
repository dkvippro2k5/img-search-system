# models/ (không commit)

File ONNX cho runtime API (mount `../models:/models:ro` khi chạy Docker).

- `siglip2-b32/`: text encoder của open_clip `ViT-B-32-SigLIP2-256` (`webli`), sinh bằng `part_b/scripts/export_text_encoder.py`: `text.onnx` (fp32, 1,1 GB), `tokenizer.json`, `encoder.json`. Không có thư mục này thì engine tắt dense ảnh và chạy BM25.

Model ảnh chỉ dùng offline (`encode_images.py`, torch) nên không đặt ở đây. Kênh văn bản qwen3 gọi API của mentor, không có file local. Phần A (SCRFD, ArcFace…) đặt model của mình ở thư mục con riêng.
