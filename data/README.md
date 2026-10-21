# data/ (không commit, xem .gitignore)

Ảnh người thật: không đẩy ảnh, CSV, embedding hay index lên GitHub.

| Thư mục | Nội dung |
|---|---|
| `raw/` | `people-*.csv` (ảnh người đã duyệt của team, đọc bởi `part_b/scripts/merge_data.py`); `raw/fetched/`: ảnh tải bù cho dòng thiếu sha256 (`fetch_log.jsonl` + `images/{sha[:2]}/`) |
| `labels/` | Nhãn duyệt tay (auto_flag, nhãn cuối, lý do reject), tập tự gán nhãn watermark/NSFW/nhạy cảm |
| `merged/` | Đầu ra `scripts/merge_data.py`: `people/` (JSONL, định dạng ở `part_b/src/imgsearch/data/dataset.py`, kèm `merge_report.json`); `golden-*/` từ `scripts/golden_to_dataset.py` |
| `features/` | `image_embeddings.{npy,ids.txt,json}` (SigLIP2, `encode_images.py`), `text_embeddings.*` (qwen3, `encode_texts.py`), `shards/` + `text_shards/` để chạy tiếp; sau này thêm đặc trưng của Phần A |
| `index/v{INDEX_VERSION}/` | `meta.sqlite` (metadata + KG), `lexical.npz` (BM25), `dense.npz` (vector ảnh), `text_dense.npz` (vector văn bản), `manifest.json` |
| `golden/` | Bộ golden (README ở đó): hệ thống chỉ đọc `metadata.json` + `images/`, `gold/` chỉ dùng khi chấm |
| `thumbs/` | Thumbnail WebP 320 px (`part_b/scripts/make_thumbs.py`), đẩy lên S3; id có thumbnail ghi ở `index/v{ver}/thumbs.txt` |

## Định dạng file embedding (`features/`)

Mỗi bộ embedding gồm ba file cùng tên gốc, `build_index.py --features` đọc (`build_meta.load_embeddings`):

| File | Nội dung |
|---|---|
| `{name}.npy` | Ma trận N×d, float16 hoặc float32, đã chuẩn hóa L2 |
| `{name}.ids.txt` | `image_id` (sha256) mỗi dòng, cùng thứ tự với hàng của `.npy` |
| `{name}.json` | `{"model": "...", ...}`; tên model được ghi vào `manifest.json`, API tắt kênh dense khi khác `DENSE_MODEL` |

- `image_embeddings`: SigLIP2 (`siglip2-b32` = open_clip `ViT-B-32-SigLIP2-256` / `webli`), sinh bằng `part_b/scripts/encode_images.py`. Phải cùng model với text encoder ONNX của API.
- `text_embeddings`: qwen3-embedding-4b trên văn bản BM25 của ảnh (float16, 2560 chiều), sinh bằng `part_b/scripts/encode_texts.py`.

Đầu ra dự kiến của Phần A (kiểm duyệt, khuôn mặt) cũng ghi vào đây, khóa `sha256`: xem [part_a/README.md](../part_a/README.md).
