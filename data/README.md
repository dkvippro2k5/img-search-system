# data/ (không commit, xem .gitignore)

Ảnh người thật: không đẩy ảnh, CSV, embedding hay index lên GitHub.

| Thư mục | Nội dung |
|---|---|
| `raw/` | `approved-results (1..5).csv` từ mentor + file crawl đã duyệt của team; dữ liệu mới kèm caption VLM |
| `labels/` | Nhãn duyệt tay (auto_flag, nhãn cuối, lý do reject), tập tự gán nhãn watermark/NSFW/nhạy cảm |
| `merged/` | Đầu ra `scripts/merge_data.py`: dữ liệu hợp nhất, khóa `(entity_id, sha256)` |
| `features/` | Parquet / `.npy` đặc trưng và điểm của Phần A (T1-T3), embedding mặt và ảnh |
| `index/v{INDEX_VERSION}/` | `meta.sqlite`, `text.faiss`, (tùy chọn) `image.faiss`, `manifest.json` |
| `thumbs/` | Thumbnail WebP: `{sha256[:2]}/{sha256}.webp` |
