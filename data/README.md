# data/ (không commit, xem .gitignore)

Ảnh người thật: không đẩy ảnh, CSV, embedding hay index lên GitHub.

| Thư mục | Nội dung |
|---|---|
| `raw/` | `people-*.csv` (ảnh người đã duyệt của team, đọc bởi `part_b/scripts/merge_data.py`); `raw/fetched/`: ảnh tải bù cho dòng thiếu sha256 (`fetch_log.jsonl` + `images/{sha[:2]}/`) |
| `labels/` | Nhãn duyệt tay (auto_flag, nhãn cuối, lý do reject), tập tự gán nhãn watermark/NSFW/nhạy cảm |
| `merged/` | Đầu ra `scripts/merge_data.py`: dữ liệu hợp nhất, khóa `(entity_id, sha256)` |
| `features/` | Parquet / `.npy` đặc trưng và điểm của Phần A (T1-T3), embedding mặt và ảnh |
| `index/v{INDEX_VERSION}/` | `meta.sqlite`, `lexical.npz` (BM25), `text.faiss`, (tùy chọn) `image.faiss`, `manifest.json` |
| `thumbs/` | Thumbnail WebP: `{sha256[:2]}/{sha256}.webp` |
