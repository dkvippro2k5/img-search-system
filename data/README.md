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
| `thumbs/` | Chưa dùng: web hotlink `image_url` gốc |
