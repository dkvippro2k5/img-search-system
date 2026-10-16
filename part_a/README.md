# Phần A: xử lý và kiểm duyệt ảnh

Đặt prototype đã có vào thư mục này (pipeline T0–T4, SCRFD + ArcFace, rule + gom cụm, bộ hiệu chỉnh đúng người...). Thiết kế và thực nghiệm: [../capstone/PLAN.md](../capstone/PLAN.md) mục 3.

- `configs/pipeline.yaml`: ngưỡng các tầng (giá trị ban đầu là giả định).
- `cost_model.py`: công thức chi phí và ngoại suy 10–100M ảnh (PLAN 3.6).

## Đầu ra mà Phần B cần

Phần B chỉ đọc file, không import code Phần A. Ghi vào `../data/` với khóa ảnh là `sha256`:

| File | Cột tối thiểu | Phần B dùng để |
|---|---|---|
| `data/features/moderation.parquet` | `image_id` (sha256), `status` (approved/rejected/review), `quality_flags`, `nsfw_score`, `watermark_score`, `sensitive_flag`, `phash`, `dup_of` (sha256 ảnh gốc hoặc null) | Chỉ index ảnh approved; lọc an toàn (Q4); gộp ảnh gần trùng (Q3); hạ hạng ảnh kém (Q1, Q2) |
| `data/features/face_centroids.npy` + `face_centroids.parquet` (`entity_id`, `row`) | tâm ArcFace mỗi entity | Gộp entity thành person bằng mặt (`part_b/src/imgsearch/data/entity_merge.py`) |
| `data/features/image_embeddings.npy` + `.parquet` (`image_id`, `row`) (tùy chọn) | embedding CLIP/SigLIP từ T2 | Dense ảnh ở Phần B, chỉ khi bake-off cho thấy có lợi |

Nếu prototype đã gộp entity thành person thì xuất thêm `data/features/entity_person.parquet` (`entity_id`, `person_id`, `merge_method`) và Phần B sẽ dùng luôn.
