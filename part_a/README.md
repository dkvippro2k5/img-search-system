# Phần A: xử lý và kiểm duyệt ảnh (future work)

Phần A **không làm trong capstone này**. Dữ liệu đầu vào đã được cả team duyệt tay (approved/rejected kèm lý do), Phần B chỉ index ảnh approved, nên thời gian dồn cho truy xuất ([part_b/](../part_b/README.md)). Thiết kế đầy đủ vẫn giữ ở [PLAN.md mục 3](../capstone/PLAN.md) làm hướng phát triển tiếp.

## Phần B đang thiếu gì vì chưa có Phần A

| Hạn chế hiện tại | Phần A sẽ cung cấp | Điểm nối đã có trong Phần B |
|---|---|---|
| Lọc ảnh nhạy cảm chặn 0 ảnh vì chưa có điểm; chỉ chặn được truy vấn bằng từ khóa (`retrieval/guard.py`, luôn lách được) | `nsfw_score`, `sensitive_flag` mỗi ảnh | Cột cùng tên trong bảng `image`, `store.unsafe_image_ids`, ngưỡng `NSFW_BLOCK_SCORE` |
| Ảnh gần trùng (khác sha256) hiện hai lần (Q3) | `dup_of`: sha256 ảnh gốc, từ pHash + cosine embedding | `dataset.py` có trường `dup_of`; `build_meta.py` bỏ ảnh có `dup_of` |
| Person bị tách đôi (`B Ray`/`BRay`, "Hà Hồ"/Hồ Ngọc Hà) và entity gán nhầm người; `entity_merge.py` v0 chỉ dùng tên | Tâm ArcFace mỗi entity, gộp entity thành person bằng khuôn mặt | `data/entity_merge.py` |
| Ảnh mờ, nhỏ, có watermark không bị hạ hạng (P5, Q1, Q2) | `quality_flags`, `watermark_score` | Cột cùng tên trong bảng `image`, chưa dùng khi xếp hạng |

## Hướng làm (tóm tắt PLAN 3.3–3.6)

- **Pipeline nhiều tầng, rẻ trước đắt sau**:
  - T0: giải mã, kích thước, tỷ lệ khung hình, sha256.
  - T1: pHash/dHash, độ mờ, dedupe tuyệt đối và gần trùng.
  - T2: chạy model nặng **một lần** (SCRFD + ArcFace cho mặt, SigLIP cho ảnh).
  - T3: bộ phân loại rẻ trên embedding (đúng người, NSFW, watermark, nhạy cảm).
  - T4: VLM có ngân sách trần (1–3% ảnh); không chắc thì chuyển hàng đợi người duyệt.
- **Cải tiến chính**: bộ hiệu chỉnh "đúng người" học từ khoảng 18k nhãn duyệt tay (đặc trưng: cosine tới tâm entity, kích thước cụm, diện tích mặt, khớp tên trong tiêu đề), chọn điểm vận hành theo precision mục tiêu. Mốc cần vượt là pipeline cũ tự quyết 72,9% ảnh với độ đúng 93,4% [Đã đo, PLAN 1.3].
- **Chính trị / nhạy cảm mức nhẹ**: chỉ gắn cờ cho người quyết, không tự loại.
- **Chi phí**: `Chi_phí(N) = N × Σ f_i × c_i` (f_i: tỷ lệ ảnh tới tầng i, c_i: chi phí mỗi ảnh ở tầng i). Đo f_i và thông lượng trên mẫu, ngoại suy cho 10–100M ảnh, so với phương án cho mọi ảnh qua VLM lớn.

## Cách nối vào Phần B

Ranh giới là **file, không import code**. Phần A ghi vào `data/features/`, khóa ảnh là `sha256`:

| File | Nội dung |
|---|---|
| `moderation.parquet` | `image_id`, `status` (approved/rejected/review), `quality_flags`, `nsfw_score`, `watermark_score`, `sensitive_flag`, `phash`, `dup_of` |
| `face_centroids.npy` + `face_centroids.parquet` (`entity_id`, `row`) | Tâm ArcFace mỗi entity |
| `entity_person.parquet` | `entity_id`, `person_id`, `merge_method` (nếu Phần A tự gộp entity) |

Hiện `part_b/src/imgsearch/data/merge.py` **chưa đọc** các file trên. Cần thêm bước ghép chúng vào dataset JSONL (các trường đã có sẵn trong `dataset.py` và `schema.sql`), rồi dựng lại index.

Embedding ảnh SigLIP2 thì Phần B đã tự sinh (`part_b/scripts/encode_images.py`, định dạng ở [data/README.md](../data/README.md)). Tầng T2 của Phần A nên dùng lại file này thay vì mã hóa lần nữa.
