# Image Data Processing & Retrieval: ảnh người nổi tiếng Việt Nam

Capstone gồm hai phần, mỗi phần một thư mục:

| Thư mục | Nội dung | Trạng thái |
|---|---|---|
| [part_a/](part_a/README.md) | Xử lý và kiểm duyệt ảnh: pipeline T0–T4, dedupe, đúng người, NSFW, watermark, mô hình chi phí | Prototype đã có, đưa vào đây |
| [part_b/](part_b/README.md) | Truy xuất ảnh bằng ngôn ngữ tự nhiên: single / multi-intent / multi-hop, API, web, Docker, benchmark | Phần chính đang phát triển |

Hai phần nối với nhau qua file trong `data/` (khóa ảnh `sha256`): Phần A ghi điểm kiểm duyệt và embedding, Phần B đọc để dựng index.

```
capstone/   PLAN.md, SCENARIOS.md: kế hoạch và kịch bản
docs/       tài liệu kỹ thuật nộp cuối
data/       dữ liệu dùng chung (không commit)
models/     file ONNX (không commit)
reports/    báo cáo sinh tự động: perf.md, eval.md, kết quả Phần A
part_a/  part_b/
```

## Quyền riêng tư

Dữ liệu là ảnh người thật: không commit ảnh, CSV, embedding, index hay model. Bucket S3 để private, ảnh hiển thị qua presigned URL ngắn hạn. Hệ thống không có chức năng tìm bằng ảnh khuôn mặt.
