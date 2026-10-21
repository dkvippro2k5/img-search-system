# Image Data Processing & Retrieval: ảnh người nổi tiếng Việt Nam

Capstone gồm hai phần, mỗi phần một thư mục:

| Thư mục | Nội dung | Trạng thái |
|---|---|---|
| [part_b/](part_b/README.md) | Truy xuất ảnh bằng ngôn ngữ tự nhiên: single / multi-intent / multi-hop, API, web, Docker, benchmark | Đã chạy được: API, web, Docker, benchmark, test set |
| [part_a/](part_a/README.md) | Xử lý và kiểm duyệt ảnh: pipeline T0–T4, dedupe, đúng người, NSFW, watermark, mô hình chi phí | Future work: chỉ có thiết kế và cách nối vào Phần B |

Dữ liệu đầu vào đã được team duyệt tay; Phần B chỉ index ảnh approved. Nếu làm Phần A sau này, hai phần nối với nhau qua file trong `data/features/` (khóa ảnh `sha256`), không import code.

```
capstone/   PLAN.md, SCENARIOS.md: kế hoạch và kịch bản
docs/       tài liệu kỹ thuật nộp cuối
data/       dữ liệu dùng chung (không commit)
models/     file ONNX (không commit)
reports/    báo cáo: eval.md (dev + test), perf.md (benchmark)
part_b/     truy xuất ảnh: code, test, web, Docker
part_a/     kiểm duyệt ảnh: future work (chỉ README)
```

## Quyền riêng tư

Dữ liệu là ảnh người thật: không commit ảnh, CSV, embedding, index hay model. Bucket S3 để private, ảnh hiển thị qua presigned URL ngắn hạn. Hệ thống không có chức năng tìm bằng ảnh khuôn mặt.
