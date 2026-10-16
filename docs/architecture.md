# Tài liệu kỹ thuật

Khung cho tài liệu nộp cuối. Sơ đồ gốc và phân tích phương án nằm trong [../capstone/PLAN.md](../capstone/PLAN.md); khi điền mục nào thì ghi số liệu kèm nhãn **[Đã đo]** / **[Ước tính]**.

Nhật ký làm việc chi tiết: [worklog/](worklog/).

## 1. Dữ liệu
- Nguồn, số ảnh trước/sau dedupe, số entity -> số person.
- Schema: [part_b/src/imgsearch/data/schema.sql](../part_b/src/imgsearch/data/schema.sql).

## 2. Phần A: pipeline kiểm duyệt
- Sơ đồ T0–T4 (PLAN 3.3).
- Bảng kết quả so với baseline (tự quyết 72,9%, đúng 93,4%).
- Thông lượng từng tầng, mô hình chi phí, ngoại suy 10–100M.

## 3. Phần B: truy xuất
- Luồng truy vấn (PLAN 4.4), đồ thị tri thức và quy tắc bằng chứng.
- Kết quả đánh giá theo loại truy vấn: `reports/eval.md`.

## 4. Hiệu năng
- Phần cứng, cấu hình container, kịch bản k6, p50/p95/p99: `reports/perf.md`.

## 5. Triển khai
- docker compose (api + MinIO), healthcheck, S3/R2, CI.

## 6. Hạn chế
- Phủ định, đếm số lượng, thời gian, ảnh sai ngữ cảnh, độ phủ KG, thiên lệch test set.
