# Báo cáo đánh giá truy xuất (Phần B)

Gộp hai lần chạy `python -m imgsearch.eval.run_testset --index-version people`: split `dev` (dùng để tinh chỉnh, 20/10/2026 16:55) và split `test` (**chạy một lần**, 20/10/2026 21:12, sau khi đã chốt mọi tham số). Chi tiết từng truy vấn nằm trong `eval_dev.md`, `eval_test.md` (sinh lại bằng lệnh trên, không commit). Mọi số trong file là **[Đã đo]**.

- Index `people`: 90.056 ảnh, 5.653 person.
- Test set `testsets/queries.v1.jsonl`: 161 truy vấn (dev 82, test 79). Sinh tự động từ `meta.sqlite` (tên, không dấu, trùng tên, multi-intent, multi-hop từ cạnh KG) và viết tay (bối cảnh, phủ định).
- Truy vấn bối cảnh ("Đen Vâu hát trong concert", "hoa hậu đội vương miện"): gộp top-20 của các cấu hình rồi chấm tay 0/1/2, 1.344/1.344 ảnh có nhãn, một người chấm.
- Cấu hình chính `w2+t0.5`: BM25 + dense ảnh SigLIP2 (trọng số RRF 2) + dense văn bản qwen3 của mentor (0,5). Lần chạy test không có lỗi mentor nào (0 cảnh báo trong log).

## Tổng hợp

| Split | Truy vấn | Hành vi đúng | Có điểm | P@10 | P@10 (nhãn 2) | P@R | R@50 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|---|---|---|
| dev | 82 | 80/82 | 74 | 0,89 | 0,75 | 0,97 | 0,93 | 0,94 | 0,97 |
| **test** | **79** | **77/79** | **72** | **0,87** | **0,75** | **0,94** | **0,89** | **0,92** | **0,96** |

Theo loại truy vấn (test):

| Loại | Truy vấn | Hành vi đúng | P@10 | R@50 | nDCG@10 |
|---|---|---|---|---|---|
| single | 41 | 40/41 | 0,81 | 0,82 | 0,87 |
| multi-intent | 15 | 15/15 | 0,94 | 0,96 | 0,95 |
| multi-hop | 23 | 22/23 | 0,95 | 1,00 | 1,00 |

- Multi-intent (test): độ phủ ý định@10 1,00; ý định ít nhất vẫn chiếm 46% top 10; tách ý định / liên kết người đúng 12/13.
- Multi-hop (test): người gốc đúng 23/23, người đích đúng 17/17, từ chối đúng ở truy vấn phủ định 5/6, bịa 1/23.

## So sánh cấu hình trên truy vấn bối cảnh (chấm tay, 18 truy vấn mỗi split)

| Cấu hình | dev P@10 (nhãn 2) | dev R@50 | dev nDCG@10 | test P@10 (nhãn 2) | test R@50 | test nDCG@10 |
|---|---|---|---|---|---|---|
| bm25 | 0,53 | 0,61 | 0,69 | 0,59 | 0,58 | 0,70 |
| w2 (BM25 + ảnh) | 0,72 | 0,76 | 0,84 | 0,75 | 0,68 | 0,80 |
| w8 (ảnh át BM25) | 0,70 | 0,68 | 0,83 | 0,72 | 0,60 | 0,78 |
| **w2 + t0,5 (mặc định)** | **0,75** | **0,80** | **0,85** | **0,76** | **0,72** | **0,81** |

- Dense ảnh là phần cải thiện chính: nDCG@10 bối cảnh +0,15 (dev) và +0,10 (test) so với chỉ BM25. Thứ tự các cấu hình giữ nguyên từ dev sang test.
- Kênh văn bản chỉ thêm ít: +0,01 nDCG@10 ở cả hai split, chủ yếu tăng R@50 (+0,04).
- Trên các truy vấn không có phần mô tả (chỉ tên, multi-hop), các cấu hình cho kết quả như nhau vì dense không được gọi.
- Số đo bối cảnh của test thấp hơn dev (nDCG@10 0,81 so với 0,85). Đây là mức kỳ vọng khi dùng hệ thống trên truy vấn mới.

## Lỗi còn lại

| Split | Truy vấn | Loại lỗi | Nguyên nhân |
|---|---|---|---|
| dev | “vo cua Tran Thanh” | liên kết người gốc sai | không dấu + trùng tên Trấn Thành / Trần Thành |
| dev | “chồng của Bích Phương” | liên kết người gốc sai | trùng tên: hệ thống hỏi lại thay vì chọn ca sĩ |
| dev | “chồng của Diệu Nhi” | bịa | cạnh sai Đức Phúc – Diệu Nhi lấy từ tiêu đề bài báo |
| dev | “Sếp Tùng” | liên kết người sai | thiếu biệt danh trong alias |
| test | “vợ của Đức Phúc” | bịa | cùng cạnh sai Đức Phúc – Diệu Nhi |
| test | “Đinh Thị Hoa Mơ” (và vế trong “Võ Trường Thành và Đinh Thị Hoa Mơ”) | hành vi sai | tên không có trong dữ liệu, nhưng khi bỏ dấu “Đinh Thị” trùng “Đình Thi” nên khớp nhầm Nguyễn Đình Thi |
| test | “Quang Hải ăn mừng bàn thắng”, “Bích Phương trong MV” | trùng tên | chia nhóm theo từng người; phần mô tả chưa được dùng để chọn người (cầu thủ, ca sĩ) |
| test | “Hà Hồ” | liên kết người sai | dữ liệu tách đôi: một person riêng tên “Hà Hồ” (76 ảnh) được chọn thay cho Hồ Ngọc Hà, 0/10 ảnh đúng |
| test | “Nguyễn Đức Cường” | trùng tên | tên thật của Đen Vâu trùng hai người khác; chia nhóm, nhóm Đen Vâu đứng đầu (nDCG@10 0,53) |

Theo quy tắc đánh giá, các lỗi trên split test không được sửa rồi chấm lại; chỉ ghi nhận làm hạn chế.

## Latency trong lúc chấm (engine, không HTTP, một luồng)

| Split | p50 | p95 | max |
|---|---|---|---|
| dev | 4,7 ms | 301,8 ms | 920,5 ms |
| test | 19,6 ms | 703,6 ms | 1.012,6 ms |

p95 cao là do lời gọi mạng tới API mentor (kênh văn bản), dao động theo tải bên mentor. Không có kênh này thì p95 của engine là 45 ms. Đo tải HTTP: [perf.md](perf.md).
