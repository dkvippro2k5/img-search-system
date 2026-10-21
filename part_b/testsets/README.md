# Test set

Định dạng JSONL, có phiên bản (`queries.v1.jsonl`, ...). Đọc/ghi bằng `imgsearch/eval/testset.py`.
Mỗi dòng:

```json
{"id": "S-N-01", "type": "single|multi_intent|multi_hop", "scenario": "E1",
 "query": "...", "split": "dev|test",
 "gt": "rule" | "pool" | ["<sha256>", ...] | [],
 "expected_behavior": "results",
 "expected_persons": [], "intents": [], "intent_persons": [[...], [...]],
 "intent_refs": [], "relation_path": [], "anchor_persons": [], "notes": ""}
```

- `scenario`: mã kịch bản trong `capstone/SCENARIOS.md` (L1, E2, M1, H3...).
- `gt`:
  - `"rule"`: mọi ảnh approved của `expected_persons` (multi-intent: theo từng vế ở
    `intent_persons`). Gồm cả các person bị tách mảnh cùng tên (ví dụ `p_mck` + `p_rpt_mck`).
  - `"pool"`: nhãn 0/1/2 chấm tay trên ảnh gộp top-20 của các cấu hình (`labels.v1.jsonl`).
    Multi-intent có bối cảnh dùng `intent_refs`: mỗi vế lấy nhãn của một truy vấn single cùng
    câu chữ.
  - danh sách sha256: tập ảnh đúng cố định (M2: ảnh có cả hai người).
  - `[]`: không có ảnh đúng (phủ định, tên bịa).
- `expected_behavior`: `results`, `no_evidence` (H3), `empty_or_notice` (E6),
  `disambiguate` (E2), `disambiguate_or_no_evidence` (H7), `partial_with_notice` (M5),
  `no_co_occurrence` (M2 khi không có ảnh chung).
- `anchor_persons`: người gốc của truy vấn multi-hop (đo độ đúng liên kết người gốc).
- Chỉ tinh chỉnh trên `split=dev`; `test` chỉ chạy khi báo cáo.
- Cạnh quan hệ dùng trong multi-hop phải được kiểm tra tay; không lấy đáp án từ tài liệu.

## File

| File | Nội dung | Sinh bằng |
|---|---|---|
| `manual.v1.jsonl` | viết tay: bối cảnh (S-C, M-C), biệt danh (L3), multi-hop (H), phủ định (H-N) | tay |
| `queries.v1.jsonl` | test set đầy đủ = sinh tự động + `manual.v1.jsonl` | `scripts/make_testset.py` |
| `pool.v1.jsonl` | ảnh cần chấm cho truy vấn `gt: "pool"` | `scripts/make_pool.py` |
| `labels.v1.jsonl` | nhãn `{qid, image_id, grade}`; dòng sau ghi đè dòng trước | `scripts/label_server.py` |
| `bench_queries.json` | truy vấn cho k6 (50% single, 30% multi-intent, 20% multi-hop) | `scripts/make_testset.py` |
| `relations_check.v1.md` | kiểm tay 27 cạnh quan hệ dùng trong test set multi-hop (20/10/2026) | tay |

Không sửa tay `queries.v1.jsonl`: sửa `manual.v1.jsonl` hoặc `imgsearch/eval/testgen.py` rồi sinh lại.

## Quy trình (trong `part_b/`)

```bash
python scripts/make_testset.py --index-version people     # -> queries.v1.jsonl
python scripts/make_pool.py --index-version people        # -> pool.v1.jsonl (chạy lại được)
python scripts/label_server.py                            # chấm ở http://127.0.0.1:8765
python -m imgsearch.eval.run_testset --index-version people --split dev   # -> ../reports/eval_dev.md
python -m imgsearch.eval.run_testset --index-version people --split test  # chỉ khi báo cáo
```

Sinh tự động (`testgen.py`) chỉ đọc `meta.sqlite`, không gọi engine, để không chọn thiên về câu
hệ thống đã làm đúng. Chọn person phân tầng theo nghề × số ảnh; bỏ tổ chức / nhãn nhóm lẫn trong
bảng person ("REE Corporation", "HCV SEA Games") và tên > 4 âm tiết (thường là nhãn ghép hai người).

Thêm hệ thống mới (ví dụ kênh văn bản, reranker) thì chạy lại `make_pool.py` với cấu hình đó để
ảnh nó trả về cũng được chấm; ảnh chưa chấm bị tính là không liên quan.
