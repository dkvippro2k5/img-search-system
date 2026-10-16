# Test set

Định dạng JSONL, có phiên bản (`queries.v1.jsonl`, ...). Mỗi dòng:

```json
{"id": "S-001", "type": "single|multi_intent|multi_hop", "scenario": "E1",
 "query": "...", "intents": [], "expected_persons": [], "relation_path": [],
 "gt": "rule" | ["<sha256>", ...], "split": "dev|test", "notes": ""}
```

- `scenario`: mã kịch bản trong `capstone/SCENARIOS.md` (L1, E2, M1, H3...).
- `gt: "rule"`: ground truth sinh tự động = mọi ảnh approved của `expected_persons`.
- Multi-hop phủ định: `expected_persons: []`, đáp án đúng là "không có".
- Chỉ tinh chỉnh trên `split=dev`; `test` chỉ chạy khi báo cáo.
- Cạnh quan hệ dùng trong multi-hop phải được kiểm tra tay; không lấy đáp án từ tài liệu.

`queries.example.jsonl` chỉ minh họa định dạng (chưa có person_id/ground truth thật).
`bench_queries.json` là danh sách truy vấn cho k6 (50% single, 30% multi-intent, 20% multi-hop).
