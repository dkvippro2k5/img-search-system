# Phần B: truy xuất ảnh

Tìm ảnh người nổi tiếng Việt Nam bằng ngôn ngữ tự nhiên: single-intent, multi-intent, multi-hop; latency < 3 s trên ≥ 80.000 ảnh. Thiết kế: [../capstone/PLAN.md](../capstone/PLAN.md) mục 4–7; kịch bản: [../capstone/SCENARIOS.md](../capstone/SCENARIOS.md).

```
src/imgsearch/
  api/        FastAPI: /search, /healthz, /metrics
  retrieval/  query_parser, lexical (FTS5), dense (ONNX + FAISS), fusion (RRF), multi_intent, multi_hop
  kg/         quan hệ từ Wikidata và tiêu đề bài báo, mỗi cạnh có bằng chứng
  data/       merge CSV, entity_merge (entity -> person), schema.sql
  eval/       chỉ số, chạy test set
  storage/    S3 / MinIO / R2
  text/       chuẩn hóa tiếng Việt
scripts/      merge_data, make_thumbs, build_index, upload_s3
testsets/     test case JSONL + truy vấn benchmark
bench/        k6 + sinh ../reports/perf.md
web/          giao diện tĩnh TypeScript
```

## Chạy (trong thư mục part_b)

```bash
cp .env.example .env
python -m pip install -e ".[dev]"     # thêm ,index để dựng index
cd web && npm install && npm run build && cd ..
uvicorn imgsearch.api.main:app --reload   # http://localhost:8000
python -m pytest -q
```

Dựng index từ dữ liệu ở `../data` (đầu ra Phần A xem [../part_a/README.md](../part_a/README.md)):

```bash
python scripts/merge_data.py && python scripts/make_thumbs.py && python scripts/build_index.py
```

## Docker

```bash
docker compose up -d --build                     # minio + init + api
docker compose --profile bench run --rm bench    # k6
python bench/make_report.py                      # -> ../reports/perf.md
```
