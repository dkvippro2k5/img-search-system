# CLAUDE.md

Capstone cá nhân "Image Data Processing & Retrieval" trên tập ảnh **người nổi tiếng Việt Nam** (hạn nộp 22/10/2026). Nguồn sự thật về yêu cầu và quyết định thiết kế:

- [capstone/PLAN.md](capstone/PLAN.md): kế hoạch, phương án, schema, lịch, rủi ro. Đọc mục 11 (đề xuất mặc định) trước khi đổi kiến trúc.
- [capstone/SCENARIOS.md](capstone/SCENARIOS.md): kịch bản người dùng có mã (L1, E2, M1, H3, Q4, R2, X1...). Test case và comment trong code tham chiếu các mã này.

## Bố cục repo

- `part_b/`: **trọng tâm hiện tại**. Truy xuất ảnh: package `imgsearch`, API, web, Docker, benchmark, test set. Mặc định mọi việc code là ở đây.
- `part_a/`: kiểm duyệt ảnh. Người dùng đã có prototype riêng và sẽ đưa vào; không viết lại hay tái cấu trúc Phần A trừ khi được yêu cầu.
- Dùng chung ở gốc: `data/`, `models/`, `reports/`, `docs/`, `capstone/`.
- **Ranh giới A/B là file, không phải import**: Phần A ghi `data/features/*.parquet|.npy` (khóa `sha256`), Phần B đọc. Hợp đồng file ở [part_a/README.md](part_a/README.md); đổi định dạng thì cập nhật file đó.

## Lệnh (chạy trong `part_b/`)

```bash
python -m pip install -e ".[dev]"            # thêm ,index để dựng index
python -m pytest -q                          # toàn bộ test
python -m pytest tests/test_fusion.py -q     # một file
ruff check src tests scripts bench && ruff format --check src tests scripts bench
cd web && npm install && npm run build       # TS -> web/dist/app.js
uvicorn imgsearch.api.main:app --reload      # http://localhost:8000
docker compose up -d --build                 # minio + init + api
docker compose --profile bench run --rm bench && python bench/make_report.py
```

Máy dev là Windows (PowerShell / Git Bash); `make` có thể không có sẵn, nên dùng lệnh trực tiếp. Python local 3.14, image Docker dùng 3.12; code phải chạy được từ 3.11. Đường dẫn mặc định trong `config.py` (`../data`, `../models`) giả định cwd là `part_b/`.

## Kiến trúc Phần B

- `api/`: FastAPI. `/search` trả `SearchResponse` (kết quả theo **nhóm**, chuỗi bằng chứng `reasoning`, `messages`, `suggestions`) và header `Server-Timing` theo từng bước. `/healthz` trả 503 cho tới khi index nạp xong.
- `retrieval/`: hướng H2. `engine.py` điều phối: `query_parser` (quy tắc + alias person) -> `lexical` (BM25 numpy, `bm25.py`; **không dùng FTS5**: đo 100–150 ms/truy vấn trên 80k ảnh, BM25 numpy 4–6 ms) + `dense` (ONNX text encoder + FAISS flat) -> `fusion.rrf` -> `multi_intent` / `multi_hop`.
- `engine.search` theo loại truy vấn: single (có person: ảnh khớp mô tả trước, rồi mọi ảnh còn lại của person; không person: BM25 giới hạn theo nghề + đa dạng hóa `per_person_cap`), trùng tên -> nhóm `person` mỗi người, tên lạ -> chỉ thông báo, multi-intent -> nhóm mỗi vế + nhóm `co_occurrence` khi >= 2 người khác nhau có ảnh chung, multi-hop -> `multi_hop.find_edges` (tra cả chiều nghịch: child<->parent, spouse, sibling), không cạnh -> thông báo, không ảnh. Lọc an toàn và cắt `top_k` ở `_hydrate`. Hiện chỉ BM25; dense gộp bằng RRF trong `_retrieve`.
- Cạnh `relation` lưu một chiều: **obj là <predicate> của subj**.
- Dữ liệu cố định theo phiên chạy (ảnh theo nghề, ảnh bị lọc an toàn) tính sẵn lúc `load()`; không truy vấn SQL nặng trong đường `/search` (đo: SQL ảnh theo nghề ~100 ms/lần trên 80k).
- `data/`: `merge.py` (hợp nhất CSV, khóa `entity_id + sha256`), `entity_merge.py` (entity -> person, bắt buộc cho liên kết person và multi-hop), `schema.sql` (person, entity, image, image_entity, image_person, relation), `dataset.py` (định dạng JSONL trung gian giữa merge và build), `build_meta.py` (dựng `index/v{ver}/`: `meta.sqlite` + `lexical.npz` + `manifest.json`). `vec_row` (0..N-1) là chỉ số chung của ảnh cho BM25 và FAISS.
- `retrieval/store.py`: đọc `meta.sqlite` chỉ đọc, mỗi luồng một kết nối.
- Dữ liệu giả `tests/fixtures/mini/` (README liệt kê tình huống cài sẵn); `tests/conftest.py` dựng index mini một lần mỗi phiên, test không được đọc `../data`. Chạy thử server: `python scripts/build_index.py --input tests/fixtures/mini --version dev`.
- `kg/`: cạnh quan hệ từ Wikidata và tiêu đề bài báo (`description`).
- `scripts/`: job offline merge -> thumbs -> build_index -> upload_s3.
- Tham số engine nằm trong `config.py` (`Settings`, đọc từ env, mẫu ở `.env.example`): ngưỡng lọc an toàn, `candidate_pool`, `per_person_cap`, `thumbs_from_s3`. Không có file YAML cấu hình.
- `web/`: HTML + TS thuần, API phục vụ luôn (`WEB_DIR`), không framework.
- Index có phiên bản `index/v{INDEX_VERSION}/` trên S3/MinIO/R2 qua `S3_ENDPOINT_URL`.

Đã cài đặt thật: `text/normalize.py`, `retrieval/fusion.py`, `retrieval/query_parser.py` (+ `aliases.py`, `lexicon.py`; test theo mã kịch bản ở `tests/test_query_parser.py`), `retrieval/bm25.py` + `lexical.py` + `store.py`, `retrieval/engine.py` + `multi_intent.py` + `multi_hop.py` (test đầu-cuối theo mã kịch bản ở `tests/test_engine_search.py`), `data/dataset.py` + `build_meta.py` + `scripts/build_index.py`, `eval/metrics.py`, `api/`, `storage/s3.py`, `web/`. Từ vựng và nhãn hiển thị (quan hệ, nghề, dấu hiệu cùng xuất hiện…) sửa trong `lexicon.py`, không rải trong code; khi sửa parser/engine phải giữ test cũ qua và thêm test cho kịch bản mới. Chưa làm: `dense.py`, `data/merge.py`, `data/entity_merge.py`, `kg/*`, `eval/run_eval.py`, tải index từ S3, bắt buộc có index khi chạy Docker.

## Quy tắc bắt buộc

- **Không commit dữ liệu**: ảnh, CSV, embedding, index, model là dữ liệu người thật hoặc file lớn; `.gitignore` đã chặn. Không thêm ngoại lệ cho chúng.
- **Quan hệ multi-hop phải có bằng chứng**: mỗi cạnh `relation` có `source` + `evidence`. Không có cạnh thì trả "không tìm thấy bằng chứng quan hệ" và **không trả ảnh**. Không suy quan hệ từ tên, không để LLM trả lời sự kiện, không lấy đáp án từ ví dụ trong tài liệu.
- **LLM chỉ là parser fallback**: gọi khi quy tắc không phân tích được, timeout 1,5 s, có cache, hết giờ thì quay về quy tắc. Đường trả kết quả không được chờ LLM (mục tiêu p95 < 3 s, server < 150 ms khi không có LLM).
- **Không đưa torch vào runtime API**: suy luận bằng ONNX Runtime. Thư viện nặng chỉ thêm vào extra `index`.
- **Tiếng Việt**: luôn khớp cả dạng có dấu và không dấu (`normalize` / `fold`), ưu tiên khớp có dấu. `đ` phải thay riêng thành `d`.
- **Danh tính**: không dùng tên trong caption VLM làm bằng chứng danh tính; ưu tiên `person_id` từ dữ liệu đã duyệt. Cẩn thận trùng tên ("Sơn Tùng" nhà văn vs "Sơn Tùng M-TP").
- **Chỉ index ảnh approved**; ảnh trùng giữa entity hiện một lần (E7, Q3); lọc an toàn bật mặc định (Q4).
- **Đánh giá**: chỉ tinh chỉnh trên `split=dev` của test set; `test` chỉ chạy khi báo cáo.
- **Số liệu trong tài liệu/báo cáo** gắn nhãn **[Đã đo]** hoặc **[Ước tính]**; không đưa số chưa đo vào báo cáo như số thật.
- Không có chức năng tìm bằng ảnh khuôn mặt (SCENARIOS X1).

## Quy ước code

- Docstring và comment viết tiếng Việt, ngắn; tham chiếu mục PLAN hoặc mã kịch bản khi liên quan.
- Ruff, dòng tối đa 100 ký tự. Test trong `part_b/tests/`, đặt tên theo module.
- Thêm kịch bản mức **B** trong SCENARIOS thì thêm test case tương ứng (JSONL trong `part_b/testsets/`, định dạng ở `part_b/testsets/README.md`).
- Thứ tự ưu tiên khi thiếu thời gian: dữ liệu hợp nhất -> single-intent + latency -> multi-intent -> multi-hop -> Docker/benchmark -> tài liệu -> deploy/CI.
