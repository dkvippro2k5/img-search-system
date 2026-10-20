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
python -m pip install -e ".[dev]"            # thêm ,index để dựng index; ,embed cho dense (torch CPU)
python -m pytest -q                          # toàn bộ test
python -m pytest tests/test_fusion.py -q     # một file
ruff check src tests scripts bench && ruff format --check src tests scripts bench
cd web && npm install && npm run build       # TS -> web/dist/app.js
uvicorn imgsearch.api.main:app --reload      # http://localhost:8000
docker compose up -d --build                 # minio + init (đẩy index) + api (tải index từ minio)
python bench/run_bench.py                    # k6 1/4/8/16 VU, nocache + cache -> ../reports/perf.md
python scripts/make_testset.py --index-version people   # test set -> testsets/queries.v1.jsonl
python -m imgsearch.eval.run_testset --index-version people --split dev   # -> ../reports/eval_dev.md
```

Máy dev là Windows (PowerShell / Git Bash); `make` có thể không có sẵn, nên dùng lệnh trực tiếp. Python local 3.14, image Docker dùng 3.12; code phải chạy được từ 3.11. Đường dẫn mặc định trong `config.py` (`../data`, `../models`) giả định cwd là `part_b/`.

## Kiến trúc Phần B

- `api/`: FastAPI. `/search` trả `SearchResponse` (kết quả theo **nhóm**, chuỗi bằng chứng `reasoning`, `messages`, `suggestions`) và header `Server-Timing` theo từng bước. `/healthz` trả 503 cho tới khi index nạp xong; `REQUIRE_INDEX=true` (Docker) thì thiếu index là 503 mãi. `INDEX_FROM_S3=true`: `engine.load` tải `index/v{ver}/` về `INDEX_DIR` (`storage/s3.download_index`, bỏ file cùng kích thước), lỗi tải thì dùng bản trên đĩa.
- **Docker**: image MinIO chính thức không còn kéo công khai được (20/10/2026) -> `cgr.dev/chainguard/minio` ghim digest, `user: 0`, không healthcheck (không có shell/curl). Service `init` dùng chính image `imgsearch:local` chạy `python -m imgsearch.storage.s3` (tạo bucket private, đẩy index). `/cache` trong image thuộc user `app` (volume mới lấy quyền này). `ONNX_THREADS=4` = số CPU của container. `QUERY_CACHE_SIZE=0` tắt cache vector truy vấn khi đo; API không có cache kết quả (`RESULT_CACHE_ENABLED` chưa được dùng).
- `retrieval/`: hướng H2. `engine.py` điều phối: `query_parser` (quy tắc + alias person) -> `lexical` (BM25 numpy, `bm25.py`; **không dùng FTS5**: đo 100–150 ms/truy vấn trên 80k ảnh, BM25 numpy 4–6 ms) + `dense` (vector ảnh SigLIP2, truy vấn mã hóa bằng ONNX; cosine bằng numpy trên toàn ma trận, không FAISS) -> `fusion.rrf` có trọng số `DENSE_WEIGHT` -> `multi_intent` / `multi_hop`.
- `engine.search` theo loại truy vấn: single (có person: ảnh khớp mô tả trước, rồi mọi ảnh còn lại của person; không person: BM25 giới hạn theo nghề + đa dạng hóa `per_person_cap`), trùng tên -> nhóm `person` mỗi người, tên lạ -> chỉ thông báo, multi-intent -> nhóm mỗi vế + nhóm `co_occurrence` khi >= 2 người khác nhau có ảnh chung, multi-hop -> `multi_hop.find_edges` (tra cả chiều nghịch: child<->parent, spouse, sibling), không cạnh -> thông báo, không ảnh. Lọc an toàn và cắt `top_k` ở `_hydrate`. Dense gộp với BM25 bằng RRF trong `_retrieve`, nhận phần mô tả **đã bỏ tên người** (model ảnh không biết mặt người nổi tiếng; danh tính đến từ lọc person); chỉ có tên thì không gọi dense.
- **Dense (bước 3)**: model `siglip2-b32` (open_clip `ViT-B-32-SigLIP2-256`/`webli`), chọn bằng `bench/dense_bakeoff.py` -> `../reports/dense_bakeoff.md`. `scripts/encode_images.py` (torch, offline) -> `../data/features/image_embeddings.{npy,ids.txt,json}`; `build_index.py --features` -> `dense.npz` theo `vec_row`; `scripts/export_text_encoder.py` -> `../models/siglip2-b32/` (`text.onnx` fp32 1,1 GB, ~30 ms/truy vấn; **int8 lệch quá nhiều, cosine 0,83–0,94, không dùng**). `DENSE_MIN_PROB` là **cổng** theo xác suất SigLIP sigmoid(scale·cos+bias) của ảnh khớp nhất: dưới ngưỡng thì bỏ cả kênh dense ("xyz abc"); không cắt từng ảnh (cắt làm RRF chỉ còn BM25, golden tụt); không áp dụng trong ảnh của một người. Tiền xử lý văn bản (`dense.canonicalize` + tokenizer.json + EOS + pad 64) phải khớp open_clip; export tự kiểm. Engine tắt dense (chạy chỉ BM25) khi thiếu `dense.npz`, thiếu model hoặc manifest khác `DENSE_MODEL`. Trọng số: không caption -> ảnh nặng hơn (2.0); có caption -> nên hạ (bake-off: dense làm giảm nhẹ khi đã có caption). Đã biết: model không hiểu "áo dài".
- **Dense văn bản (P3, bật mặc định, trọng số 0,5)**: qwen3-embedding-4b của mentor (`retrieval/mentor.py`, API kiểu OpenAI qua tunnel tạm, key trong `.env`). `scripts/encode_texts.py` mã hóa văn bản BM25 của ảnh -> `../data/features/text_embeddings.*` (f16, 2560) -> `text_dense.npz` (`build_index` hoặc `--write-index`). Server không nhận `dimensions`: cắt `TEXT_DENSE_DIM` lúc nạp. Engine gọi song song với BM25 + ảnh, `prefetch` mọi vế (API gọi `engine.prefetch` trước semaphore), timeout `MENTOR_TIMEOUT_SECONDS`, lỗi/429 -> bỏ kênh cho request đó. Thiếu file hoặc key thì tự tắt. Dev thật (18 truy vấn bối cảnh): nDCG@10 0,84 -> 0,85, P@10 nhãn 2 0,72 -> 0,75; trọng số 1–2 làm giảm (golden thì 2 tốt nhất). Giá phải trả: p95 engine 45 -> ~300 ms, QPS Docker 31 -> 20–25, RAM +0,4 GB. **Key mentor giới hạn 100 request/phút** (team 600): tải cao thì kênh bị 429 và chất lượng về mức không có kênh; `run_bench` đếm lỗi mentor. Reranker jina: bỏ (không caption thì kéo ảnh đúng về hình xuống; `reports/mentor_bakeoff.md`).
- Cạnh `relation` lưu một chiều: **obj là <predicate> của subj**.
- Dữ liệu cố định theo phiên chạy (ảnh theo nghề, ảnh bị lọc an toàn) tính sẵn lúc `load()`; không truy vấn SQL nặng trong đường `/search` (đo: SQL ảnh theo nghề ~100 ms/lần trên 80k).
- `data/`: `merge.py` (CSV `../data/raw/people-*.csv` -> JSONL; dòng thiếu sha256 lấy từ ảnh cùng URL hoặc nhật ký `fetch.py`, không có thì bỏ, không dùng hash URL; nghề trống lấy theo entity; cột `keywords` = cột `query` của crawler + slug URL khi không có tiêu đề, chỉ để tìm), `fetch.py` (tải ảnh thiếu sha256, nhật ký JSONL chạy tiếp được, lỗi tạm thử lại), `entity_merge.py` (entity -> person, bắt buộc cho liên kết person và multi-hop), `schema.sql` (person, entity, image, image_entity, image_person, relation), `dataset.py` (định dạng JSONL trung gian giữa merge và build), `build_meta.py` (dựng `index/v{ver}/`: `meta.sqlite` + `lexical.npz` + `manifest.json`). `vec_row` (0..N-1) là chỉ số chung của ảnh cho BM25 và FAISS.
- `retrieval/store.py`: đọc `meta.sqlite` chỉ đọc, mỗi luồng một kết nối.
- Dữ liệu giả `tests/fixtures/mini/` (README liệt kê tình huống cài sẵn); `tests/conftest.py` dựng index mini một lần mỗi phiên, test không được đọc `../data`. Chạy thử server: `python scripts/build_index.py --input tests/fixtures/mini --version dev`.
- `data/entity_merge.py` (v0, chỉ dùng tên): bỏ cụm mô tả đầu tên (`lexicon.ENTITY_PREFIXES`, `ACADEMIC_TITLES`), cụm theo (tên **giữ dấu**, nhóm nghề `CATEGORY_FAMILY`), tên 2 âm tiết gộp vào tên dài duy nhất cùng nhóm, entity nhắc nhiều người (tên >= 2 từ) gắn cho tất cả; `method` mỗi entity để rà lại. Biệt danh trong ngoặc chỉ nhận khi một trong hai tên >= 3 từ, và chỉ gộp với **tên chính** của cụm khác; hai biệt danh trùng nhau thì bỏ (tên chương trình/ban nhạc trong ngoặc từng nối chuỗi 128 entity thành một người). Không sửa được lỗi gán nhãn (cần khuôn mặt, Phần A).
- `kg/article_relations.py` (v0, ưu tiên độ chính xác): cạnh từ tiêu đề bài báo; tên phải khớp đúng (không khớp mờ), viết hoa, >= 2 từ, không cụt hai đầu, không phải địa danh; mỗi mệnh đề (`; : | /`) xét riêng, cắt đuôi tên báo (`PUBLISHER_MARKERS`), bỏ mọi ngoặc kép. Chỉ nhận dạng đồng vị ("X - con gái nhạc sĩ Y", "X, em gái Y" khi X đầu mệnh đề) hoặc từ tự xưng (`SELF_RELATIONS`, `TITLE_RELATIONS`: ông xã, cưới, làm vợ); "X <động từ> em trai Y" bị bỏ. Bẫy có test lấy từ tiêu đề thật. Wikidata chưa làm.
- Cụm viết hoa không phải người (địa danh, môn thể thao, chương trình) ở `lexicon.NON_PERSON_NAMES`: không thành biệt danh, không là "tên lạ".
- **Bộ golden** `../data/golden/` (README ở đó; không commit): hệ thống chỉ được đọc `metadata.json` + `images/`; `gold/` là đáp án, chỉ dùng khi chấm. Quy trình: `scripts/golden_to_dataset.py --mode auto` -> `build_index.py --input ../data/merged/golden-auto --version golden-auto` -> `python -m imgsearch.eval.run_eval --index-version golden-auto` -> `../reports/eval_golden_<ver>.md|json`. `--mode oracle` lấy person + quan hệ từ gold để chẩn đoán, không phải kết quả hệ thống. `--no-caption` bỏ caption để mô phỏng dữ liệu thật (dùng khi đo dense); embedding golden: `encode_images.py --golden ../data/golden --out ../data/features/golden`. Golden đang là tập phát triển (không tách dev/test): số đo lạc quan.
- **Test set trên dữ liệu thật** (`testsets/`, README ở đó): `eval/testgen.py` sinh tự động từ `meta.sqlite` (không gọi engine; gt "rule" gồm cả person tách mảnh cùng tên + cùng nhóm nghề; bỏ tổ chức/nhãn nhóm lẫn trong bảng person), `testsets/manual.v1.jsonl` viết tay (bối cảnh gt "pool", L3, multi-hop từ cạnh KG, phủ định). Bối cảnh: `scripts/make_pool.py` gộp top-20 của cấu hình `bm25`/`w2`/`w8` (`eval/pooling.py`) -> `scripts/label_server.py` (người dùng chấm 0/1/2, http://127.0.0.1:8765) -> `labels.v1.jsonl`. `eval/run_testset.py` chấm theo PLAN 5.2–5.3 (P@10, P@R, R@50 chuẩn hóa theo min(|rel|,50), nDCG@10, MRR, phủ ý định, người gốc/đích, bịa) và so cấu hình. Chỉ chạy `--split test` khi báo cáo.
- `scripts/`: job offline merge -> thumbs -> build_index -> upload_s3.
- Tham số engine nằm trong `config.py` (`Settings`, đọc từ env, mẫu ở `.env.example`): ngưỡng lọc an toàn, `candidate_pool`, `per_person_cap`, `thumbs_from_s3`. Không có file YAML cấu hình.
- `web/`: HTML + TS thuần, API phục vụ luôn (`WEB_DIR`), không framework.
- Index có phiên bản `index/v{INDEX_VERSION}/` trên S3/MinIO/R2 qua `S3_ENDPOINT_URL`.

- **Dữ liệu thật** (`../data/raw/people-*.csv`, không commit): `scripts/fetch_missing_images.py [--all]` -> `scripts/merge_data.py` (-> `../data/merged/people/` + `merge_report.json`, ~40 s) -> `scripts/encode_images.py` -> `build_index.py --input ../data/merged/people --version people` (tự đọc `../data/features`). Ảnh tải về ở `../data/raw/fetched/` (2,2 GB). `image_url` dạng `/api/entities/local-image/…` là ảnh trên máy của tool gán nhãn, không tải được.

Đã cài đặt thật: `text/normalize.py`, `retrieval/fusion.py`, `retrieval/query_parser.py` (+ `aliases.py`, `lexicon.py`; test theo mã kịch bản ở `tests/test_query_parser.py`), `retrieval/bm25.py` + `lexical.py` + `store.py`, `retrieval/engine.py` + `multi_intent.py` + `multi_hop.py` (test đầu-cuối theo mã kịch bản ở `tests/test_engine_search.py`), `data/dataset.py` + `build_meta.py` + `scripts/build_index.py`, `data/merge.py` + `data/fetch.py` + `scripts/merge_data.py` + `scripts/fetch_missing_images.py`, `retrieval/dense.py` + `scripts/encode_images.py` + `scripts/export_text_encoder.py` + `bench/dense_bakeoff.py`, `data/entity_merge.py`, `kg/article_relations.py`, `data/golden.py` + `scripts/golden_to_dataset.py`, `eval/metrics.py` + `eval/run_eval.py`, `eval/testset.py` + `testgen.py` + `pooling.py` + `run_testset.py` + `scripts/make_testset.py` + `make_pool.py` + `label_server.py`, `api/`, `storage/s3.py` (+ `scripts/upload_s3.py`), `bench/run_bench.py` + `make_report.py`, `retrieval/mentor.py` + `scripts/encode_texts.py` + `bench/mentor_bakeoff.py`, `web/`. Từ vựng và nhãn hiển thị (quan hệ, nghề, dấu hiệu cùng xuất hiện…) sửa trong `lexicon.py`, không rải trong code; khi sửa parser/engine phải giữ test cũ qua và thêm test cho kịch bản mới. Chưa làm: dense văn bản local (e5, không phụ thuộc mentor), caption VLM (CSV không có caption), `kg/wikidata.py`, multi-hop 2 bước (H5), đẩy thumbnail lên S3.

## Quy tắc bắt buộc

- **Không commit dữ liệu**: ảnh, CSV, embedding, index, model là dữ liệu người thật hoặc file lớn; `.gitignore` đã chặn. Không thêm ngoại lệ cho chúng.
- **Quan hệ multi-hop phải có bằng chứng**: mỗi cạnh `relation` có `source` + `evidence`. Không có cạnh thì trả "không tìm thấy bằng chứng quan hệ" và **không trả ảnh**. Không suy quan hệ từ tên, không để LLM trả lời sự kiện, không lấy đáp án từ ví dụ trong tài liệu.
- **LLM chỉ là parser fallback**: gọi khi quy tắc không phân tích được, timeout 1,5 s, có cache, hết giờ thì quay về quy tắc. Đường trả kết quả không được chờ LLM (mục tiêu p95 < 3 s, server < 150 ms khi không có LLM).
- **Không đưa torch vào runtime API**: suy luận bằng ONNX Runtime. Thư viện nặng chỉ thêm vào extra `index` (dựng index) hoặc `embed` (torch, open_clip: mã hóa ảnh, xuất ONNX).
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
