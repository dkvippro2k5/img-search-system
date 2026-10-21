# Phần B: truy xuất ảnh

Tìm ảnh người nổi tiếng Việt Nam bằng ngôn ngữ tự nhiên: single-intent, multi-intent, multi-hop; latency < 3 s trên ≥ 80.000 ảnh. Thiết kế: [../capstone/PLAN.md](../capstone/PLAN.md) mục 4–7. Kịch bản: [../capstone/SCENARIOS.md](../capstone/SCENARIOS.md). Kiến trúc, quyết định và số đo: [../docs/architecture.md](../docs/architecture.md).

```
src/imgsearch/
  api/        FastAPI: /search, /healthz, /metrics, phục vụ web tĩnh
  retrieval/  query_parser (+ aliases, lexicon), lexical (BM25 numpy), dense (SigLIP2, ONNX),
              mentor (dense văn bản qwen3 qua API), fusion (RRF), multi_intent, multi_hop, engine
  kg/         article_relations: cạnh quan hệ từ tiêu đề bài báo, mỗi cạnh có bằng chứng
  data/       merge CSV, fetch ảnh, entity_merge (entity -> person), build_meta, schema.sql
  eval/       chỉ số, sinh test set, pooling, chạy test set và bộ golden
  storage/    S3 / MinIO / R2
  text/       chuẩn hóa tiếng Việt (có dấu / không dấu)
scripts/      fetch -> merge -> encode -> build_index -> upload_s3; test set, pool, label server
testsets/     test set JSONL, nhãn chấm tay, truy vấn benchmark (README ở đó)
bench/        k6 + run_bench.py -> ../reports/perf.md; bake-off model
web/          giao diện tĩnh TypeScript
```

## Cần gì

| Thứ | Ghi chú |
|---|---|
| Python 3.11+ | đã chạy trên 3.12 (Docker) và 3.14 (Windows) |
| Node 18+ | chỉ để build `web/` (TypeScript) |
| Docker Desktop | chạy compose (MinIO + API) và benchmark k6 |
| Dữ liệu `../data/raw/people-*.csv` | **không có trong repo** (ảnh người thật). Không có thì dùng dữ liệu giả `tests/fixtures/mini/` |
| Key API mentor (`MENTOR_API_KEY`, `TEXT_EMBED_URL`) | **không có trong repo**. Không có thì kênh dense văn bản tự tắt, hệ thống chạy BM25 + dense ảnh |
| ~35 GB đĩa trống | ảnh tải về 28 GB (`--all`), model ONNX 1,1 GB, embedding + index `people` ~1,3 GB, image Docker |

Lệnh bên dưới chạy trong `part_b/` (đường dẫn mặc định `../data`, `../models` tính từ đây). Trên Windows dùng PowerShell hoặc Git Bash; `make` không bắt buộc, mọi target trong [Makefile](Makefile) đều là một lệnh viết ở đây.

## 1. Chạy nhanh với dữ liệu giả (không cần dữ liệu thật)

```bash
python -m venv .venv && . .venv/Scripts/activate      # PowerShell: .venv\Scripts\Activate.ps1; Linux/macOS: . .venv/bin/activate
cp .env.example .env                                  # INDEX_VERSION=dev
python -m pip install -e ".[dev]"
python -m pytest -q                                   # không đọc ../data; test tải ảnh cần extra index (Pillow), thiếu thì tự bỏ qua
python scripts/build_index.py --input tests/fixtures/mini --version dev
cd web && npm install && npm run build && cd ..
uvicorn imgsearch.api.main:app --reload               # http://localhost:8000
```

Không có model ONNX nên dense ảnh tự tắt, engine chạy BM25 (`/healthz` trả `"dense": null`). Đã chạy thử từ bản clone sạch trên Python 3.12, cả bằng `uvicorn` lẫn `docker compose` với `INDEX_VERSION=dev` [Đã đo 21/10/2026]. Các truy vấn mẫu nằm ở `tests/fixtures/mini/README.md`.

## 2. Dựng lại toàn bộ từ dữ liệu thật

Đặt `people-*.csv` vào `../data/raw/`. Trong `.env`, đặt `INDEX_VERSION=people` và điền key/URL mentor nếu có. Thời gian ghi bên dưới là **[Đã đo]** trên Core Ultra 7 155H (CPU, không GPU).

```bash
python -m pip install -e ".[index,dev]"
# torch bản CPU cho bước mã hóa ảnh (chỉ offline, không vào image API)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[embed]"

# 1. Tải ảnh (chạy tiếp được nếu ngắt giữa chừng) -> ../data/raw/fetched/ (28 GB)
python scripts/fetch_missing_images.py --all
# 2. Hợp nhất CSV -> ../data/merged/people/ + merge_report.json (~40 s)
python scripts/merge_data.py
# 3. Mã hóa ảnh SigLIP2 -> ../data/features/image_embeddings.* (~1,7 h)
python -X utf8 scripts/encode_images.py
# 4. Xuất text encoder sang ONNX -> ../models/siglip2-b32/ (text.onnx 1,1 GB, tự kiểm tra khớp torch)
python -X utf8 scripts/export_text_encoder.py
# 5. Dựng index -> ../data/index/vpeople/ (meta.sqlite, lexical.npz, dense.npz, manifest.json)
python scripts/build_index.py --input ../data/merged/people --version people
# 6. (cần key mentor) mã hóa văn bản ảnh bằng qwen3 rồi gắn text_dense.npz vào index (~23 phút)
python -X utf8 scripts/encode_texts.py --index-version people --write-index
# 7. Thumbnail WebP 320 px -> ../data/thumbs/ (0,72 GB) + ../data/index/vpeople/thumbs.txt
python -X utf8 scripts/make_thumbs.py --index-version people
```

Bước 3 và 6 ghi theo shard, chạy lại thì bỏ qua phần đã xong. Bỏ bước 6 thì hệ thống chạy BM25 + dense ảnh. Khi đã có `../data/features/text_embeddings.*` từ một lần chạy trước, bước 5 tự gắn `text_dense.npz` vào index.

Chạy thử bằng `uvicorn imgsearch.api.main:app`. `/healthz` trả 503 cho tới khi nạp xong.

**Demo trên máy có ảnh đã tải** (bước 1): đặt `LOCAL_IMAGES=true` trong `.env`. Web khi đó lấy ảnh từ `../data/raw/fetched/` qua `/img/{image_id}` thay vì hotlink báo gốc: đo trên 12 truy vấn mẫu, 216/219 ảnh có file trên máy, ảnh còn lại chuyển hướng tới URL gốc. Hotlink thì 9–19% ô ảnh trống do CDN chặn tải dồn [Đã đo]. Docker không mount ảnh nên để `false`.

## 3. Docker (MinIO + API)

```bash
docker compose up -d --build        # minio, init (đẩy index + thumbnail lên bucket), api
curl http://localhost:8000/healthz  # 200 khi đã tải index từ MinIO và nạp xong
```

- `init` đọc `../data/index` và `../data/thumbs`, `api` mount `../models` (chỉ đọc), nên bước 2 phải chạy trước. Với dữ liệu giả, dùng `INDEX_VERSION=dev`.
- Ảnh có thumbnail (`thumbs.txt`) hiển thị bằng presigned URL tới MinIO, ký với `S3_PUBLIC_ENDPOINT_URL=http://localhost:9000` vì trình duyệt không tới được `minio:9000`. Ảnh không có thumbnail, hoặc khi chưa chạy bước 7, thì hotlink `image_url`.
- `api` giới hạn 4 CPU / 4 GB và đặt `SEARCH_CONCURRENCY=2`. Lý do và số đo ở [../docs/architecture.md](../docs/architecture.md).
- MinIO dùng `cgr.dev/chainguard/minio` ghim digest, vì image MinIO chính thức không còn kéo công khai được.

## 4. Bản public: Cloudflare R2 + Cloudflare Tunnel

API chạy trên máy này (cùng cấu hình 4 CPU / 4 GB đã benchmark). Index, model và thumbnail nằm trên bucket R2 private; trình duyệt tải thumbnail thẳng từ R2 bằng presigned URL hạn 6 giờ (dùng lại trong 3 giờ để trình duyệt cache được). Người ngoài vào qua tunnel `*.trycloudflare.com`, phải nhập mật khẩu dùng chung.

```bash
# 1. part_b/.env.r2: S3_ENDPOINT_URL (https://<account>.r2.cloudflarestorage.com), S3_BUCKET, key R2
#    (token "Object Read & Write" chỉ cho bucket đó). part_b/.env: DEMO_PASSWORD=...
# 2. Đẩy index + thumbnail + model lên R2 (bỏ qua file đã có)
python -X utf8 -m imgsearch.storage.s3 --env-file .env.r2 --version people --src ../data/index --thumbs ../data/thumbs --models ../models/siglip2-b32
# 3. Chạy api (tải mọi thứ từ R2) + cloudflared; lấy URL public trong log.
#    Lần đầu (volume trống) api tải ~1,5–2 GB rồi mới healthy: lệnh up chờ vài phút là bình thường.
docker compose -f docker-compose.public.yml -p imgsearch-public up -d --build
docker compose -f docker-compose.public.yml -p imgsearch-public logs cloudflared | grep trycloudflare
# 4. Tắt khi xong buổi demo
docker compose -f docker-compose.public.yml -p imgsearch-public down
```

An toàn khi public (SCENARIOS X2, X4):
- `DEMO_PASSWORD` bắt buộc (thiếu thì compose không chạy); hỏi mật khẩu mọi đường dẫn trừ `/healthz`.
- `RATE_LIMIT_PER_MINUTE=30` truy vấn mỗi IP (IP thật lấy từ header `CF-Connecting-IP`).
- `AUTH_FAIL_PER_MINUTE=10`: sai mật khẩu quá số lần này trong một phút thì IP đó nhận 429 trong một phút (chống dò mật khẩu).
- `ALLOW_UNSAFE_SEARCH=false`: client không tắt được lọc an toàn.
- Truy vấn tìm ảnh nhạy cảm của người thật bị từ chối và ghi log (`retrieval/guard.py`, từ vựng `lexicon.SENSITIVE`). Đây chỉ là lớp phụ ở mức truy vấn; lọc ảnh nhạy cảm thật sự cần điểm kiểm duyệt của Phần A (future work, [../part_a/README.md](../part_a/README.md)).
- URL tunnel đổi mỗi lần chạy và chỉ sống khi máy này bật.

## 5. Đánh giá và benchmark

```bash
# test set (đã có trong testsets/; chỉ sinh lại khi index đổi)
python scripts/make_testset.py --index-version people
# chấm trên dev (dùng khi tinh chỉnh) -> ../reports/eval_dev.md
python -X utf8 -m imgsearch.eval.run_testset --index-version people --split dev
# split test: chỉ chạy khi báo cáo -> ../reports/eval_test.md
python -X utf8 -m imgsearch.eval.run_testset --index-version people --split test
# benchmark HTTP: dựng lại api, k6 1/4/8/16 VU, nocache + cache -> ../reports/perf.md
python bench/run_bench.py
```

Chấm tay truy vấn bối cảnh: chạy `scripts/make_pool.py` rồi `scripts/label_server.py` (http://127.0.0.1:8765), xem [testsets/README.md](testsets/README.md). Bộ golden (`../data/golden/`, không có trong repo) dùng `scripts/golden_to_dataset.py` và `imgsearch.eval.run_eval`, xem docstring của hai script đó.

## Cấu hình

Mọi tham số nằm trong [src/imgsearch/config.py](src/imgsearch/config.py) và được đọc từ biến môi trường. Mẫu ở [.env.example](.env.example). Các tham số chính:

- `DENSE_WEIGHT`, `TEXT_DENSE_WEIGHT`: trọng số RRF.
- `DENSE_MIN_PROB`: cổng của kênh dense.
- `SEARCH_CONCURRENCY`: số truy vấn tính toán đồng thời.
- `QUERY_CACHE_SIZE`: cache vector truy vấn.
- `SAFE_SEARCH_DEFAULT`, `ALLOW_UNSAFE_SEARCH`, `NSFW_BLOCK_SCORE`: lọc an toàn.
- `DEMO_PASSWORD`, `RATE_LIMIT_PER_MINUTE`, `AUTH_FAIL_PER_MINUTE`: bản public.
- `THUMBS_FROM_S3`, `S3_PUBLIC_ENDPOINT_URL`, `PRESIGN_TTL_SECONDS`, `LOCAL_IMAGES`: nguồn ảnh hiển thị.
- `INDEX_FROM_S3`, `MODELS_FROM_S3`: tải index / model từ S3 lúc khởi động.
- `CANDIDATE_POOL`, `PER_PERSON_CAP`.

## Chưa làm

- Quan hệ từ Wikidata: `kg/wikidata.py` mới có khung.
- Multi-hop 2 bước.
- Dense văn bản chạy local, không phụ thuộc mentor.
- Caption VLM.
