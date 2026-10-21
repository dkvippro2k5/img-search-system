"""Cấu hình đọc từ biến môi trường / file .env (xem .env.example)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("../data")
    index_dir: Path = Path("../data/index")
    models_dir: Path = Path("../models")
    web_dir: Path = Path("web")  # thư mục chứa index.html, style.css, dist/app.js

    index_version: str = "dev"
    # true: tải index/v{INDEX_VERSION}/ từ S3/MinIO về INDEX_DIR lúc khởi động (storage/s3.py)
    index_from_s3: bool = False
    # true: tải models/{DENSE_MODEL}/ từ S3 về MODELS_DIR lúc khởi động (bản public không mount)
    models_from_s3: bool = False
    # true (Docker): không có index thì /healthz trả 503 mãi, không chạy với index rỗng
    require_index: bool = False

    s3_endpoint_url: str | None = None
    s3_bucket: str = "imgsearch"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"
    presign_ttl_seconds: int = 600
    # Địa chỉ S3 mà trình duyệt tới được, dùng để ký presigned URL thumbnail. Bỏ trống: như
    # S3_ENDPOINT_URL (R2/AWS). Compose: http://localhost:9000 (API gọi MinIO qua minio:9000).
    s3_public_endpoint_url: str | None = None
    # False: hiển thị bằng image_url gốc (hotlink, PLAN 7.2 [Cần xác nhận]);
    # True: presigned URL tới thumbs/{sha256[:2]}/{sha256}.webp trên S3, chỉ cho ảnh có trong
    # index/v{ver}/thumbs.txt (scripts/make_thumbs.py); ảnh còn lại vẫn hotlink.
    thumbs_from_s3: bool = False
    # Demo cục bộ: ảnh lấy từ ảnh đã tải trên máy (data/raw/fetched, fetch_missing_images.py)
    # qua /img/{image_id}, không có thì chuyển hướng tới image_url. Hotlink hỏng 9–19% ảnh do
    # CDN chặn tải dồn (docs/architecture.md). Docker không mount ảnh nên để tắt.
    local_images: bool = False

    # Dense (retrieval/dense.py): bật khi index có dense.npz và có model ở models_dir/dense_model.
    dense_enabled: bool = True
    dense_model: str = "siglip2-b32"
    dense_onnx_file: str = "text.onnx"  # int8 lệch quá nhiều (cosine 0,83–0,94), không dùng
    # Trọng số RRF của kênh ảnh so với BM25 (= 1). bench/dense_bakeoff.py: khi KHÔNG có caption,
    # ảnh nặng hơn tốt hơn (văn bản ít, BM25 hay trượt); khi có caption nên hạ về ~1.
    dense_weight: float = 2.0
    # Bỏ kết quả dense có xác suất khớp SigLIP sigmoid(scale·cos + bias) thấp hơn ngưỡng này
    # (câu vô nghĩa "xyz abc": p < 0,01; truy vấn thật: top 10 p > 0,3). Không có bias: bỏ qua.
    dense_min_prob: float = 0.1
    # Số luồng ONNX Runtime cho text encoder; 0 = mặc định (mọi lõi của máy). Trong container
    # giới hạn CPU nên đặt bằng số CPU được cấp, tránh tranh luồng khi tải đồng thời.
    onnx_threads: int = 0
    # Cache LRU vector truy vấn dense; 0 = tắt (khi đo latency, PLAN 6).
    query_cache_size: int = 1024

    # Dịch vụ của mentor (retrieval/mentor.py): đường hầm tạm thời, mỗi lời gọi có timeout,
    # lỗi thì bỏ kênh. Key chỉ đặt trong .env.
    mentor_api_key: str | None = None
    text_embed_url: str | None = None
    text_embed_model: str = "qwen3-embedding-4b"
    rerank_url: str | None = None
    rerank_model: str = "jina-reranker-v2"
    mentor_timeout_seconds: float = 1.0
    # Kênh dense văn bản (qwen3 trên văn bản BM25 của ảnh, scripts/encode_texts.py ->
    # text_dense.npz). Bật khi index có file và có key mentor; thiếu thì tự tắt. Trọng số 0,5:
    # dev thật +0,02 nDCG (1 và 2 làm giảm; golden thì 2 tốt nhất), đổi lại phụ thuộc mạng
    # mentor + latency (p95 engine 45 -> 278 ms). DIM: cắt Matryoshka lúc nạp (server không nhận
    # `dimensions`); 1024 float32 ≈ 368 MB / 90k ảnh.
    text_dense_enabled: bool = True
    text_dense_weight: float = 0.5
    text_dense_dim: int = 1024

    llm_parser_enabled: bool = False
    llm_timeout_seconds: float = 1.5

    safe_search_default: bool = True
    # false: client không tắt được lọc an toàn (safe_search=false bị bỏ qua); bật khi dev/đo.
    allow_unsafe_search: bool = False
    # Lọc an toàn (Q4). Ngưỡng và tên cờ là giả định, chờ chính sách của mentor.
    nsfw_block_score: float = 0.8
    unsafe_flags: list[str] = ["nsfw", "sexual"]
    max_query_chars: int = 300
    # Bản public (api/main.py): mật khẩu dùng chung (HTTP Basic, mọi đường dẫn trừ /healthz;
    # trống = không hỏi) và số /search tối đa mỗi IP mỗi phút (0 = không giới hạn).
    demo_password: str = ""
    rate_limit_per_minute: int = 0
    # Số lần gửi sai mật khẩu mỗi IP mỗi phút; quá thì 429, không thử mật khẩu (chống dò).
    auth_fail_per_minute: int = 10
    # Số truy vấn được tính toán đồng thời (api/main.py); 0 = không giới hạn.
    search_concurrency: int = 0

    # Truy xuất
    candidate_pool: int = 200  # số ứng viên BM25 mỗi lượt
    per_person_cap: int = 3  # truy vấn chủ đề: tối đa ảnh mỗi người ở đầu danh sách (S3, L4)

    @property
    def index_path(self) -> Path:
        return self.index_dir / f"v{self.index_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
