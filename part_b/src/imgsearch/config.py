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
    index_from_s3: bool = False

    s3_endpoint_url: str | None = None
    s3_bucket: str = "imgsearch"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_region: str = "us-east-1"
    presign_ttl_seconds: int = 600
    # False: hiển thị bằng image_url gốc (hotlink, PLAN 7.2 [Cần xác nhận]);
    # True: presigned URL tới thumbs/{sha256[:2]}/{sha256}.webp trên S3.
    thumbs_from_s3: bool = False

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

    llm_parser_enabled: bool = False
    llm_timeout_seconds: float = 1.5

    safe_search_default: bool = True
    # Lọc an toàn (Q4). Ngưỡng và tên cờ là giả định, chờ chính sách của mentor.
    nsfw_block_score: float = 0.8
    unsafe_flags: list[str] = ["nsfw", "sexual"]
    max_query_chars: int = 300
    result_cache_enabled: bool = True

    # Truy xuất
    candidate_pool: int = 200  # số ứng viên BM25 mỗi lượt
    per_person_cap: int = 3  # truy vấn chủ đề: tối đa ảnh mỗi người ở đầu danh sách (S3, L4)

    @property
    def index_path(self) -> Path:
        return self.index_dir / f"v{self.index_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
