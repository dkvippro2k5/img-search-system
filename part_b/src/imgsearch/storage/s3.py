"""Client S3 dùng chung cho MinIO (local/compose), Cloudflare R2 và AWS S3.

Bucket private. Khóa: thumbs/{sha256[:2]}/{sha256}.webp, index/v{INDEX_VERSION}/...

Đẩy index (+ thumbnail nếu có, scripts/make_thumbs.py) lên (service `init` của docker-compose,
hoặc scripts/upload_s3.py):
    python -m imgsearch.storage.s3 --src ../data/index --thumbs ../data/thumbs --wait 60
Bản public (R2, README mục 4): thêm --env-file .env.r2 --version people
    --models ../models/siglip2-b32
"""

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from imgsearch.config import Settings


def client(settings: Settings, public: bool = False):
    """public=True: client để ký presigned URL cho trình duyệt. Trong compose, API gọi MinIO qua
    http://minio:9000 nhưng trình duyệt chỉ tới được S3_PUBLIC_ENDPOINT_URL (localhost:9000);
    chữ ký gắn với host nên phải ký bằng đúng địa chỉ trình duyệt dùng."""
    endpoint = settings.s3_endpoint_url
    if public and settings.s3_public_endpoint_url:
        endpoint = settings.s3_public_endpoint_url
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
        # SigV4 cho mọi endpoint: với us-east-1 boto3 mặc định ký presign kiểu SigV2, R2 không nhận.
        # Pool 32 >= số luồng upload_thumbs (mặc định 10: "Connection pool is full", chậm hẳn).
        config=Config(signature_version="s3v4", max_pool_connections=32),
    )


def thumb_key(sha256: str) -> str:
    return f"thumbs/{sha256[:2]}/{sha256}.webp"


def index_prefix(version: str) -> str:
    return f"index/v{version}/"


def presigned_thumb_url(settings: Settings, sha256: str, s3=None) -> str:
    """Ký URL tạm (không gọi mạng). Truyền `s3` để dùng lại client khi ký nhiều ảnh."""
    return (s3 or client(settings)).generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": thumb_key(sha256)},
        ExpiresIn=settings.presign_ttl_seconds,
    )


def model_prefix(name: str) -> str:
    return f"models/{name}/"


def download_index(settings: Settings, s3=None) -> int:
    """Tải index/v{INDEX_VERSION}/ về settings.index_path; bỏ qua file đã có cùng kích thước.

    Trả số file đã tải. Không có object nào dưới prefix -> FileNotFoundError.
    """
    s3 = s3 or client(settings)
    return download_prefix(settings, index_prefix(settings.index_version), settings.index_path, s3)


def download_model(settings: Settings, s3=None) -> int:
    """Tải models/{DENSE_MODEL}/ (text.onnx, tokenizer.json, encoder.json) về MODELS_DIR, cho máy
    chủ không mount sẵn thư mục model (bản public)."""
    s3 = s3 or client(settings)
    name = settings.dense_model
    return download_prefix(settings, model_prefix(name), settings.models_dir / name, s3)


def download_prefix(settings: Settings, prefix: str, dest: Path, s3) -> int:
    """Tải các file phẳng dưới prefix về dest; bỏ qua file đã có cùng kích thước."""
    objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=prefix
    ):
        objects += page.get("Contents", [])
    if not objects:
        raise FileNotFoundError(f"s3://{settings.s3_bucket}/{prefix} rỗng")
    n = 0
    for obj in objects:
        name = obj["Key"][len(prefix) :]
        if not name or "/" in name:  # chỉ file phẳng trong thư mục phiên bản
            continue
        target = dest / name
        if target.exists() and target.stat().st_size == obj["Size"]:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        s3.download_file(settings.s3_bucket, obj["Key"], str(tmp))
        tmp.replace(target)
        n += 1
    return n


def ensure_bucket(settings: Settings, s3=None, wait: float = 0) -> None:
    """Tạo bucket nếu chưa có (private: không gắn policy công khai). Chờ tối đa `wait` giây cho
    tới khi S3 nhận kết nối (MinIO vừa khởi động trong compose)."""
    s3 = s3 or client(settings)
    deadline = time.monotonic() + wait
    while True:
        try:
            s3.head_bucket(Bucket=settings.s3_bucket)
            return
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchBucket", "NotFound"):
                s3.create_bucket(Bucket=settings.s3_bucket)
                return
            raise
        except BotoCoreError:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)


def upload_index(settings: Settings, src: Path, s3=None) -> int:
    """Đẩy {src}/v{INDEX_VERSION}/* lên index/v{ver}/; bỏ qua object đã có cùng kích thước."""
    s3 = s3 or client(settings)
    folder = Path(src) / f"v{settings.index_version}"
    if not (folder / "meta.sqlite").exists():
        raise FileNotFoundError(f"{folder} không có meta.sqlite")
    prefix = index_prefix(settings.index_version)
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=prefix
    ):
        have.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    n = 0
    for f in sorted(p for p in folder.iterdir() if p.is_file()):
        key = prefix + f.name
        if have.get(key) == f.stat().st_size:
            continue
        s3.upload_file(str(f), settings.s3_bucket, key)
        n += 1
    return n


def upload_thumbs(settings: Settings, src: Path, s3=None, workers: int = 16) -> int:
    """Đẩy {src}/{id[:2]}/{id}.webp lên thumbs/...; bỏ qua object đã có cùng kích thước."""
    s3 = s3 or client(settings)
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix="thumbs/"
    ):
        have.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    todo = [
        (f, thumb_key(f.stem))
        for f in sorted(Path(src).glob("*/*.webp"))
        if have.get(thumb_key(f.stem)) != f.stat().st_size
    ]
    extra = {"ContentType": "image/webp", "CacheControl": "public, max-age=86400"}

    def one(item: tuple[Path, str]) -> None:
        s3.upload_file(str(item[0]), settings.s3_bucket, item[1], ExtraArgs=extra)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    return len(todo)


def upload_model(settings: Settings, src: Path, s3=None) -> int:
    """Đẩy thư mục model ONNX (text.onnx, tokenizer.json, encoder.json) lên models/{tên thư mục}/
    cho MODELS_FROM_S3; bỏ qua object đã có cùng kích thước."""
    s3 = s3 or client(settings)
    prefix = model_prefix(Path(src).name)
    have = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=settings.s3_bucket, Prefix=prefix
    ):
        have.update({o["Key"]: o["Size"] for o in page.get("Contents", [])})
    n = 0
    for f in sorted(p for p in Path(src).iterdir() if p.is_file()):
        if have.get(prefix + f.name) != f.stat().st_size:
            s3.upload_file(str(f), settings.s3_bucket, prefix + f.name)
            n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Đẩy index/v{INDEX_VERSION}/ lên S3/MinIO/R2")
    ap.add_argument("--src", type=Path, default=Path("../data/index"))
    ap.add_argument("--wait", type=float, default=0, help="giây chờ S3 sẵn sàng")
    ap.add_argument("--thumbs", type=Path, default=None, help="thư mục thumbnail (make_thumbs.py)")
    ap.add_argument(
        "--models", type=Path, default=None, help="thư mục model, vd ../models/siglip2-b32"
    )
    ap.add_argument("--env-file", default=".env", help="vd .env.r2 để đẩy lên R2 thay vì MinIO")
    ap.add_argument("--version", default=None, help="INDEX_VERSION (mặc định: theo env)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    override = {"index_version": args.version} if args.version else {}
    settings = Settings(_env_file=args.env_file, **override)
    ensure_bucket(settings, wait=args.wait)
    t = time.perf_counter()
    n = upload_index(settings, args.src)
    logging.info(
        "Đã đẩy %d file index v%s lên s3://%s (%.1f s)",
        n,
        settings.index_version,
        settings.s3_bucket,
        time.perf_counter() - t,
    )
    if args.thumbs is not None and args.thumbs.is_dir():
        t = time.perf_counter()
        n = upload_thumbs(settings, args.thumbs)
        logging.info("Đã đẩy %d thumbnail (%.1f s)", n, time.perf_counter() - t)
    if args.models is not None:
        n = upload_model(settings, args.models)
        logging.info("Đã đẩy %d file model %s", n, args.models.name)


if __name__ == "__main__":
    main()
