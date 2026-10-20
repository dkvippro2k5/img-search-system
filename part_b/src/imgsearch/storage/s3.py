"""Client S3 dùng chung cho MinIO (local/compose), Cloudflare R2 và AWS S3.

Bucket private. Khóa: thumbs/{sha256[:2]}/{sha256}.webp, index/v{INDEX_VERSION}/...

Đẩy index lên (service `init` của docker-compose, hoặc scripts/upload_s3.py):
    python -m imgsearch.storage.s3 --src ../data/index --wait 60
"""

import argparse
import logging
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from imgsearch.config import Settings


def client(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
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


def download_index(settings: Settings, s3=None) -> int:
    """Tải index/v{INDEX_VERSION}/ về settings.index_path; bỏ qua file đã có cùng kích thước.

    Trả số file đã tải. Không có object nào dưới prefix -> FileNotFoundError.
    """
    s3 = s3 or client(settings)
    prefix = index_prefix(settings.index_version)
    dest = settings.index_path
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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Đẩy index/v{INDEX_VERSION}/ lên S3/MinIO/R2")
    ap.add_argument("--src", type=Path, default=Path("../data/index"))
    ap.add_argument("--wait", type=float, default=0, help="giây chờ S3 sẵn sàng")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
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


if __name__ == "__main__":
    main()
