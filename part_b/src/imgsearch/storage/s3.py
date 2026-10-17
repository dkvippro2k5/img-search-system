"""Client S3 dùng chung cho MinIO (local/compose), Cloudflare R2 và AWS S3.

Bucket private. Khóa: thumbs/{sha256[:2]}/{sha256}.webp, index/v{INDEX_VERSION}/...
"""

import boto3

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
