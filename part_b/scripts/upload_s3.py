"""Đẩy index/v{INDEX_VERSION}/ lên S3/R2/MinIO (S3_ENDPOINT_URL, khóa trong .env).

python scripts/upload_s3.py --src ../data/index

Thumbnail (thumbs/) chưa đẩy: web đang hotlink image_url gốc (THUMBS_FROM_S3=false).
"""

from imgsearch.storage.s3 import main

if __name__ == "__main__":
    main()
