"""Đẩy index/v{INDEX_VERSION}/ (+ thumbnail, model) lên S3/R2/MinIO (khóa trong .env).

python scripts/upload_s3.py --src ../data/index --thumbs ../data/thumbs
# bản public trên R2 (README mục 4):
python scripts/upload_s3.py --env-file .env.r2 --version people --src ../data/index
    --thumbs ../data/thumbs --models ../models/siglip2-b32
"""

from imgsearch.storage.s3 import main

if __name__ == "__main__":
    main()
