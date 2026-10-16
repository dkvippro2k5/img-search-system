"""Hợp nhất các file approved-results*.csv của cả team -> data/merged/.

Luôn nối bằng (entity_id, sha256). In số dòng trước/sau mỗi bước để kiểm tra lệch cột.
Đầu ra: images.parquet, image_entity.parquet, entities.parquet + số đếm cuối (>= 80.000).
"""


def merge(raw_dir: str, out_dir: str) -> None:
    raise NotImplementedError
