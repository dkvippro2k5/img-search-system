"""Thumbnail WebP cho hiển thị (PLAN 7.4): {out}/{image_id[:2]}/{image_id}.webp.

Nguồn là ảnh đã tải (data/fetch.py), tra theo image_url như api/local_images.py. Đo 21/10/2026
trên 300 ảnh: 320 px, q75 -> 7,8 KB/ảnh, ~0,72 GB / 90k ảnh, 28 ảnh/s mỗi luồng.
Cần Pillow (extra `index`), nhập lười để runtime API không phụ thuộc.
"""

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger(__name__)

THUMB_SIZE = 320
THUMB_QUALITY = 75
LIST_FILE = "thumbs.txt"  # trong index/v{ver}/: image_id có thumbnail, mỗi dòng một id


def thumb_path(out_dir: Path, image_id: str) -> Path:
    return Path(out_dir) / image_id[:2] / f"{image_id}.webp"


def make_thumb(src: Path, dest: Path, size: int = THUMB_SIZE, quality: int = THUMB_QUALITY) -> bool:
    """Ghi dest (qua file .part rồi đổi tên). Ảnh hỏng / không mở được -> False."""
    from PIL import Image

    try:
        with Image.open(src) as im:
            im.draft("RGB", (2 * size, 2 * size))  # JPEG: giải mã ở độ phân giải thấp, nhanh hơn
            im = im.convert("RGB")
            im.thumbnail((size, size))
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            im.save(tmp, "WEBP", quality=quality, method=4)
        tmp.replace(dest)
        return True
    except Exception as e:  # Pillow ném nhiều loại lỗi khác nhau cho ảnh hỏng
        log.warning("Không tạo được thumbnail %s: %s", src.name, e)
        return False


def build_thumbs(files: dict[str, Path], out_dir: Path, workers: int = 8) -> list[str]:
    """Tạo thumbnail cho {image_id: file nguồn}, bỏ qua cái đã có. Trả image_id có thumbnail."""

    def one(item: tuple[str, Path]) -> str | None:
        image_id, src = item
        dest = thumb_path(out_dir, image_id)
        if dest.exists() or make_thumb(src, dest):
            return image_id
        return None

    with ThreadPoolExecutor(workers) as ex:
        return sorted(i for i in ex.map(one, files.items()) if i)


def write_list(path: Path, image_ids: Iterable[str]) -> None:
    path.write_text("".join(f"{i}\n" for i in image_ids), encoding="utf-8")


def read_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
