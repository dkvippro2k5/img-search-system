"""Tải ảnh thiếu sha256 trong CSV (chưa có file tải về) để tính sha256, kích thước. Chạy offline.

Nhật ký JSONL (mỗi URL một dòng, ghi nối tiếp) là đầu vào của merge.resolve_sha, nên chạy lại
được sau khi bị ngắt: URL đã có kết quả cuối thì bỏ qua. Lỗi tạm (timeout, 5xx, 429) được thử
lại ở lần chạy sau (kể cả 403: CDN chặn tạm khi tải dồn dập); lỗi cố định (404, không phải
ảnh, quá lớn) thì không.

Ảnh lưu theo nội dung: {images_dir}/{sha256[:2]}/{sha256}{ext}. Không commit (data/ bị ignore).
"""

import hashlib
import io
import json
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

MAX_BYTES = 25 * 1024 * 1024
TIMEOUT_S = 20
USER_AGENT = "Mozilla/5.0 (compatible; imgsearch-capstone/0.1; offline dataset fetch)"
_EXT = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}
# 403 cũng thử lại: CDN ảnh (cloudfront) trả 403 khi tải dồn dập, thử lại sau thì 200.
_RETRYABLE = {"timeout", "error", "http_403", "http_429"} | {f"http_{c}" for c in range(500, 600)}
_URL_SAFE = ":/?#[]@!$&'()*+,;=%~"


def ascii_url(url: str) -> str:
    """Mã hóa phần trăm ký tự ngoài ASCII ("…/MC_Khánh_Long.jpg"); header HTTP chỉ nhận latin-1."""
    return quote(url, safe=_URL_SAFE)


@dataclass
class FetchResult:
    url: str
    status: str  # ok | http_<code> | not_image | too_large | not_http | timeout | error
    sha256: str | None = None
    path: str | None = None
    width: int | None = None
    height: int | None = None
    bytes: int | None = None
    content_type: str | None = None
    error: str | None = None
    fetched_at: str | None = None

    @property
    def final(self) -> bool:
        return self.status not in _RETRYABLE


def select_targets(rows: list[dict]) -> list[tuple[str, str | None]]:
    """(image_url, referer) của các dòng thiếu sha256, bỏ URL trùng và URL đã có ảnh cùng
    URL mang sha256 (merge.resolve_sha tự điền những dòng đó)."""
    known = {r["image_url"] for r in rows if r.get("sha256")}
    out: dict[str, str | None] = {}
    for r in rows:
        url = r.get("image_url")
        if url and not r.get("sha256") and url not in known:
            out.setdefault(url, r.get("source_url") or None)
    return list(out.items())


def select_all_targets(rows: list[dict]) -> list[tuple[str, str | None]]:
    """(image_url, referer) của MỌI dòng, bỏ URL trùng: tải file để mã hóa ảnh (bước 3).
    Ảnh đã có sha256 trong CSV vẫn giữ khóa đó; file tải về chỉ dùng để mã hóa (CDN có thể
    nén lại nên sha256 của file có thể khác)."""
    out: dict[str, str | None] = {}
    for r in rows:
        if r.get("image_url"):
            out.setdefault(r["image_url"], r.get("source_url") or None)
    return list(out.items())


def read_log(path: str | Path) -> dict[str, dict]:
    """URL -> kết quả mới nhất trong nhật ký."""
    path = Path(path)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {(d := json.loads(line))["url"]: d for line in fh if line.strip()}


def _image_size(data: bytes) -> tuple[int | None, int | None, bool]:
    """(rộng, cao, mở được). Không có Pillow thì không kiểm được, coi là mở được."""
    try:
        from PIL import Image
    except ImportError:
        return None, None, True
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height, True
    except Exception:
        return None, None, False


def fetch_one(
    url: str,
    referer: str | None,
    images_dir: Path,
    get: Callable[..., object],
) -> FetchResult:
    """`get` có chữ ký giống requests.Session.get (tiêm vào để test không cần mạng)."""
    now = datetime.now(UTC).isoformat(timespec="seconds")
    if not url.startswith(("http://", "https://")):
        # "/api/entities/local-image/…": ảnh nằm trên máy chủ nội bộ của tool gán nhãn
        return FetchResult(url, "not_http", fetched_at=now)
    headers = {"User-Agent": USER_AGENT}
    if referer:
        headers["Referer"] = ascii_url(referer)  # nhiều CDN báo chặn ảnh khi thiếu Referer
    try:
        resp = get(ascii_url(url), headers=headers, timeout=TIMEOUT_S, stream=True)
        try:
            if resp.status_code != 200:
                return FetchResult(url, f"http_{resp.status_code}", fetched_at=now)
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            buf = bytearray()
            for chunk in resp.iter_content(64 * 1024):
                buf += chunk
                if len(buf) > MAX_BYTES:
                    return FetchResult(url, "too_large", content_type=ctype, fetched_at=now)
        finally:
            resp.close()
    except Exception as e:  # requests.Timeout, ConnectionError, SSL...
        status = "timeout" if "timeout" in type(e).__name__.lower() else "error"
        return FetchResult(url, status, error=f"{type(e).__name__}: {e}"[:300], fetched_at=now)

    data = bytes(buf)
    width, height, ok = _image_size(data)
    if not ok or (ctype and not ctype.startswith("image/") and ctype != "application/octet-stream"):
        return FetchResult(url, "not_image", content_type=ctype, bytes=len(data), fetched_at=now)
    sha = hashlib.sha256(data).hexdigest()
    target = images_dir / sha[:2] / f"{sha}{_EXT.get(ctype, '.img')}"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(target)
    return FetchResult(url, "ok", sha, str(target), width, height, len(data), ctype, None, now)


def fetch_all(
    targets: Iterable[tuple[str, str | None]],
    log_path: str | Path,
    images_dir: str | Path,
    get: Callable[..., object],
    workers: int = 8,
    on_result: Callable[[FetchResult], None] | None = None,
) -> dict[str, int]:
    """Tải các URL chưa có kết quả cuối trong nhật ký; ghi nối tiếp từng kết quả."""
    log_path, images_dir = Path(log_path), Path(images_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    targets = list(targets)
    done = {u for u, d in read_log(log_path).items() if FetchResult(**d).final}
    todo = [(u, ref) for u, ref in targets if u not in done]
    counts: dict[str, int] = {"skipped_done": len(targets) - len(todo)}
    lock = threading.Lock()
    with (
        log_path.open("a", encoding="utf-8", newline="\n") as log,
        ThreadPoolExecutor(workers) as pool,
    ):
        futures = [pool.submit(fetch_one, u, ref, images_dir, get) for u, ref in todo]
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                log.write(json.dumps(asdict(res), ensure_ascii=False) + "\n")
                log.flush()
                counts[res.status] = counts.get(res.status, 0) + 1
            if on_result:
                on_result(res)
    return counts


def image_files(images: list[dict], log: dict[str, dict]) -> list[tuple[str, str]]:
    """(image_id, đường dẫn file đã tải) cho ảnh có trong nhật ký tải với trạng thái ok.
    `images`: bản ghi images.jsonl (cần image_id, image_url)."""
    out = []
    for img in images:
        rec = log.get(img.get("image_url") or "")
        if rec and rec.get("status") == "ok" and rec.get("path"):
            out.append((img["image_id"], rec["path"]))
    return out
