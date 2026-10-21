"""Kiểm soát truy cập cho bản public (SCENARIOS X4): mật khẩu dùng chung + giới hạn tần suất.

Đủ cho demo có link gửi hội đồng, không thay cho đăng nhập thật. HTTP Basic: trình duyệt tự hỏi
mật khẩu một lần rồi gửi kèm mọi request cùng origin (web, /search, /img); tên đăng nhập tùy ý.
Sau Cloudflare Tunnel mọi request đều đến từ 127.0.0.1, nên IP lấy từ header CF-Connecting-IP.
"""

import base64
import secrets
import threading
import time
from collections import defaultdict, deque

from starlette.requests import Request


def authorized(request: Request, password: str) -> bool:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "basic":
        return False
    try:
        _, _, given = base64.b64decode(value).decode("utf-8").partition(":")
    except ValueError:  # binascii.Error, UnicodeDecodeError, chuỗi không phải ASCII
        return False
    return secrets.compare_digest(given.encode(), password.encode())


def client_ip(request: Request) -> str:
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "?"
    )


class RateLimiter:
    """Cửa sổ trượt 60 s: tối đa `per_minute` request mỗi khóa (IP)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> deque[float]:
        q = self._hits[key]
        while q and now - q[0] >= 60:
            q.popleft()
        return q

    def allow(self, key: str, now: float | None = None) -> bool:
        """Ghi một lần và trả False nếu đã đủ hạn mức."""
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._recent(key, now)
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True

    def blocked(self, key: str, now: float | None = None) -> bool:
        """Đã đủ hạn mức chưa, không ghi thêm (chặn trước khi thử mật khẩu)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            return len(self._recent(key, now)) >= self.per_minute
