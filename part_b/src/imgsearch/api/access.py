"""Kiểm soát truy cập cho bản public (SCENARIOS X4): mật khẩu dùng chung + giới hạn tần suất.

Đủ cho demo có link gửi hội đồng, không thay cho đăng nhập thật. HTTP Basic: trình duyệt tự hỏi
mật khẩu một lần rồi gửi kèm mọi request cùng origin (web, /search, /img); tên đăng nhập tùy ý.
Sau Cloudflare Tunnel mọi request đều đến từ 127.0.0.1, nên IP lấy từ header CF-Connecting-IP.
"""

import base64
import binascii
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
    except (binascii.Error, UnicodeDecodeError):
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

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            return True
