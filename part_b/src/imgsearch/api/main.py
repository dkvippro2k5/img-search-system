"""FastAPI: /search, /healthz, /metrics và phục vụ web tĩnh.

Latency truy xuất được báo qua header Server-Timing theo từng bước (PLAN.md 4.6).
"""

import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from imgsearch.api.access import RateLimiter, authorized, client_ip
from imgsearch.api.local_images import LocalImages
from imgsearch.api.metrics import Metrics
from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.config import get_settings
from imgsearch.retrieval.engine import SearchEngine

state: dict = {"engine": None, "local_images": None}
_IMAGE_ID = re.compile(r"[0-9a-f]{64}")
# Giới hạn số truy vấn tính toán cùng lúc (SEARCH_CONCURRENCY); request còn lại xếp hàng.
# Không giới hạn thì N request x (ONNX + numpy nhiều luồng) tranh nhau vài CPU, thông lượng
# giảm khi tăng tải (reports/perf.md).
_n = get_settings().search_concurrency
_gate = threading.BoundedSemaphore(_n) if _n > 0 else None
metrics_store = Metrics()
_limiter = RateLimiter(get_settings().rate_limit_per_minute)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = SearchEngine(settings)
    engine.load()  # tải index (từ S3 nếu INDEX_FROM_S3) rồi mới báo healthy
    state["engine"] = engine
    if settings.local_images and engine.store is not None:
        state["local_images"] = LocalImages(settings.data_dir, engine.store)
    yield
    state["engine"] = None
    state["local_images"] = None


app = FastAPI(title="imgsearch", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def access_control(request: Request, call_next):
    """Bản public (X4): DEMO_PASSWORD hỏi mật khẩu cho mọi đường dẫn trừ /healthz (healthcheck
    của Docker); RATE_LIMIT_PER_MINUTE giới hạn /search theo IP."""
    settings = get_settings()
    path = request.url.path
    pw = settings.demo_password
    if pw and path != "/healthz" and not authorized(request, pw):
        realm = 'Basic realm="imgsearch demo", charset="UTF-8"'
        return Response(status_code=401, headers={"WWW-Authenticate": realm})
    if _limiter.per_minute and path == "/search" and not _limiter.allow(client_ip(request)):
        body = {"detail": "Quá nhiều truy vấn, thử lại sau ít giây."}
        return JSONResponse(body, status_code=429, headers={"Retry-After": "10"})
    return await call_next(request)


@app.get("/healthz")
def healthz() -> JSONResponse:
    engine = state["engine"]
    ready = engine is not None and engine.ready
    body = {"status": "ok" if ready else "loading", **(engine.info() if engine else {})}
    return JSONResponse(body, status_code=200 if ready else 503)


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, response: Response) -> SearchResponse:
    engine = state["engine"]
    t0 = time.perf_counter()
    try:
        if _gate is None:
            result, timings = engine.search(req)
        else:
            # Gửi embedding kênh văn bản (gọi mạng) trước khi xếp hàng: chờ mạng chồng lên chờ
            # semaphore thay vì giữ chỗ tính toán trong lúc CPU rảnh.
            engine.prefetch(req.query)
            with _gate:
                t_wait = time.perf_counter()
                result, timings = engine.search(req)
            timings = {"queue": (t_wait - t0) * 1000, **timings}
    except Exception:
        metrics_store.observe_error()
        raise
    timings["total"] = (time.perf_counter() - t0) * 1000
    metrics_store.observe(result, timings)
    response.headers["Server-Timing"] = ", ".join(
        f"{name};dur={ms:.1f}" for name, ms in timings.items()
    )
    return result


@app.get("/img/{image_id}", response_model=None)
def local_image(image_id: str) -> Response:
    """Demo cục bộ (LOCAL_IMAGES=true): trả ảnh đã tải trên máy; không có file thì chuyển hướng
    tới image_url gốc."""
    images: LocalImages | None = state["local_images"]
    if images is None or not _IMAGE_ID.fullmatch(image_id):
        return Response(status_code=404)
    path = images.path(image_id)
    if path is not None:
        return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})
    url = images.url(image_id)
    return RedirectResponse(url, status_code=302) if url else Response(status_code=404)


@app.get("/metrics")
def metrics() -> dict:
    """Số đo từ lúc khởi động: request, truy vấn rỗng, latency, lỗi/429 của mentor."""
    engine = state["engine"]
    return {
        **metrics_store.snapshot(),
        "mentor": engine.mentor_stats() if engine else None,
        "index": engine.info() if engine else None,
    }


_web_dir = get_settings().web_dir
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")
