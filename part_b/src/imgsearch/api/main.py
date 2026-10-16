"""FastAPI: /search, /healthz, /metrics và phục vụ web tĩnh.

Latency truy xuất được báo qua header Server-Timing theo từng bước (PLAN.md 4.6).
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from imgsearch.api.schemas import SearchRequest, SearchResponse
from imgsearch.config import get_settings
from imgsearch.retrieval.engine import SearchEngine

state: dict[str, SearchEngine | None] = {"engine": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    engine = SearchEngine(settings)
    engine.load()  # tải index (từ S3 nếu INDEX_FROM_S3) rồi mới báo healthy
    state["engine"] = engine
    yield
    state["engine"] = None


app = FastAPI(title="imgsearch", version="0.1.0", lifespan=lifespan)


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
    result, timings = engine.search(req)
    timings["total"] = (time.perf_counter() - t0) * 1000
    response.headers["Server-Timing"] = ", ".join(
        f"{name};dur={ms:.1f}" for name, ms in timings.items()
    )
    return result


@app.get("/metrics")
def metrics() -> dict:
    # TODO: đếm request, truy vấn rỗng, tỷ lệ gọi LLM, histogram latency.
    return {}


_web_dir = get_settings().web_dir
if _web_dir.exists():
    app.mount("/", StaticFiles(directory=_web_dir, html=True), name="web")
