# syntax=docker/dockerfile:1

# --- Stage 1: build web tĩnh (TypeScript -> JS) ---
FROM node:22-slim AS web
WORKDIR /web
COPY web/package.json web/tsconfig.json ./
RUN npm install --no-audit --no-fund
COPY web/ ./
RUN npm run build

# --- Stage 2: cài wheel Python (không torch) ---
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install .

# --- Stage 3: runtime ---
FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN useradd --create-home app \
    # volume index-cache gắn vào /cache: tạo sẵn thuộc app để volume mới lấy quyền này
    && mkdir -p /cache && chown app:app /cache
WORKDIR /app
COPY --from=build /opt/venv /opt/venv
COPY --from=web /web/index.html /web/style.css web/
COPY --from=web /web/dist web/dist
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"
CMD ["uvicorn", "imgsearch.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
