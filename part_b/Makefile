.PHONY: install install-index lint format test web run merge thumbs index eval up down bench

PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

install-index:
	$(PY) -m pip install -e ".[index,dev]"

lint:
	ruff check src tests scripts bench
	ruff format --check src tests scripts bench

format:
	ruff format src tests scripts bench
	ruff check --fix src tests scripts bench

test:
	$(PY) -m pytest -q

web:
	cd web && npm install && npm run build

run:
	uvicorn imgsearch.api.main:app --reload --port 8000

# --- Dựng dữ liệu và index (đọc đầu ra Phần A trong ../data) ---
merge:
	$(PY) scripts/merge_data.py

thumbs:
	$(PY) scripts/make_thumbs.py

index:
	$(PY) scripts/build_index.py

eval:
	$(PY) -m imgsearch.eval.run_eval

# --- Docker ---
up:
	docker compose up -d --build

down:
	docker compose down

bench:
	docker compose --profile bench run --rm bench
	$(PY) bench/make_report.py
