.PHONY: install install-index lint format test web run fetch merge encode index text eval up down bench

PY ?= python
VER ?= people

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

# --- Dựng dữ liệu và index từ ../data/raw/people-*.csv (không có trong repo) ---
fetch:
	$(PY) scripts/fetch_missing_images.py --all

merge:
	$(PY) scripts/merge_data.py

# cần extra embed (torch CPU)
encode:
	$(PY) -X utf8 scripts/encode_images.py
	$(PY) -X utf8 scripts/export_text_encoder.py

index:
	$(PY) scripts/build_index.py --input ../data/merged/people --version $(VER)

# kênh văn bản: cần index và key mentor trong .env
text:
	$(PY) -X utf8 scripts/encode_texts.py --index-version $(VER) --write-index

eval:
	$(PY) -X utf8 -m imgsearch.eval.run_testset --index-version $(VER) --split dev

# --- Docker ---
up:
	docker compose up -d --build

down:
	docker compose down

bench:
	$(PY) bench/run_bench.py
