.PHONY: venv install download-data sample-data train test lint api build up down smoke logs clean

VENV := mlops
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

download-data:
	$(PY) scripts/download_data.py

sample-data:
	$(PY) scripts/generate_sample_data.py

split-data:
	$(PY) -c "from src.data_processing import create_splits; print(create_splits('data/raw', 'data/processed'))"

train:
	$(PY) -m src.train

test:
	$(PY) -m pytest tests/ -v

api:
	$(VENV)/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

build:
	docker build -t cats-dogs-api:local .

up:
	docker compose up -d --build

down:
	docker compose down

smoke:
	bash scripts/smoke_test.sh http://localhost:8000

track:
	$(PY) scripts/track_performance.py

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
