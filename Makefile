# ==============================================================================
# AI Pharmacovigilance Intelligence Platform — Makefile
# ==============================================================================
# Usage: make <target>
# ==============================================================================

PYTHON      := python
PIP         := pip
VENV_DIR    := venv
VENV_PYTHON := $(VENV_DIR)/bin/python
API_HOST    := 0.0.0.0
API_PORT    := 8000
DASH_PORT   := 8501
N_RECORDS   := 10000

# Windows compatibility
ifeq ($(OS),Windows_NT)
    VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
    PIP_CMD     := $(VENV_DIR)/Scripts/pip.exe
else
    PIP_CMD     := $(VENV_DIR)/bin/pip
endif

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:
	@echo "=================================================================="
	@echo "  AI Pharmacovigilance Intelligence Platform"
	@echo "=================================================================="
	@echo ""
	@echo "  Setup:"
	@echo "    make setup          Full environment setup"
	@echo "    make install        Install Python dependencies"
	@echo "    make spacy          Download spaCy language model"
	@echo ""
	@echo "  Data & Pipeline:"
	@echo "    make ingest         Run data ingestion pipeline"
	@echo "    make pipeline       Run full pipeline (ingest + NLP + signals + report)"
	@echo "    make detect         Run risk signal detection only"
	@echo "    make report         Generate JSON safety report"
	@echo ""
	@echo "  Services:"
	@echo "    make api            Start FastAPI backend"
	@echo "    make dashboard      Start Streamlit dashboard"
	@echo "    make api-dev        Start API in development mode"
	@echo ""
	@echo "  Testing:"
	@echo "    make test           Run all tests"
	@echo "    make test-unit      Run unit tests only"
	@echo "    make test-integration  Run integration tests"
	@echo "    make coverage       Run tests with coverage report"
	@echo ""
	@echo "  Maintenance:"
	@echo "    make lint           Run code linting"
	@echo "    make format         Auto-format code"
	@echo "    make clean          Clean build artifacts"
	@echo "    make clean-db       Remove database file"
	@echo "    make clean-all      Full clean (including venv)"
	@echo "=================================================================="

# ── Setup ─────────────────────────────────────────────────────────────────────
.PHONY: setup
setup:
	@echo "Setting up AI Pharmacovigilance Platform..."
	$(PYTHON) scripts/setup_environment.py

.PHONY: install
install:
	$(PIP_CMD) install --upgrade pip
	$(PIP_CMD) install -r requirements.txt
	@echo "Dependencies installed."

.PHONY: spacy
spacy:
	$(VENV_PYTHON) -m spacy download en_core_web_sm
	@echo "spaCy model downloaded."

.PHONY: env
env:
	@test -f .env || cp .env.example .env
	@echo ".env file ready."

# ── Data & Pipeline ───────────────────────────────────────────────────────────
.PHONY: ingest
ingest:
	@echo "Running data ingestion pipeline..."
	$(VENV_PYTHON) pipelines/pipeline_orchestrator.py ingest --n-records $(N_RECORDS)

.PHONY: pipeline
pipeline:
	@echo "Running full pipeline..."
	$(VENV_PYTHON) pipelines/pipeline_orchestrator.py full --n-records $(N_RECORDS)

.PHONY: detect
detect:
	@echo "Running risk signal detection..."
	$(VENV_PYTHON) pipelines/pipeline_orchestrator.py detect

.PHONY: report
report:
	@echo "Generating safety report..."
	$(VENV_PYTHON) pipelines/pipeline_orchestrator.py report --format json

# ── Services ──────────────────────────────────────────────────────────────────
.PHONY: api
api:
	@echo "Starting FastAPI backend on http://$(API_HOST):$(API_PORT)"
	$(VENV_PYTHON) -m uvicorn api_gateway.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--workers 1

.PHONY: api-dev
api-dev:
	@echo "Starting FastAPI in development mode..."
	$(VENV_PYTHON) -m uvicorn api_gateway.main:app \
		--host $(API_HOST) \
		--port $(API_PORT) \
		--reload \
		--log-level debug

.PHONY: dashboard
dashboard:
	@echo "Starting Streamlit dashboard on http://localhost:$(DASH_PORT)"
	$(VENV_PYTHON) -m streamlit run dashboard/app.py \
		--server.port $(DASH_PORT) \
		--server.address 0.0.0.0 \
		--server.headless true

# ── Testing ───────────────────────────────────────────────────────────────────
.PHONY: test
test:
	$(VENV_PYTHON) -m pytest tests/ -v

.PHONY: test-unit
test-unit:
	$(VENV_PYTHON) -m pytest tests/unit/ -v

.PHONY: test-integration
test-integration:
	$(VENV_PYTHON) -m pytest tests/integration/ -v

.PHONY: coverage
coverage:
	$(VENV_PYTHON) -m pytest tests/ \
		--cov=services \
		--cov=database \
		--cov=api_gateway \
		--cov-report=term-missing \
		--cov-report=html:htmlcov
	@echo "Coverage report: htmlcov/index.html"

# ── Code Quality ──────────────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(VENV_PYTHON) -m flake8 services/ api_gateway/ database/ pipelines/ \
		--max-line-length=110 \
		--ignore=E501,W503

.PHONY: format
format:
	$(VENV_PYTHON) -m black services/ api_gateway/ database/ dashboard/ pipelines/ tests/ \
		--line-length 110
	$(VENV_PYTHON) -m isort services/ api_gateway/ database/ dashboard/ pipelines/ tests/ \
		--profile black

.PHONY: typecheck
typecheck:
	$(VENV_PYTHON) -m mypy services/ api_gateway/ database/ \
		--ignore-missing-imports \
		--no-strict-optional

# ── Cleanup ───────────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage 2>/dev/null || true
	@echo "Cleaned."

.PHONY: clean-db
clean-db:
	rm -f data/pharmacovigilance.db
	@echo "Database removed."

.PHONY: clean-data
clean-data:
	rm -rf data/raw/* data/processed/* data/exports/*
	@echo "Data directories cleaned."

.PHONY: clean-logs
clean-logs:
	rm -rf logs/*.log
	@echo "Logs cleaned."

.PHONY: clean-all
clean-all: clean clean-db clean-data clean-logs
	rm -rf venv/
	@echo "Full clean complete."

# ── Docker ────────────────────────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker build -t ai-pharmacovigilance:latest .

.PHONY: docker-run
docker-run:
	docker-compose up -d

.PHONY: docker-stop
docker-stop:
	docker-compose down

# ── Database ──────────────────────────────────────────────────────────────────
.PHONY: db-init
db-init:
	$(VENV_PYTHON) -c "from database.connection import create_all_tables; create_all_tables(); print('Schema created.')"

.PHONY: db-reset
db-reset: clean-db db-init
	@echo "Database reset complete."
