# ==============================================================================
# Trading System Makefile
# Unified interface for infrastructure, execution, and development tasks.
# ==============================================================================

.PHONY: help up down status-infra install setup-env init-db login refresh-universe \
        backfill-ohlcv run-paper run-live run-equity run-crypto go-live-check train train-crypto \
        promote-model status-risk status-model reset-breaker report-daily report-weekly \
        lint format test type-check check-all clean setup start dev

# Default market for run-live (override with: MARKET=both make run-live)
MARKET ?= equity

# Master commands
setup:
	@bash scripts/bootstrap.sh

start: up
	@echo "Starting orchestrator..."
	uv run python -m orchestrator.main

dev: setup login backfill-ohlcv start

# Default target: show help
help:
	@echo "Trading System Control Center"
	@echo "============================"
	@echo "Master Commands (One-Command Flow):"
	@echo "  make setup            ONE-TIME SETUP: env, deps, infra, db schema"
	@echo "  make start            RUN SYSTEM: starts infra + paper trading"
	@echo "  make dev              TOTAL FLOW: setup -> login -> backfill -> start"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make up               Start core services (TimescaleDB, Redis, MLflow)"
	@echo "  make down             Stop core services"
	@echo "  make status-infra     Show docker container status"
	@echo ""
	@echo "Setup & Data:"
	@echo "  make install          Install dependencies using uv"
	@echo "  make setup-env        Create .env from example"
	@echo "  make init-db          Initialise database schema and hypertables"
	@echo "  make login            Initial Zerodha Kite login (OAuth)"
	@echo "  make refresh-universe Refresh NSE instrument universe"
	@echo "  make backfill-ohlcv   Backfill 3 years of daily OHLCV data"
	@echo ""
	@echo "Execution:"
	@echo "  make run-paper        Run orchestrator in paper mode (default)"
	@echo "  make go-live-check    Pre-flight validation before going live"
	@echo "  make run-live         Run orchestrator in live mode (DANGER — runs go-live-check first)"
	@echo "  MARKET=both make run-live   Live mode for both equity and crypto"
	@echo "  make run-equity       Run NSE Equity only"
	@echo "  make run-crypto       Run Binance Crypto only"
	@echo ""
	@echo "Training & Models:"
	@echo "  make train            Train NSE model (RELIANCE baseline)"
	@echo "  make train-crypto     Train Crypto ensemble models"
	@echo "  make promote-model    Register a model run-id as production"
	@echo ""
	@echo "Monitoring & Risk:"
	@echo "  make status-risk      Show circuit-breaker and drawdown status"
	@echo "  make status-model     Show currently active production model"
	@echo "  make reset-breaker    Manual reset of the risk circuit breaker"
	@echo "  make report-daily     Show summary of today's trades"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff check and fix"
	@echo "  make format           Run ruff format"
	@echo "  make test             Run pytest (mocked)"
	@echo "  make type-check       Run mypy on all core modules"
	@echo "  make check-all        Run all quality checks (lint, format, test, type)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove caches and temporary files"

# --- Infrastructure ---

up:
	docker compose up -d timescaledb redis mlflow

down:
	docker compose down

status-infra:
	docker compose ps

# --- Setup & Data ---

install:
	uv sync

setup-env:
	test -f .env || cp .env.example .env
	@echo ".env created. Please update with your credentials."

init-db:
	uv run python -m data.store --init-schema

login:
	uv run python -m data.ingest --login

refresh-universe:
	uv run python -m data.universe --refresh

backfill-ohlcv:
	uv run python -m data.ingest --backfill --days 1095

# --- Execution ---

run-paper:
	uv run python -m orchestrator.main

go-live-check:
	uv run python scripts/go_live_check.py

run-live: go-live-check
	@echo "WARNING: Running in LIVE mode with real capital."
	@echo "Pre-flight check passed. Starting live orchestrator..."
	PRODUCTION_LIVE_CONFIRMED=true PAPER_TRADE_MODE=false uv run python -m orchestrator.main --live --market $(MARKET)

run-equity:
	uv run python -m orchestrator.main --market equity

run-crypto:
	uv run python -m orchestrator.main --market crypto

# --- Training ---

train:
	uv run python -m signals.train --symbol RELIANCE --start 2021-01-01 --end 2024-01-01

train-crypto:
	uv run python -m backtest.train_crypto

promote-model:
	@read -p "Enter MLflow Run ID: " run_id; \
	uv run python -m signals.model --register --run-id $$run_id --alias production

# --- Monitoring ---

status-risk:
	uv run python -m risk.breakers --status

status-model:
	uv run python -m signals.model --status

reset-breaker:
	uv run python -m risk.breakers --reset

report-daily:
	uv run python -m execution.logger --summary --date today

report-weekly:
	uv run python -m monitoring.reporters --weekly

# --- Code Quality ---

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

test:
	uv run pytest tests/ -x -q --no-cov

type-check:
	uv run mypy audit/ data/ signals/ orchestrator/ execution/ risk/ monitoring/ llm/ backtest/ options/ portfolio/

check-all: lint format test type-check

# --- Cleanup ---

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
