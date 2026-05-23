# NSE/Crypto Algorithmic Trading System

A production-grade algorithmic trading system for NSE equities and crypto, combining XGBoost/LightGBM/PatchTST ensemble signals, FinBERT sentiment, and hard risk guardrails — connected to Zerodha Kite Connect.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [First-Time Setup](#first-time-setup)
4. [Configuration Reference](#configuration-reference)
5. [Running the System](#running-the-system)
6. [The Hermes Dual-Engine](#the-hermes-dual-engine)
7. [Signal Flow](#signal-flow)
8. [Daily Schedule](#daily-schedule)
9. [Risk Controls](#risk-controls)
10. [Model Training & Registry](#model-training--registry)
11. [Backtesting](#backtesting)
12. [Monitoring & Alerts](#monitoring--alerts)
13. [Code Quality](#code-quality)
14. [Project Layout](#project-layout)

---

## Architecture Overview

```
Market Data (Kite / Binance)
        │
        ▼
  [data/] Ingestor → TimescaleDB (OHLCV) ─────────────────────┐
                   → Redis (tick cache, CB state, token)        │
                                                                │
  [llm/] FinBERT sentiment ──────────────────────────────────► │
                                                                ▼
  [signals/] ScannerEngine ──► XGBoost / LightGBM / PatchTST ─►
             MomentumSentinel    ensemble (Phase 8)
             WealthArchitect     × AlphaEngine (sector + options + regime)
                                 = final probability score
                                          │
                               score ≥ SIGNAL_THRESHOLD (0.65)?
                                          │
                                     [risk/] CircuitBreaker check
                                     [risk/] PositionSizer (half-Kelly)
                                          │
                               [execution/] OrderExecutor
                               paper_trades | live_trades
                                          │
                               [monitoring/] Telegram + MLflow
```

**Tech stack:** Python 3.13, Redis, TimescaleDB (PostgreSQL), XGBoost, LightGBM, PatchTST, FinBERT, APScheduler, MLflow, Structlog, Pydantic-settings, Zerodha Kite Connect.

---

## Prerequisites

### Infrastructure

| Service | Purpose | Default |
|---------|---------|---------|
| TimescaleDB (PostgreSQL 16) | OHLCV, trades, signals, audit log | `localhost:5432` |
| Redis 7 | Tick cache, circuit-breaker state, access token | `localhost:6379` |
| MLflow | Experiment tracking, model registry | `localhost:5001` |

Start all three with Docker Compose:

```bash
docker compose up -d timescaledb redis mlflow
```

### Python

Python ≥ 3.13 required. Use `uv` for dependency management:

```bash
uv sync
```

For Upstox support (optional):

```bash
uv sync --extra upstox
```

---

## First-Time Setup

### 1. Environment variables

```bash
cp .env.example .env
```

Edit `.env` — minimum required fields:

```dotenv
KITE_API_KEY=your_api_key
KITE_API_SECRET=your_api_secret
TIMESCALE_URL=postgresql://trader:password@localhost:5432/nse_trading
REDIS_URL=redis://localhost:6379/0
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

See [Configuration Reference](#configuration-reference) for all variables.

### 2. Initialise the database schema

Runs Alembic migrations and creates TimescaleDB hypertables:

```bash
python -m data.store --init-schema
```

### 3. Kite login (first time)

```bash
python -m data.ingest --login
```

Opens a browser to Zerodha OAuth. After authorising, the session token is cached in Redis. Subsequent logins happen automatically at 08:45 IST each morning.

### 4. Seed the instrument universe

```bash
python -m data.universe --refresh
```

Downloads the NSE instrument list from Kite Connect. Re-run at weekends whenever NSE updates instruments.

### 5. Backfill historical OHLCV

```bash
python -m data.ingest --backfill --days 1095
```

Pulls daily bars for the top 50 liquid instruments (≈3 years). Takes 10–20 minutes on first run.

---

## Configuration Reference

All settings live in `config/settings.py` and are loaded from `.env` (Pydantic-settings with validation at startup).

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_PROVIDER` | `kite` | `kite` or `upstox` |
| `KITE_API_KEY` | — | Zerodha Kite API key (required) |
| `KITE_API_SECRET` | — | Zerodha Kite API secret (required) |
| `KITE_ACCESS_TOKEN` | — | Generated daily by the pre-market job |
| `TIMESCALE_URL` | — | PostgreSQL connection string (required) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | MLflow server URL |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | — | Telegram chat/channel ID |
| `SIGNAL_THRESHOLD` | `0.65` | Minimum model probability to generate a signal |
| `MAX_POSITION_PCT` | `0.02` | Maximum capital per position (2%) |
| `DAILY_DD_LIMIT` | `0.03` | Daily drawdown circuit-breaker limit (3%) |
| `WEEKLY_DD_LIMIT` | `0.07` | Weekly drawdown circuit-breaker limit (7%) |
| `PAPER_TRADE_MODE` | `true` | Paper mode — no real orders placed |
| `FINNHUB_API_KEY` | — | Finnhub news API key (optional) |
| `NEWSAPI_KEY` | — | NewsAPI key (optional) |

Strategy-level parameters (PE thresholds, momentum lookbacks, etc.) are in `config/strategy_params.yaml`.

---

## Running the System

### Paper trading (recommended starting point)

Paper mode is the default. Orders are logged to `paper_trades` but never sent to Kite.

```bash
python -m orchestrator.main
```

The scheduler starts and logs all registered jobs. If outside market hours, jobs fire at their next scheduled time. Run until stopped with `Ctrl+C`.

**Verify it is working:**

```bash
# Check recent paper trades
python -m execution.logger --summary --date today

# Check circuit-breaker state
python -m risk.breakers --status

# Check active signal model
python -m signals.model --status
```

After 200+ paper trades, review metrics before switching to live:

```bash
python -m execution.logger --summary --days 30
```

Target: win rate > 50% and mean P&L per trade above transaction costs.

### Live trading

> **Warning:** Live mode places real orders with real money. Enable only after completing the paper trading checklist.

In `.env`:

```dotenv
PAPER_TRADE_MODE=false
```

Restart the system. The startup log emits `live_trading_mode_active` to confirm.

### Market mode

The system supports three market modes:

```bash
# Equity only (NSE, weekdays IST)
python -m orchestrator.main --market equity

# Crypto only (Binance, 24/7 UTC)
python -m orchestrator.main --market crypto

# Both simultaneously
python -m orchestrator.main --market both
```

---

## The Hermes Dual-Engine

Hermes is the strategy layer — two legs operating concurrently:

### Leg 1: Momentum Sentinel (Aggressive — Intraday/Daily)

Scans the equity universe for high-momentum breakout candidates. Runs inside `ScannerEngine` as a `BaseStrategy` subclass (pure CPU, no I/O — runs in a ProcessPoolExecutor worker).

**Criteria:**
- Close price > SMA(50)
- Volume > 2× the 20-day average volume
- RSI(14) > 60 (Wilder-smoothed)

**Outputs:** signal dict with `symbol`, `volume_ratio`, `rsi_14`, `distance_to_sma50_pct`, sorted by volume ratio descending.

**When it runs:** Every 5 minutes during the trading loop (09:15–15:25 IST).

### Leg 2: Wealth Architect (Conservative — Weekly SIP Scanner)

Screens a symbol universe for blue-chip compounding candidates using fundamentals cached in Redis. Runs standalone (not in a worker) because it reads Redis.

**Criteria:**
- PE ratio < sector average PE (fallback cap: 25 if sector average unavailable)
- ROE > 15%

**Redis cache format** (key: `FUND:{symbol}`):

```json
{
  "pe": 18.5,
  "roe": 22.4,
  "sector": "Banking",
  "sector_avg_pe": 21.0
}
```

**When it runs:** Saturday 09:00 IST (weekly). Top-3 candidates are sent to Telegram.

**Populate the fundamentals cache:**

```python
import json
import redis

r = redis.from_url("redis://localhost:6379/0")
r.set("FUND:INFY", json.dumps({
    "pe": 24.5,
    "roe": 29.0,
    "sector": "IT",
    "sector_avg_pe": 28.0
}))
```

### AlphaEngine (Composite Multiplier)

Both legs feed through the `AlphaEngine` which applies a scalar multiplier (0.5× – 1.5×) to the base model probability based on:

1. **Sector strength** — `SectorRanker` computes 5-day and 20-day relative strength vs. Nifty 50. Top-sector stocks get a boost; bottom-sector stocks get suppressed (or boosted on the SELL side).

2. **Options flow** — `OptionsFlowAnalyzer` reads intraday OI changes and PCR trends. Returns an institutional sentiment score from −1.0 to +1.0.

3. **Regime alignment** — Maps the current market regime to a multiplier delta:

   | Regime | BUY delta | SELL delta |
   |--------|-----------|------------|
   | `trending_bull` | +0.10 | −0.10 |
   | `trending_bear` | −0.30 | +0.10 |
   | `choppy` | −0.20 | −0.10 |
   | `high_vol` | −0.10 | 0.00 |
   | `normal` | 0.00 | 0.00 |

**Final confidence = model probability × AlphaEngine multiplier**

This final confidence is what `PositionSizer` uses for Kelly-fraction sizing.

---

## Signal Flow

```
1. Pre-market (08:45 IST)
   └── Refresh Kite token
   └── Run FinBERT sentiment on overnight news
   └── Load production model from MLflow registry

2. Trading loop (every 5 min, 09:15–15:25 IST)
   └── ScannerEngine.scan_universe()
       └── MomentumSentinelStrategy.scan(symbol, df) [worker]
       └── VCPStrategy.scan(symbol, df) [worker]
       └── (+ any enabled strategy in strategy_params.yaml)
   └── SignalValidator.validate(signal)
   └── AlphaEngine.calculate_multiplier(symbol, sector, regime, side)
   └── final_prob = xgb_prob * alpha_multiplier
   └── CircuitBreaker.is_halted() → reject if true
   └── PositionSizer.size(symbol, prob, portfolio) → qty, risk_pct
   └── OrderExecutor.place(symbol, qty, side) → paper or live

3. Post-market (15:35 IST)
   └── Compute daily P&L, win rate, signal count
   └── Telegram daily summary

4. Wealth Architect (Saturday 09:00 IST)
   └── WealthArchitectScanner.run(universe)
   └── Telegram: top-3 SIP candidates ranked by ROE
```

---

## Daily Schedule

All times are IST (Asia/Kolkata) unless noted.

| Time | Job | What it does |
|------|-----|-------------|
| Mon 06:00 | Concept drift check | KL divergence on last week's features vs. reference distribution |
| Mon 08:30 | Weekly CB reset | Resets weekly circuit-breaker baseline to current capital |
| Mon–Fri 08:45 | Pre-market setup | Token refresh, FinBERT pipeline, load model from MLflow |
| Mon–Fri 09:15–15:25 | Trading loop (every 5 min) | Scan → Signal → Risk gate → Order |
| Mon–Fri 15:35 | Post-market summary | Logs daily P&L, checks model drift |
| Mon–Fri 16:00 | Daily reporting | Sends daily report to Telegram |
| Fri 17:00 | Weekly reporting | Sends weekly performance summary to Telegram |
| Fri 17:05 | A/B test reporting | Champion vs. challenger comparison, promotion check |
| Sat 09:00 | Wealth Architect scan | Conservative fundamental scan, sends top-3 to Telegram |
| Sun 02:00 | Retrain check | Flags win-rate drift; does not retrain automatically |
| 1st of month 02:00 | Ensemble retraining | Walk-forward retrain: XGBoost + LightGBM + PatchTST |
| Jan/Apr/Jul/Oct 1 03:00 | Quarterly HPO | Bayesian hyperparameter optimisation, saves best params to Redis |
| Last day of month 17:00 | Monthly reporting | Strategy rankings, multibagger candidates, P&L review |

**Crypto jobs** run on the same pattern but in UTC, 24/7.

---

## Risk Controls

### Circuit Breaker

Three independent halt conditions — any one triggers a full halt:

| Condition | Default | Env variable |
|-----------|---------|-------------|
| Daily drawdown | 3% | `DAILY_DD_LIMIT` |
| Weekly drawdown | 7% | `WEEKLY_DD_LIMIT` |
| Consecutive losses | 5 | `max_consecutive_losses` in `risk/breakers.py` |

The halt state is persisted to Redis immediately on trigger. It **never** auto-resets.

```bash
# Check status
python -m risk.breakers --status

# Manual reset (after investigating the cause)
python -m risk.breakers --reset
```

### Position Sizer

Uses half-Kelly with three adjustments:

1. **Volatility scaling** — position shrinks when 20-day ATR is elevated
2. **Correlation penalty** — reduces size when the symbol is highly correlated with existing positions
3. **Regime multiplier** — reduces size in choppy or high-volatility regimes

Hard caps:
- **2%** max capital per position (`MAX_POSITION_PCT`)
- **8%** max portfolio heat (sum of all open position risk)

### Signal Validator

All signals must pass `SignalValidator` before reaching the order executor. Rejects signals with:
- Missing required fields
- Price outside circuit-limit bands
- Volume below minimum threshold
- Symbol not in approved universe

---

## Model Training & Registry

### Train from scratch

```bash
python -m signals.train --symbol RELIANCE --start 2021-01-01 --end 2024-01-01
```

Uses walk-forward cross-validation (24-month train, 3-month test, 5-day purge gap). Results and the best model are logged to MLflow at `http://localhost:5001`.

### Register for production

After reviewing the run in the MLflow UI:

```bash
python -m signals.model --register --run-id <mlflow_run_id> --alias production
```

The orchestrator loads the `production` alias at each pre-market setup. Old models remain in the registry for rollback.

### Ensemble (Phase 8)

When `ab_test_enabled=true` in settings, the system uses a three-model ensemble:

- **XGBoost** (40% weight)
- **LightGBM** (30% weight)
- **PatchTST** (30% weight)

Voting strategy: `majority` (default) or `weighted`. Controlled by `config/strategy_params.yaml`:

```yaml
ensemble:
  model_dir: ./models/ensemble
  voting_strategy: majority
  weights: [0.4, 0.3, 0.3]
  confidence_threshold: 0.65
```

Monthly automated retraining runs at 02:00 IST on the 1st of each month.

### A/B Testing

Champion vs. challenger model routing at 50/50. Results tracked in Redis (30-day TTL) with statistical comparison (t-test, win rate, Sharpe ratio). Weekly reports sent every Friday 17:05 IST.

---

## Backtesting

```bash
# Run walk-forward backtest
python -m backtest.runner --strategy momentum_sentinel --start 2022-01-01 --end 2024-01-01

# Results are saved to ./results/ with a timestamp
ls results/
```

The backtester applies realistic costs: exchange fees (0.0003), STT (0.001 sell-side), SEBI charges, GST. Results are logged to MLflow for comparison against live performance.

---

## Monitoring & Alerts

### Telegram alerts

Configure `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`. The system sends:

- **Pre-market**: Regime state + sentiment score
- **Signal alerts**: Symbol, direction, confidence, entry/stop/target (rate-limited: 1 per symbol per 5 minutes)
- **Trade alerts**: Entry/exit with P&L
- **Risk alerts**: Sector concentration, correlation, liquidity warnings
- **Daily summary**: 16:00 IST — scans, signals, trades, P&L, win rate
- **Weekly summary**: Friday 17:00 IST — strategy rankings, performance
- **Saturday**: Top-3 Wealth Architect SIP candidates ranked by ROE

### MLflow

Open the experiment tracking UI:

```
http://localhost:5001
```

Tracks: model runs, hyperparameters, feature importance, walk-forward metrics, drift alerts.

### Grafana

Pre-built dashboards in `deploy/grafana/provisioning/`. Start with:

```bash
docker compose --profile monitoring up -d grafana prometheus
```

Open `http://localhost:3000` (default admin password: from `GRAFANA_ADMIN_PASSWORD`).

---

## Code Quality

Run all checks before committing:

```bash
# Tests (all mocked — no external services needed)
uv run pytest tests/ -x -q --no-cov

# Format
uv run ruff format .

# Lint
uv run ruff check . --fix

# Type check
uv run mypy data/ signals/ orchestrator/ execution/ risk/ monitoring/ llm/ backtest/ options/

# Security scan
uv run bandit -c pyproject.toml -r .

# Dependency audit
uv run pip-audit
```

Pre-commit hooks run `ruff` + `bandit` automatically on every commit after:

```bash
uv run pre-commit install
```

Coverage must stay above 80%. Current: ~88%.

---

## Project Layout

```
trading-system/
├── config/              Settings (Pydantic-settings) + strategy_params.yaml
├── data/                Kite/Upstox ingestor, TimescaleDB store, universe
├── signals/
│   ├── strategies/      BaseStrategy subclasses (VCP, MomentumSentinel, …)
│   ├── training/        Ensemble models, walk-forward trainer, drift detector, HPO
│   ├── alpha_composite.py  AlphaEngine (sector + options + regime multiplier)
│   ├── sector_alpha.py  SectorRanker (5d/20d RS vs Nifty 50)
│   ├── options_alpha.py OptionsFlowAnalyzer (OI + PCR)
│   └── wealth_architect_scanner.py  Conservative fundamental scanner
├── llm/                 FinBERT sentiment pipeline, Finnhub/NewsAPI fetchers
├── risk/                CircuitBreaker, PositionSizer, PortfolioMonitor
├── execution/           OrderExecutor (paper + live), trade logger
├── options/             Black-Scholes + IV surface model
├── backtest/            Walk-forward backtester, cost model, metrics
├── monitoring/          TelegramNotifier, daily/weekly/monthly reporters
├── audit/               Signal/trade/risk audit log (Redis + future TimescaleDB)
├── orchestrator/
│   ├── main.py          TradingSystem entry point
│   ├── scheduler.py     APScheduler job registration
│   ├── feature_engineer.py  Feature extraction + Redis caching (5-min TTL)
│   ├── ab_tester.py     Champion vs. challenger routing + reporting
│   └── runner.py        ScannerEngine worker pool
├── fundamentals/        Balance sheet / P&L fetchers for WealthArchitect cache
├── portfolio/           Holdings tracker, sector concentration checks
├── migrations/          Alembic schema migrations
├── tests/               Pytest unit tests (80%+ coverage, all mocked)
├── docs/                Architecture docs, env-variables reference
├── deploy/              Docker Compose, Nginx, Grafana, Prometheus configs
└── models/
    ├── ensemble/        Walk-forward trained XGB + LGB + PatchTST models
    └── production/      READ-ONLY — never overwrite production models
```

### Non-negotiable constraints

- `./models/production/` — read-only, never overwrite
- `.env` — never echo or log any value
- Broker API calls must always go through the rate-limiter wrapper
- All signals must pass `SignalValidator` before reaching the order executor
- Circuit-breaker halts are manual-reset only — no auto-recovery
- No bare `except:` — catch specific exceptions
- Parameterised queries always — no string-formatted SQL
