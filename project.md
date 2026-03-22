# NSE Trading System — Complete Architecture & Build Guide
>
> **For:** Devesh | **Capital:** ₹5L–₹25L | **Markets:** NSE Equities + F&O
> **Target:** 40%+ annual return | **Stack:** Python + Zerodha Kite + XGBoost + FinBERT
> **Built with Claude Code** — use this file as your persistent context across all sessions

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Module 1 — Data Pipeline](#4-module-1--data-pipeline)
5. [Module 2 — Feature Engineering](#5-module-2--feature-engineering)
6. [Module 3 — LLM Sentiment Layer](#6-module-3--llm-sentiment-layer)
7. [Module 4 — Signal Model (XGBoost)](#7-module-4--signal-model-xgboost)
8. [Module 5 — Risk Manager](#8-module-5--risk-manager)
9. [Module 6 — Execution Engine](#9-module-6--execution-engine)
10. [Module 7 — Backtesting Engine](#10-module-7--backtesting-engine)
11. [Module 8 — Orchestrator](#11-module-8--orchestrator)
12. [Module 9 — Monitoring & Observability](#12-module-9--monitoring--observability)
13. [Module 10 - NSE F&O — Options Model](#13-nse-fo--options-model)
14. [Module 11 - Walk-Forward Validation Protocol](#14-walk-forward-validation-protocol)
15. [Module 12 -Go-Live Checklist](#15-go-live-checklist)
16. [16-Week Build Roadmap](#16-16-week-build-roadmap)
17. [Performance Thresholds](#17-performance-thresholds)
18. [India-Specific Notes](#18-india-specific-notes)

---

## 1. System Overview

### Architecture in One Sentence

Live NSE market data → feature engineering + FinBERT sentiment → XGBoost signal → risk manager gate → Zerodha Kite execution → MLflow + Grafana observability.

### Data Flow

```text
Zerodha Kite WebSocket (live ticks)
        │
        ▼
TimescaleDB (OHLCV + tick storage)
        │
        ├──► Feature pipeline (TA-Lib: RSI, MACD, BB, ATR, vol regime)
        │
        ├──► FinBERT pipeline (news headlines → sentiment float −1 to +1)
        │
        ▼
XGBoost model → P(+2% in 5 days) score
        │
        ▼ P > 0.65
Risk Manager (circuit breakers + half-Kelly sizing)
        │
        ▼ approved
Kite order placement → MLflow log → Telegram alert
```

### Market Split

| Segment | Signal Type | Model | Label |
| --------- | ------------ | ------- | ------- |
| NSE Equities (NSE500) | Momentum + sentiment | XGBoost classifier | +2% in 5 days |
| NSE F&O (weekly expiry) | IV rank + delta-neutral | Separate XGBoost | IV reversion within 3 days |

### Non-Negotiable Constraints

- Max position size: 2% of total capital per trade
- Daily drawdown circuit breaker: 3%
- Weekly drawdown kill switch: 7%
- Walk-forward validation only — no in-sample backtesting
- Paper trade 200+ trades before live capital deployment
- Every trade logged: features, confidence score, outcome

---

## 2. Project Structure

```text
nse_trading_system/
│
├── data/
│   ├── __init__.py
│   ├── ingest.py            # Kite WebSocket + historical fetch
│   ├── store.py             # TimescaleDB read/write helpers
│   ├── clean.py             # Outlier removal, corporate action adjustments
│   └── universe.py          # NSE500 instrument list management
│
├── signals/
│   ├── __init__.py
│   ├── features.py          # TA-Lib feature engineering pipeline
│   ├── regime.py            # Volatility regime detector (HMM)
│   ├── model.py             # XGBoost inference wrapper
│   └── train.py             # Walk-forward training loop
│
├── llm/
│   ├── __init__.py
│   ├── sentiment.py         # FinBERT scoring engine
│   ├── pipeline.py          # News fetch → score → DB write
│   └── sources.py           # Finnhub + Moneycontrol RSS connectors
│
├── risk/
│   ├── __init__.py
│   ├── breakers.py          # Circuit breaker logic
│   ├── sizer.py             # Half-Kelly position sizing
│   └── monitor.py           # Real-time P&L + drawdown tracker
│
├── execution/
│   ├── __init__.py
│   ├── orders.py            # Kite order placement + cancellation
│   ├── paper.py             # Paper trading simulator
│   └── logger.py            # Trade log → MLflow
│
├── backtest/
│   ├── __init__.py
│   ├── walk_forward.py      # Rolling train/test validation engine
│   ├── costs.py             # Realistic cost model (brokerage + STT + slippage)
│   └── metrics.py           # Sharpe, drawdown, profit factor calculators
│
├── options/
│   ├── __init__.py
│   ├── iv_features.py       # IV rank, IV percentile, skew
│   ├── greeks.py            # Delta, gamma, theta calculations
│   └── strategy.py          # Delta-neutral entry logic
│
├── orchestrator/
│   ├── __init__.py
│   ├── main.py              # Main trading loop (market hours scheduler)
│   └── scheduler.py         # APScheduler jobs for pre/post market
│
├── monitoring/
│   ├── __init__.py
│   ├── mlflow_tracker.py    # Experiment + model tracking
│   ├── grafana_exporter.py  # Prometheus metrics exporter
│   └── alerts.py            # Telegram bot for circuit breaker alerts
│
├── config/
│   ├── settings.py          # Pydantic settings (env vars)
│   ├── instruments.json     # NSE500 token list
│   └── strategy_params.yaml # Model thresholds, risk params
│
├── tests/
│   ├── test_features.py
│   ├── test_risk.py
│   ├── test_execution.py
│   └── test_backtest.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_backtest_results.ipynb
│
├── docker-compose.yml       # TimescaleDB + Redis + Grafana
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 3. Environment Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose (for TimescaleDB, Redis, Grafana)
- Zerodha Kite Connect API subscription (₹2,000/month)
- Finnhub API key (free tier sufficient to start)

### `.env.example` — copy to `.env` and fill in

```env
# Zerodha Kite
KITE_API_KEY=your_api_key_here
KITE_API_SECRET=your_api_secret_here
KITE_ACCESS_TOKEN=             # refreshed daily via login flow

# Database
TIMESCALE_URL=postgresql://trader:password@localhost:5432/nse_trading
REDIS_URL=redis://localhost:6379/0

# News APIs
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key     # optional

# Monitoring
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
MLFLOW_TRACKING_URI=http://localhost:5000

# Strategy
SIGNAL_THRESHOLD=0.65
MAX_POSITION_PCT=0.02
DAILY_DD_LIMIT=0.03
WEEKLY_DD_LIMIT=0.07
PAPER_TRADE_MODE=true            # set false only after 200+ paper trades
```

### `docker-compose.yml`

```yaml
version: '3.8'
services:
  timescaledb:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: password
      POSTGRES_DB: nse_trading
    ports:
      - "5432:5432"
    volumes:
      - timescale_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    command: mlflow server --host 0.0.0.0

volumes:
  timescale_data:
  grafana_data:
```

### `requirements.txt`

```text
# Core
pandas==2.2.0
numpy==1.26.4
scipy==1.12.0
pydantic-settings==2.2.0
python-dotenv==1.0.0
pyyaml==6.0.1
apscheduler==3.10.4

# Market data
kiteconnect==5.0.1
finnhub-python==2.4.19
feedparser==6.0.11          # RSS for Moneycontrol

# Technical analysis
TA-Lib==0.4.28
pandas-ta==0.3.14b

# ML
lightgbm==4.3.0
xgboost==2.0.3
scikit-learn==1.4.1
hmmlearn==0.3.2             # regime detection

# NLP / LLM
transformers==4.39.0
torch==2.2.1
sentence-transformers==2.5.1

# Database
sqlalchemy==2.0.28
psycopg2-binary==2.9.9
redis==5.0.3
timescale-vector==0.0.4

# Monitoring
mlflow==2.11.1
prometheus-client==0.20.0
python-telegram-bot==21.1

# Backtesting
vectorbt==0.26.2

# Testing
pytest==8.1.0
pytest-asyncio==0.23.5
```

### Install commands

```bash
# Clone and setup
git clone <your-repo>
cd nse_trading_system

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

# Install TA-Lib (requires system library first)
# Ubuntu: sudo apt-get install -y libta-lib-dev
# Mac:    brew install ta-lib
pip install -r requirements.txt

# Start infrastructure
docker-compose up -d

# Initialize database schema (run once)
python -m data.store --init-schema

# Copy and fill env
cp .env.example .env
```

---

## 4. Module 1 — Data Pipeline

### `data/ingest.py`

**Purpose:** Pull OHLCV historical data + stream live ticks from Zerodha Kite into TimescaleDB.

**Build instructions for Claude Code:**

```text
Build data/ingest.py with:
- KiteIngestor class with methods:
  - fetch_historical(instrument_token, from_date, to_date, interval) → DataFrame
    - intervals: "minute", "3minute", "5minute", "15minute", "30minute", "60minute", "day"
    - writes to timescaledb ohlcv table
  - stream_live(tokens: list[int]) → starts KiteTicker WebSocket
    - on_ticks callback: writes to Redis cache (key: f"tick:{token}") with 1s TTL
    - on_connect: subscribe to all tokens
    - on_error: log + attempt reconnect with exponential backoff (max 5 retries)
  - refresh_access_token() → handles daily token refresh via Kite login URL
- All DB writes use bulk insert (executemany), not row-by-row
- Use structlog for all logging
- Connection pooling via SQLAlchemy pool_size=5
```

**Key schema — TimescaleDB `ohlcv` table:**

```sql
CREATE TABLE ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    token       INTEGER NOT NULL,
    symbol      TEXT NOT NULL,
    open        DOUBLE PRECISION,
    high        DOUBLE PRECISION,
    low         DOUBLE PRECISION,
    close       DOUBLE PRECISION,
    volume      BIGINT,
    interval    TEXT NOT NULL   -- 'day', '5minute', etc.
);
SELECT create_hypertable('ohlcv', 'time');
CREATE INDEX ON ohlcv (token, time DESC);
```

### `data/store.py`

**Purpose:** Read/write helpers for TimescaleDB and Redis.

**Build instructions for Claude Code:**

```text
Build data/store.py with:
- get_ohlcv(token, from_date, to_date, interval) → pd.DataFrame
- write_ohlcv(df: pd.DataFrame) → None (bulk upsert, conflict on time+token+interval)
- get_latest_tick(token) → dict from Redis cache
- get_universe(segment="EQ") → list of instrument dicts from instruments.json
- All functions use connection from a module-level SQLAlchemy engine
- Include a init_schema() function that runs the CREATE TABLE + hypertable SQL above
```

### `data/clean.py`

**Purpose:** Data quality — outlier removal, survivorship bias handling, corporate action adjustments.

**Build instructions for Claude Code:**

```text
Build data/clean.py with:
- remove_outliers(df, col='close', method='zscore', threshold=4.0) → df
- adjust_splits(df, token) → df (fetch corporate action from Kite and back-adjust close/volume)
- fill_missing_bars(df, interval) → df (forward fill max 3 bars, flag gaps > 3 as missing)
- validate_ohlcv(df) → (bool, list[str]) — checks: high>=low, close within high/low, volume>=0
- flag_circuit_limit_days(df) → df with 'circuit_hit' bool column
  - NSE circuit: abs(pct_change) > 19.9% on daily bars
```

### `data/universe.py`

**Purpose:** Manage the NSE500 instrument universe.

**Build instructions for Claude Code:**

```text
Build data/universe.py with:
- load_nse500_tokens() → list[dict] — reads config/instruments.json
- refresh_instruments(kite) → writes updated instruments.json from Kite instrument dump
- get_fo_instruments(kite) → list of active F&O instruments with lot sizes
- filter_liquid(df_universe, min_avg_volume=500000) → filtered list
  - liquidity filter: 20-day avg volume > 500,000 shares
```

---

## 5. Module 2 — Feature Engineering

### `signals/features.py`

**Purpose:** Convert raw OHLCV into the feature vector XGBoost will consume.

**Build instructions for Claude Code:**

```text
Build signals/features.py with a single function:
build_features(df: pd.DataFrame) → pd.DataFrame

Features to engineer (all using TA-Lib unless noted):
MOMENTUM
  - rsi_14 = RSI(close, 14)
  - rsi_7 = RSI(close, 7)
  - macd, macd_signal, macd_hist = MACD(close, 12, 26, 9)
  - macd_cross = sign(macd - macd_signal)  # +1 bullish cross, -1 bearish
  - mom_10 = MOM(close, 10)
  - roc_5 = ROC(close, 5)
  - roc_21 = ROC(close, 21)

VOLATILITY
  - atr_14 = ATR(high, low, close, 14)
  - atr_pct = atr_14 / close  # normalized
  - bb_upper, bb_mid, bb_lower = BBANDS(close, 20, 2.0)
  - bb_position = (close - bb_lower) / (bb_upper - bb_lower)  # 0=lower band, 1=upper
  - realized_vol_10 = close.pct_change().rolling(10).std() * sqrt(252)
  - realized_vol_20 = close.pct_change().rolling(20).std() * sqrt(252)

VOLUME
  - volume_zscore_20 = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
  - obv = OBV(close, volume)
  - obv_slope = obv.diff(5) / 5  # 5-day OBV trend
  - vwap_dev = (close - vwap) / vwap  # deviation from daily VWAP

TREND
  - ema_9 = EMA(close, 9)
  - ema_21 = EMA(close, 21)
  - ema_50 = EMA(close, 50)
  - ema_cross_9_21 = sign(ema_9 - ema_21)
  - price_vs_ema50 = (close - ema_50) / ema_50
  - adx_14 = ADX(high, low, close, 14)  # trend strength
  - di_plus = PLUS_DI(high, low, close, 14)
  - di_minus = MINUS_DI(high, low, close, 14)

MEAN REVERSION
  - zscore_20 = (close - close.rolling(20).mean()) / close.rolling(20).std()
  - distance_from_52w_high = (close - high.rolling(252).max()) / high.rolling(252).max()
  - distance_from_52w_low = (close - low.rolling(252).min()) / low.rolling(252).min()

REGIME (from regime.py)
  - vol_regime = 1 if realized_vol_20 > realized_vol_20.rolling(60).median() else 0

LABEL (for training only)
  - forward_return_5d = close.shift(-5) / close - 1
  - label = (forward_return_5d > 0.02).astype(int)  # 1 = +2% in 5 days

Important:
- Drop rows with NaN (first 60 rows typically)
- Do NOT include forward_return_5d or label in inference features
- Return feature df with DatetimeIndex preserved
```

### `signals/regime.py`

**Purpose:** Classify current market regime (trending vs mean-reverting vs high-vol).

**Build instructions for Claude Code:**

```text
Build signals/regime.py with:
- VolRegimeDetector class using hmmlearn GaussianHMM with 2 states
  - fit(returns_series) → trains 2-state HMM on log returns
  - predict(returns_series) → array of 0/1 regime labels
  - label_regimes(df) → df with 'regime' column and 'vol_regime_hmm' column
- SimpleVolRegime class (fallback, no ML):
  - classify(df, lookback=20, multiplier=1.5) → 'high_vol' | 'low_vol' | 'normal'
  - uses realized_vol vs rolling median comparison
- Both classes: save/load state with joblib
```

---

## 6. Module 3 — LLM Sentiment Layer

### `llm/sentiment.py`

**Purpose:** Run FinBERT on news headlines to produce a sentiment score per instrument.

**Build instructions for Claude Code:**

```text
Build llm/sentiment.py with:
- FinBERTScorer class:
  - __init__: load ProsusAI/finbert from HuggingFace (cache locally in .models/)
  - score(texts: list[str]) → list[float]
    - each float: positive_score - negative_score (range −1 to +1)
    - batch size: 32 for GPU, 8 for CPU
    - truncate to 512 tokens
  - score_aggregate(texts: list[str], method='mean') → float
    - aggregate multiple headlines to single score
    - methods: 'mean', 'weighted_recency' (more recent = higher weight)
  - device detection: use CUDA if available, else MPS (Mac), else CPU
  - cache scores in Redis with key f"sentiment:{symbol}:{date}" TTL=3600
```

### `llm/sources.py`

**Purpose:** Fetch news from Finnhub and Moneycontrol RSS.

**Build instructions for Claude Code:**

```text
Build llm/sources.py with:
- FinnhubFetcher class:
  - fetch_news(symbol: str, from_ts: int, to_ts: int) → list[dict]
    - each dict: {headline, summary, datetime, source, url}
  - fetch_market_news(category='general') → list[dict]
  - rate limit: max 30 calls/min (add sleep if needed)

- MoneycontrolRSS class:
  - FEEDS dict mapping sector → RSS URL
    - 'markets': 'https://www.moneycontrol.com/rss/MCtopnews.xml'
    - 'economy': 'https://www.moneycontrol.com/rss/economy.xml'
    - 'companies': 'https://www.moneycontrol.com/rss/business.xml'
  - fetch(feed_name: str, max_items=20) → list[dict]
    - parse with feedparser
  - deduplicate by URL before returning

- merge_and_rank(sources: list[list[dict]], hours_lookback=24) → list[dict]
  - merge all sources, filter to last N hours, sort by datetime desc
```

### `llm/pipeline.py`

**Purpose:** Orchestrate fetch → score → store for all instruments in the universe.

**Build instructions for Claude Code:**

```text
Build llm/pipeline.py with:
- SentimentPipeline class:
  - run_daily(universe: list[str]) → dict[symbol, float]
    - for each symbol: fetch news last 24h, score, store to DB
    - store to timescaledb table 'sentiment_scores':
        (time TIMESTAMPTZ, symbol TEXT, score FLOAT, headline_count INT)
  - get_latest_score(symbol: str) → float (from Redis cache or DB)
  - run_continuous(universe, interval_minutes=30) → async loop
    - refresh scores every 30 minutes during market hours (9:15–15:30 IST)

DB schema:
CREATE TABLE sentiment_scores (
    time         TIMESTAMPTZ NOT NULL,
    symbol       TEXT NOT NULL,
    score        DOUBLE PRECISION,
    headline_count INT
);
SELECT create_hypertable('sentiment_scores', 'time');
```

---

## 7. Module 4 — Signal Model (XGBoost)

### `signals/model.py`

**Purpose:** XGBoost inference wrapper — load trained model, run prediction.

**Build instructions for Claude Code:**

```text
Build signals/model.py with:
- SignalModel class:
  - __init__(model_path: str): load XGBoost model from mlflow or local path
  - predict(features: pd.DataFrame) → pd.Series of probabilities (0–1)
  - predict_single(feature_row: dict) → float
  - feature_names: list of expected feature columns (validated on load)
  - explain(feature_row: dict) → dict of SHAP values (top 10 features)
    - use shap library TreeExplainer
  - is_healthy() → bool (checks model loaded, feature schema matches)

- ModelRegistry class:
  - get_latest_model(segment='EQ') → SignalModel
    - fetches from MLflow model registry, stage='Production'
  - register_model(run_id, segment) → registers to MLflow
```

### `signals/train.py`

**Purpose:** Walk-forward training loop.

**Build instructions for Claude Code:**

```text
Build signals/train.py with:
- WalkForwardTrainer class:
  - __init__(train_months=24, test_months=3, purge_days=5)
  - run(df: pd.DataFrame, features: list[str], label: str) → dict of results per fold
    - split data temporally: no random shuffle ever
    - apply purge_days gap between train end and test start (avoid leakage at boundary)
    - for each fold:
        - train XGBoost with these params:
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, eval_metric='auc',
            early_stopping_rounds=50
        - log to MLflow: fold metrics (AUC, precision, recall, Sharpe of signal)
        - save model artifact per fold
  - best_model() → XGBoost model from highest out-of-sample AUC fold
  - print_summary() → table of fold results + aggregate metrics

XGBoost hyperparams to tune (use Optuna):
  - max_depth: [4, 8]
  - learning_rate: [0.01, 0.1]
  - n_estimators: [200, 1000]
  - min_child_weight: [20, 100]  # critical for financial data — prevents overfitting
  - subsample: [0.6, 0.9]
```

---

## 8. Module 5 — Risk Manager

### `risk/breakers.py`

**Purpose:** Circuit breakers that halt trading when drawdown limits are breached.

**Build instructions for Claude Code:**

```text
Build risk/breakers.py with:
- CircuitBreaker class:
  - __init__(daily_limit=0.03, weekly_limit=0.07, max_consecutive_losses=5)
  - check(current_capital: float) → (allowed: bool, reason: str | None)
    - checks: daily drawdown > 3%, weekly drawdown > 7%, consecutive losses > 5
  - reset_daily() → called at 9:15 IST each morning
  - reset_weekly() → called Monday morning
  - is_halted() → bool
  - halt(reason: str) → sets halted flag + sends Telegram alert
  - state stored in Redis (survives process restarts):
      key: "circuit:state" → JSON {halted, reason, daily_start_capital, peak_capital, consecutive_losses}

- critical: if halted, EVERY call to check() must return (False, reason)
  - only manual reset (via CLI command) can clear halt
  - never auto-reset from a halt caused by drawdown breach
```

### `risk/sizer.py`

**Purpose:** Half-Kelly position sizing with volatility scaling.

**Build instructions for Claude Code:**

```text
Build risk/sizer.py with:
- PositionSizer class:
  - __init__(total_capital: float, max_position_pct=0.02)
  - size(signal_probability: float, asset_volatility: float,
         current_capital: float) → rupee_amount: float

    Formula:
      edge = signal_probability - 0.5        # excess probability over coin flip
      kelly_fraction = edge / (1 - edge)     # full Kelly
      half_kelly = kelly_fraction * 0.5      # use half-Kelly always
      vol_scalar = min(1.0, 0.20 / max(asset_volatility, 0.05))  # scale down in high vol
      base_size = current_capital * max_position_pct
      final_size = base_size * min(half_kelly, 1.0) * vol_scalar
      return round(final_size, 2)

  - shares(rupee_amount: float, current_price: float, lot_size=1) → int
    - for equities: floor(rupee_amount / current_price)
    - for F&O: floor(rupee_amount / (current_price * lot_size)) * lot_size
  - max_allowed(current_capital) → float (hard cap: 2% of capital)
```

### `risk/monitor.py`

**Purpose:** Real-time P&L and drawdown tracking.

**Build instructions for Claude Code:**

```text
Build risk/monitor.py with:
- PortfolioMonitor class:
  - update_position(symbol, qty, avg_price, current_price) → None
  - get_pnl() → dict {realized, unrealized, total, pct_return}
  - get_drawdown() → dict {daily_dd, weekly_dd, max_dd, current_dd}
  - get_exposure() → dict {gross_exposure, net_exposure, largest_position_pct}
  - export_prometheus_metrics() → for Grafana dashboard
  - state: stored in Redis, updated every tick
  - alert thresholds: warn at 2% daily DD, halt at 3%
```

---

## 9. Module 6 — Execution Engine

### `execution/orders.py`

**Purpose:** Place, modify, and cancel orders via Kite Connect.

**Build instructions for Claude Code:**

```text
Build execution/orders.py with:
- OrderExecutor class:
  - __init__(kite, paper_mode=True)  # ALWAYS default to paper mode

  - place_market_order(symbol, transaction_type, quantity, tag) → order_id
    - transaction_type: 'BUY' | 'SELL'
    - uses PRODUCT_MIS for intraday, PRODUCT_CNC for delivery
    - validates: circuit breaker not halted, quantity > 0, symbol in universe
    - logs to MLflow before placing

  - place_limit_order(symbol, transaction_type, quantity, price, tag) → order_id
    - prefer limit orders for sizes > ₹50,000 (reduce slippage)

  - cancel_order(order_id) → bool

  - get_order_status(order_id) → dict

  - slippage_estimate(symbol, quantity, side) → float
    - estimate: 0.05% liquid large-cap, 0.15% mid-cap, 0.20% small-cap
    - based on avg volume vs order size ratio

  - paper mode: all methods log the "trade" to paper_trades table in DB
    but do NOT call Kite API

DB schema for paper trades:
CREATE TABLE paper_trades (
    id          SERIAL PRIMARY KEY,
    time        TIMESTAMPTZ DEFAULT NOW(),
    symbol      TEXT,
    side        TEXT,
    quantity    INT,
    price       DOUBLE PRECISION,
    signal_prob DOUBLE PRECISION,
    position_size_inr DOUBLE PRECISION,
    tag         TEXT
);
```

### `execution/logger.py`

**Purpose:** Log every trade decision to MLflow for audit trail.

**Build instructions for Claude Code:**

```text
Build execution/logger.py with:
- TradeLogger class:
  - log_signal(symbol, features_dict, signal_prob, action_taken) → run_id
    - logs to MLflow as a run with:
        params: symbol, date, signal_prob, action
        metrics: all feature values at time of decision
        tags: strategy_version, regime

  - log_outcome(run_id, exit_price, exit_date, pnl_pct) → None
    - updates the original run with realized outcome

  - log_circuit_breaker_event(reason, capital_at_halt) → None

  - daily_summary() → dict
    - trades today, win rate, P&L, Sharpe of today's signals
    - push to Telegram via alerts.py
```

---

## 10. Module 7 — Backtesting Engine

### `backtest/walk_forward.py`

**Purpose:** Walk-forward validation — the ONLY valid backtesting method.

**Build instructions for Claude Code:**

```text
Build backtest/walk_forward.py with:
- WalkForwardBacktest class:
  - __init__(train_months=24, test_months=3, purge_days=5)
  - run(df: pd.DataFrame) → BacktestResults
    - CRITICAL: never allow future data to leak into training features
    - for each fold:
        1. slice train window
        2. apply purge_days gap
        3. slice test window
        4. train model on train window
        5. generate signals on test window
        6. apply cost model (see costs.py)
        7. calculate metrics
    - return all fold results + aggregate

- BacktestResults dataclass:
  - fold_results: list[FoldResult]
  - aggregate_sharpe: float
  - aggregate_max_dd: float
  - aggregate_profit_factor: float
  - win_rate: float
  - total_trades: int
  - plot() → matplotlib figure (equity curve + drawdown subplot)
```

### `backtest/costs.py`

**Purpose:** Realistic cost model for NSE — critical for not fooling yourself.

**Build instructions for Claude Code:**

```text
Build backtest/costs.py with:
- NSECostModel class:
  - equity_cost(trade_value: float, side: str, intraday=False) → float
    Breakdown per trade:
      brokerage = min(20, trade_value * 0.0003)   # Zerodha: ₹20 or 0.03%
      stt = trade_value * 0.001 if side=='SELL' else 0   # STT on sell side equity
      exchange_fee = trade_value * 0.0000345       # NSE fees
      sebi_fee = trade_value * 0.000001            # SEBI charges
      gst = (brokerage + exchange_fee) * 0.18     # 18% GST on brokerage+fees
      stamp_duty = trade_value * 0.00015 if side=='BUY' else 0
      total_cost = brokerage + stt + exchange_fee + sebi_fee + gst + stamp_duty
      return total_cost

  - slippage(trade_value: float, liquidity_tier: str) → float
    - 'large_cap': 0.05% of trade_value
    - 'mid_cap': 0.12%
    - 'small_cap': 0.20%

  - round_trip_cost(trade_value, liquidity_tier='large_cap') → float
    - buy_cost + sell_cost + buy_slippage + sell_slippage
    - for ₹10,000 trade large_cap: approx ₹55–75 (0.55–0.75%)

  - CRITICAL NOTE: a strategy needs >0.75% per trade edge just to break even
    on transaction costs. Build this into all signal threshold decisions.
```

### `backtest/metrics.py`

**Purpose:** Calculate all performance metrics.

**Build instructions for Claude Code:**

```text
Build backtest/metrics.py with standalone functions:
- sharpe_ratio(returns: pd.Series, risk_free=0.065) → float
  - annualized, using 252 trading days
  - risk_free default: 6.5% (India 10Y G-Sec approx)

- max_drawdown(equity_curve: pd.Series) → float
  - returns max peak-to-trough as negative fraction

- profit_factor(returns: pd.Series) → float
  - gross_profit / gross_loss (absolute value)

- calmar_ratio(returns, max_dd) → float
  - annualized_return / abs(max_dd)

- win_rate(returns: pd.Series) → float

- expectancy(returns: pd.Series) → float
  - average win * win_rate - average loss * loss_rate
  - must be positive for viable strategy

- print_tearsheet(returns: pd.Series, equity_curve: pd.Series) → None
  - prints formatted table of all metrics
  - flags PASS/FAIL against minimum thresholds (see section 17)
```

---

## 11. Module 8 — Orchestrator

### `orchestrator/main.py`

**Purpose:** Main trading loop — ties all modules together.

**Build instructions for Claude Code:**

```text
Build orchestrator/main.py with:
- TradingSystem class:
  - __init__: initialize all modules (data, signals, llm, risk, execution)
  - start() → main async loop

  Pre-market (8:45–9:14 IST):
    1. refresh_access_token()
    2. run sentiment pipeline for full universe
    3. load latest model from MLflow registry
    4. reset daily circuit breaker
    5. log system health check to Telegram

  Market hours (9:15–15:25 IST), every 5 minutes:
    1. fetch latest OHLCV from Kite (last 60 bars)
    2. build features for each instrument in universe (top 50 liquid)
    3. append latest sentiment score per instrument
    4. run XGBoost inference → probability scores
    5. filter: P > SIGNAL_THRESHOLD (default 0.65)
    6. for each signal above threshold:
        a. check circuit breaker (skip if halted)
        b. calculate position size via Kelly sizer
        c. check: not already holding this symbol
        d. place order (paper or live based on PAPER_TRADE_MODE)
        e. log signal + decision to MLflow
    7. update portfolio monitor with current prices
    8. check drawdown → trigger circuit breaker if needed

  Post-market (15:35 IST):
    1. generate daily summary (trades, P&L, Sharpe)
    2. send Telegram report
    3. check model drift (win rate vs baseline)
    4. retrain trigger: if rolling 20-trade win rate < 50%, flag for retraining

- Graceful shutdown: on SIGTERM/SIGINT → cancel open orders → save state → exit

- main() entry point: python -m orchestrator.main
```

### `orchestrator/scheduler.py`

**Build instructions for Claude Code:**

```text
Build orchestrator/scheduler.py with APScheduler:
- Jobs:
  - 08:45 IST daily → pre_market_setup()
  - 09:15–15:25 every 5 min → trading_loop()
  - 15:35 IST daily → post_market_summary()
  - Monday 08:30 → reset_weekly_circuit_breaker()
  - Sunday 02:00 → retrain_check() (compare last week's live vs backtest Sharpe)
- Use Asia/Kolkata timezone throughout
- All jobs: try/except with Telegram alert on failure
```

---

## 12. Module 9 — Monitoring & Observability

### `monitoring/alerts.py`

**Build instructions for Claude Code:**

```text
Build monitoring/alerts.py with:
- TelegramAlerter class:
  - send(message: str) → None (async, with retry)
  - alert_circuit_breaker(reason, dd_pct, capital) → formatted alert
  - alert_daily_summary(trades, pnl_pct, sharpe, win_rate) → formatted summary
  - alert_model_drift(current_win_rate, baseline_win_rate) → formatted warning
  - alert_system_error(module, error_msg) → formatted error

Message format example:
  🔴 CIRCUIT BREAKER TRIGGERED
  Reason: Daily drawdown exceeded 3%
  DD: −3.21% | Capital: ₹4,83,200
  All trading halted. Manual reset required.
  Time: 11:34 IST
```

### `monitoring/mlflow_tracker.py`

**Build instructions for Claude Code:**

```text
Build monitoring/mlflow_tracker.py with:
- Experiment names: "nse_equity_signals", "nse_options_signals", "backtest_results"
- ModelRegistry: register trained models with stage Staging → Production workflow
- ModelDriftMonitor class:
  - compare_live_vs_backtest(window_trades=20) → drift_score: float
  - if drift_score > 0.3: trigger alert + flag for retraining
  - metrics tracked: rolling win_rate, rolling Sharpe, signal_probability_calibration
```

---

## 13. NSE F&O — Options Model

### `options/iv_features.py`

**Purpose:** Build IV-based features for options strategies.

**Build instructions for Claude Code:**

```text
Build options/iv_features.py with:
- build_fo_features(symbol, expiry_date, kite) → pd.DataFrame
  Features:
  - iv_rank = (current_IV - 52w_low_IV) / (52w_high_IV - 52w_low_IV)  # 0–1
  - iv_percentile = percentile rank of current IV over 252 days
  - iv_premium = implied_vol - realized_vol_20d  # positive = IV elevated
  - put_call_ratio = put_OI / call_OI  # from NSE F&O OI data
  - max_pain = strike with max OI across calls+puts (options pinning target)
  - days_to_expiry = (expiry_date - today).days

Signal logic (implement in options/strategy.py):
  - if iv_rank > 0.7 AND iv_premium > 0.05: SELL premium (iron condor or straddle)
  - if iv_rank < 0.3: BUY premium (long straddle before expected event)
  - max position: 1 lot per signal, max 3 concurrent F&O positions
  - always delta-hedge: buy/sell underlying to flatten delta within ±0.10
```

---

## 14. Walk-Forward Validation Protocol

**This section is the most important in the document. Do not skip or abbreviate.**

### Rules (Non-Negotiable)

1. Data split is always temporal — never random
2. Purge gap of 5 trading days between train end and test start (avoids leakage)
3. Minimum 5 folds before trusting aggregate metrics
4. Out-of-sample test period: 3 months per fold
5. Train period: rolling 24 months (not expanding)

### Stress Test Windows (Always Include)

| Event | Date Range |
| ------- | ----------- |
| COVID crash | Mar 2020 – May 2020 |
| Recovery rally | Jun 2020 – Dec 2020 |
| FTX/crypto contagion | Nov 2022 |
| India rate hike cycle | May 2022 – Feb 2023 |
| Liberation Day tariff shock | Apr 2025 |
| India demonetization replay | Nov 2016 (if using historical data) |

### Minimum Thresholds to Pass Before Live Deployment

| Metric | Minimum | Good | Exceptional |
| -------- | --------- | ------ | ------------- |
| Sharpe Ratio | 1.0 | 1.5 | 2.0+ |
| Max Drawdown | ≤25% | ≤15% | ≤10% |
| Profit Factor | 1.5 | 2.0 | 4.0+ |
| Win Rate | 52% | 58% | 65%+ |
| Calmar Ratio | 0.5 | 1.0 | 2.0+ |
| Min Trades (backtest) | 100 | 200 | 500+ |

**If any metric is below minimum: do not proceed. Fix the feature engineering.**

---

## 15. Go-Live Checklist

Run through this checklist before switching `PAPER_TRADE_MODE=false`.

```text
PRE-LIVE CHECKLIST

Data pipeline
  [ ] 2+ years of OHLCV loaded for NSE500 universe into TimescaleDB
  [ ] Corporate action adjustments applied and verified
  [ ] Live tick streaming tested for 5+ trading days without errors
  [ ] Data quality checks passing (no outliers, no missing bars > 3 consecutive)

Signal model
  [ ] Walk-forward backtest completed across 5+ folds
  [ ] Aggregate Sharpe ≥ 1.5 (after costs)
  [ ] Max drawdown ≤ 15% (after costs)
  [ ] Profit factor ≥ 2.0
  [ ] Model registered in MLflow as 'Production'
  [ ] Feature schema validated: inference features match training features exactly

LLM sentiment
  [ ] FinBERT running locally (not via API — latency and cost)
  [ ] Sentiment scores populating for 50+ instruments daily
  [ ] Redis cache hit rate > 80%

Risk manager
  [ ] Circuit breaker state persisted in Redis
  [ ] Telegram alert received on test halt
  [ ] Daily reset tested at 9:15 IST
  [ ] Position sizer tested with min and max capital scenarios

Execution
  [ ] Paper trading completed: 200+ trades minimum
  [ ] Paper trade Sharpe within 30% of backtest Sharpe
  [ ] Kite API connectivity tested: place + cancel order cycle
  [ ] Order logger writing to MLflow for each paper trade

Monitoring
  [ ] Grafana dashboard live with: P&L, drawdown, win rate, position count
  [ ] Telegram daily summary working
  [ ] MLflow experiment tracking all paper trade runs
  [ ] System error alerts tested

SEBI compliance
  [ ] Algo trading registered under Zerodha's Category 3 framework
  [ ] All orders tagged with strategy tag in Kite API calls
  [ ] Audit log retained for 5 years (MLflow covers this)
```

---

## 16. 16-Week Build Roadmap

| Week | Focus | Deliverable |
| ------ | ------- | ------------- |
| 1 | Environment + DB | Docker up, TimescaleDB schema, instruments.json |
| 2 | data/ingest.py | Historical OHLCV for NSE500, 2 years loaded |
| 3 | data/clean.py | Outlier removal, corporate action adjustments verified |
| 4 | Live streaming | KiteTicker running, ticks to Redis |
| 5 | signals/features.py | Full feature vector, unit tested |
| 6 | signals/regime.py | HMM regime detector trained and validated |
| 7 | llm/sentiment.py | FinBERT scoring 50+ instruments daily |
| 8 | signals/train.py | First walk-forward training run, baseline Sharpe logged |
| 9 | backtest/ | Full backtest engine with cost model, tearsheet output |
| 10 | Iterate on features | Beat Sharpe 1.5 threshold — iterate features, not model |
| 11 | risk/ | All circuit breakers, sizer, monitor — fully tested |
| 12 | execution/paper.py | Paper trading mode live, Telegram alerts working |
| 13 | 200 paper trades | Run 200+ trades, analyze slippage vs backtest |
| 14 | orchestrator/ | Full system loop running market hours without errors |
| 15 | Monitoring | Grafana live, MLflow tracking, daily summaries |
| 16 | Go-live | ₹50,000 live capital, scale only after 60 live trades confirm Sharpe |

---

## 17. Performance Thresholds

### Live Monitoring Rules

- If rolling 20-trade win rate drops from 58% to <51%: pause and investigate
- If live Sharpe (60 trade rolling) diverges >30% from backtest Sharpe: halt and diagnose
- If any single day P&L > +8%: review for data error or model anomaly (wins can indicate bugs)
- Monthly review: retrain model if market regime has shifted (check HMM state distribution)

### Scaling Rules (₹5L–₹25L)

| Live Trade Count | Max Capital Deployed |
| ----------------- | --------------------- |
| 0–60 trades | ₹50,000 |
| 60–120 trades | ₹2,00,000 |
| 120–200 trades | ₹5,00,000 |
| 200+ trades (Sharpe ≥ 1.5 confirmed) | ₹10,00,000+ |

---

## 18. India-Specific Notes

### SEBI Algo Trading Compliance

- Category 3 algo: fully automated, no manual intervention per trade
- Must register algo with broker (Zerodha has self-serve registration)
- All orders must have unique algo tag in order placement
- Audit trail mandatory — MLflow logging covers this requirement
- Reference: SEBI circular SEBI/HO/MRD2/DCAP/CIR/P/2021/589

### NSE Market Hours

```text
Pre-open session:    09:00–09:08 IST
Pre-open matching:   09:08–09:15 IST
Normal market:       09:15–15:30 IST
Closing session:     15:40–16:00 IST
F&O expiry:          Thursday (weekly), last Thursday of month (monthly)
```

### NSE Circuit Breakers (Market-Wide)

- 10% index drop: 45-minute trading halt
- 15% index drop: 1-hour 45-minute halt (2-hour halt after 2pm)
- 20% index drop: trading closed for the day
- Build awareness of these into circuit breaker logic — don't fight market-wide halts

### Zerodha Kite API Limits

- Historical data: 60 days per request for minute data, no limit for day data
- Order rate limit: 10 orders/second (sufficient for this system)
- WebSocket: max 3,000 instrument tokens per connection
- Access token: expires daily at 6:00 AM IST — must refresh

### Tax Implications (India)

- STCG (equity < 1 year): 15% on gains
- Intraday equity: taxed as business income (slab rate)
- F&O: always taxed as business income regardless of holding period
- Maintain trade-wise P&L log for ITR-3 filing — MLflow covers this

---

## Claude Code Session Prompts

Use these to continue work efficiently in new Claude Code sessions:

```text
# Start a new module
"Read NSE_TRADING_SYSTEM.md and build [module name] exactly as specified
in the Build instructions section. Use the tech stack defined in
requirements.txt. Write unit tests in tests/."

# Debug a module
"Read NSE_TRADING_SYSTEM.md section [X]. Here is the current code for
[module]. It has this error: [error]. Fix it while maintaining the
architecture spec."

# Run backtest
"Read NSE_TRADING_SYSTEM.md section 14 (Walk-Forward Validation Protocol).
Run the backtest engine against the data in TimescaleDB for the NSE500
universe. Print a full tearsheet and flag any metrics below minimum threshold."

# Add a feature
"Read NSE_TRADING_SYSTEM.md signals/features.py spec. Add [new feature]
to the feature pipeline. Ensure it: (1) has no look-ahead bias, (2) is
normalized, (3) is covered by a unit test in tests/test_features.py."
```

---

*Architecture version 1.0 — March 2026*
*Review and update after each module completion*
*Do not deploy live capital until Go-Live Checklist is fully checked*
