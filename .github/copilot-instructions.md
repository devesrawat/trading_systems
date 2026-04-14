# Copilot Instructions — NSE/Crypto Algorithmic Trading System

## Build, Test, and Lint

All commands use `uv run`. The project uses Python 3.13.

```bash
# Install dependencies
uv sync --frozen --dev

# Apply DB schema migrations (required before first test run)
uv run alembic upgrade head

# Run all tests (80% coverage required)
uv run pytest tests/ -q

# Run a single test file
uv run pytest tests/test_signals_features.py -x -q

# Run a single test by name
uv run pytest tests/test_signals_features.py::test_build_features_shape -x -q

# Lint and auto-fix
uv run ruff check . --fix

# Format check
uv run ruff format --check .

# Type check
uv run mypy . --strict
```

The orchestrator entry point:
```bash
python -m orchestrator.main                  # uses MARKET_TYPE from .env
python -m orchestrator.main --market equity
python -m orchestrator.main --market crypto
python -m orchestrator.main --market both
```

## Architecture

```
Zerodha Kite WebSocket (live ticks)
        │
        ▼
TimescaleDB (OHLCV) + Redis (tick cache, risk state, sentiment cache)
        │
        ├──► signals/features.py  — TA indicators → FEATURE_COLUMNS
        ├──► llm/sentiment.py     — FinBERT (ProsusAI/finbert) → [-1, +1] score
        ▼
signals/model.py — XGBoost → P(+2% in 5 days)
        │
        ▼ score > 0.65
risk/breakers.py (CircuitBreaker) + risk/sizer.py (half-Kelly sizing)
        │
        ▼ approved
execution/orders.py → OrderExecutor → BrokerAdapter
        │
        ▼
MLflow (trade log) + Telegram (alert)
```

**Market modes** (`MARKET_TYPE` in `.env`): `equity` (NSE via Kite/Upstox), `crypto` (Binance), or `both`.

**Data provider** (`DATA_PROVIDER` in `.env`): `kite` (default), `upstox`, or `binance`. Provider is injected via `data/providers/base.py` — swapping providers doesn't require touching other modules.

**Schema** is managed with Alembic (`migrations/`). Run `alembic upgrade head` after pulling.

**FinBERT model** is cached locally in `.models/` (auto-downloaded on first run). Never committed to git.

## Key Conventions

### Signal pipeline
- All scan strategies extend `signals/base_strategy.py::BaseStrategy`. Set class attributes `name`, `lookback_days`, `interval`, `min_bars`. Implement `scan(symbol, df) → dict | None`.
- `scan()` **must be pure CPU** — no DB, Redis, or network calls. It runs inside a `ProcessPoolExecutor`.
- `FEATURE_COLUMNS` in `signals/features.py` is the canonical feature schema. The trained XGBoost model is coupled to this exact list — never add/remove/rename features without retraining.
- `signals/model.py::SignalModel` validates the feature schema against `FEATURE_COLUMNS` at load time.
- Signal threshold is `0.65` (configurable in `config/strategy_params.yaml`).

### Risk controls (non-negotiable)
- `max_position_pct` is hard-capped at 2% via a Pydantic validator — the validator raises if you try to exceed it.
- `daily_dd_limit` is hard-capped at 5%. Default is 3%.
- `CircuitBreaker.check()` must be called before every order. Drawdown halts **never auto-reset** — only `manual_reset()` clears them. State is persisted in Redis and survives restarts.
- `paper_trade_mode=True` is the default. Live trading requires completing the 200+ paper-trade checklist.

### Broker and API calls
- **Every** external API call must go through `data/rate_limiter.py::RateLimiter` as a context manager: `with limiter: resp = api.call(...)`. Never call broker APIs bare.
- All Redis keys are defined in `data/redis_keys.py::RedisKeys`. Use the helpers there — never hardcode key strings.
- `data/store.py::get_engine()` and `get_redis()` are process-level singletons. Don't create new engines/clients.

### Logging and secrets
- Use `structlog.get_logger(__name__)` for all logging. Never use `print()` or the stdlib `logging` module directly.
- Never echo or log `.env` contents. Never hardcode the Telegram bot token or any API credentials.

### Read-only paths
- `./models/production/` — never overwrite these files. Use `ModelRegistry` to promote a new model.

## Known Gotchas

- **Kite WebSocket disconnects silently.** Always wrap live feed connections with a reconnect loop (`data/live_feed.py` shows the pattern).
- **NSE bhavcopy URL changes on expiry days.** Use `data/ingest.py`'s adaptive fetcher — don't hardcode the bhavcopy URL.
- **TimescaleDB hypertable** requires the time dimension column in every query's `WHERE` clause or the planner won't use the hypertable partition index.
- **`conftest.py` injects stub env vars** so `config/settings.py` can be imported in tests without a real `.env`. If you add a required `Settings` field, add a corresponding stub in `tests/conftest.py`.
