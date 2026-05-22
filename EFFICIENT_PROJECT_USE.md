# Efficient Use of Trading System Project

## Project Overview

**NSE/Crypto Algorithmic Trading System** — A sophisticated Python-based trading platform for Indian equities (NSE) and cryptocurrencies (Binance). The system combines:
- Real-time market data (Zerodha Kite WebSocket + live feeds)
- ML-driven signal generation (XGBoost + LightGBM ensemble + PatchTST)
- Sentiment analysis (FinBERT)
- Advanced risk management (circuit breakers, half-Kelly sizing, sector limits)
- Execution layer (order management, slippage tracking)
- Comprehensive monitoring (MLflow, Grafana, Telegram alerts)
- Walk-forward backtesting with concept drift detection

**Target**: 40%+ annual return on ₹5L–₹25L capital with <5% daily drawdown limits.

**Current Phase**: Phase 9 (ML Pipeline & Orchestration Quick Wins) — Ready to deploy.

---

## How to Work with This Project Efficiently

### 1. **Environment Setup (5 minutes)**

```bash
# Install dependencies (Python 3.13)
uv sync --frozen --dev

# Apply database schema
uv run alembic upgrade head

# Set up pre-commit hooks (auto-linting on commit)
uv run pre-commit install
```

**Key files to understand first**:
- `.env.example` — Template for environment variables (copy to `.env`)
- `config/settings.py` — Configuration schema (Pydantic)
- `config/strategy_params.yaml` — Strategy configuration

---

### 2. **Key Development Workflows**

#### A. **Running Tests** (Baseline: 1100+ tests, 80%+ coverage required)
```bash
# All tests
uv run pytest tests/ -q

# Single test file
uv run pytest tests/test_signals_features.py -x -q

# Single test by name
uv run pytest tests/test_signals_features.py::test_build_features_shape -x -q

# With coverage report
uv run pytest tests/ --cov=. -q
```

**Test patterns**:
- Unit tests mock external dependencies (Redis, Zerodha Kite, TimescaleDB)
- Integration tests use `runner_mocked` fixture (see `tests/conftest.py`)
- All tests run in parallel via pytest-xdist

#### B. **Code Quality Checks** (Before every commit)
```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check . --fix

# Type check
uv run mypy . --strict

# Security audit
uv run bandit -c pyproject.toml -r .

# Dependency vulnerabilities
uv run pip-audit
```

**Pre-commit hooks** automatically run `ruff check + format` + `bandit` on every commit.

#### C. **Running the System**
```bash
# Paper trading (safe mode, no real capital)
python -m orchestrator.main

# With specific market
python -m orchestrator.main --market equity      # NSE only
python -m orchestrator.main --market crypto      # Binance only
python -m orchestrator.main --market both        # Both

# Check logs
tail -f logs/trading.log
```

**Entry point**: `orchestrator/main.py` — TradingSystem class orchestrates all subsystems.

---

### 3. **Architecture at a Glance**

```
┌─────────────────────────────────────────────────────┐
│            ORCHESTRATOR / SCHEDULER                 │
│  (Main entry point, coordinates all modules)        │
└────────┬────────────┬────────────┬────────────┬─────┘
         │            │            │            │
    ┌────▼────┐  ┌────▼────┐  ┌───▼───┐  ┌───▼────┐
    │ DATA     │  │ SIGNALS │  │ RISK  │  │EXECUTION
    │ ────     │  │ ─────── │  │ ──── │  │────────
    │ • Kite   │  │ • Feat. │  │ •CB  │  │ • Orders
    │ • Redis  │  │ • ML    │  │ •Risk│  │ • Fills
    │ • TSDB   │  │ • Score │  │ • SE │  │
    └──────────┘  └─────────┘  └──────┘  └────────┘
         │            │            │            │
         └────────────┬────────────┬────────────┘
                      │
            ┌─────────▼────────┐
            │ MONITORING       │
            │ ────────────     │
            │ • MLflow (logs)  │
            │ • Telegram       │
            │ • Audit trail    │
            └──────────────────┘
```

**Key modules**:
- `data/` — Market data ingestion, storage, caching
- `signals/` — Feature engineering, model inference, signal generation
- `llm/` — Sentiment analysis (FinBERT)
- `risk/` — Circuit breakers, position sizing, limits
- `execution/` — Order placement, fill tracking, slippage
- `orchestrator/` — Main loop, scheduling, state management
- `monitoring/` — Reporting, alerts, audit persistence
- `portfolio/` — Portfolio state, risk checks
- `fundamentals/` — Multibagger scoring for watchlist
- `backtest/` — Walk-forward validation, performance attribution
- `options/` — F&O specific models (IV rank, delta-neutral)

---

### 4. **Code Review Graph (MCP Tool Integration)**

**CRITICAL**: The project has a code knowledge graph. Use it first before manual exploration:

```bash
# Build the graph (auto-updates on file changes)
# (Done automatically if MCP tools are available)

# Use graph tools for:
# - Finding function relationships: query_graph(pattern="callers_of", target="extract_features")
# - Understanding impact: get_impact_radius() for code changes
# - Semantic search: semantic_search_nodes(query="signal filtering")
# - Architecture: get_architecture_overview()
```

**Graph tools are faster and cheaper than grep/glob** — always prefer them for exploration.

---

### 5. **Common Tasks & Workflows**

#### Task: Add a New Trading Strategy
1. Extend `signals/base_strategy.py::BaseStrategy`
2. Implement `scan(symbol: str, df: pd.DataFrame) → dict | None`
3. Register in `config/strategy_params.yaml` under `strategies` section
4. Tests should validate:
   - Pure CPU function (no DB/Redis/network calls)
   - Returns normalized `Signal` object or None
   - Handles edge cases (empty data, NaN values)

**Example**: `signals/vcp_strategy.py` (Volume-Change-Price breakout)

#### Task: Optimize Performance
1. Profile with `cProfile` or `py-spy` to find bottlenecks
2. Check **technical reviews** for known issues:
   - `DATA_LAYER_REVIEW.md` — Data layer optimizations
   - `SIGNAL_PIPELINE_REVIEW.md` — Feature & ML layer (Phase 9 quick wins)
   - `ML_PIPELINE_REVIEW.md` — Ensemble model parallelization
   - `ORCHESTRATOR_REVIEW.md` — Scheduling & state management
3. Apply quick wins from `PHASE_9_KICKOFF_GUIDE.md` (17 hours total, 40%+ improvement potential)

#### Task: Add a Risk Check
1. Implement in `portfolio/risk_manager.py::PreExecutionRiskCheck`
2. Call from `orchestrator/runner.py::OrchestratorRunner.run_signal()`
3. Must enforce hard limits (e.g., max position 2%, daily DD 5%)
4. Never use soft limits — always halt execution if breached

#### Task: Debug a Test Failure
```bash
# Run failing test with verbose output
uv run pytest tests/test_file.py::test_name -xvs

# See full traceback
uv run pytest tests/test_file.py::test_name -xvs --tb=long

# Check mocks/patches in conftest.py (lines 1-100)
view tests/conftest.py
```

---

### 6. **Critical Do's & Don'ts**

#### ✅ DO:
- Use `uv sync --frozen` for dependency stability
- Run tests before committing (`pytest tests/ -q`)
- Use `structlog` for logging (never `print()`)
- Cache features in Redis with 5-min TTL (Phase 9 optimization)
- Call broker APIs through `data/rate_limiter.py` wrapper
- Validate all signals through `signals/contracts.py::Signal`
- Use walk-forward validation only (no in-sample backtesting)
- Log all trades/signals/risk decisions (audit trail)
- Never hardcode Telegram token or API keys in code

#### ❌ DON'T:
- Modify `./models/production/` — Read-only, versioned
- Echo or log `.env` contents
- Call external APIs without rate limiter
- Hardcode TimescaleDB queries without time dimension
- Use threads for CPU-bound code (GIL contention) — use ThreadPoolExecutor for I/O
- Deploy live without 200+ paper-trade validation
- Override circuit breaker limits programmatically
- Modify `FEATURE_COLUMNS` without retraining all models

---

### 7. **Phase 9 Quick Wins (Next Steps)**

**17 hours of work → 40%+ performance improvement + ₹180K-₹240K annual ROI**

**Task Group 1: Data Layer** (1 hour)
- [ ] Add threading.Lock() to bar aggregator (line 130-146 in `data/live_feed.py`)
- [ ] Replace `redis.keys()` with `redis.smembers()` (line 365)
- [ ] Add tick buffer backpressure + retry logic (line 336)

**Task Group 2: Signal Pipeline** (7 hours)
- [ ] Implement feature caching in Redis (TTL 24h)
- [ ] Batch model predictions (ensemble.predict_batch instead of loop)
- [ ] Remove redundant astype() calls (combine into single call)
- [ ] Parallelize data fetching with ThreadPoolExecutor

**Task Group 3: Risk & Execution** (3 hours)
- [ ] Batch Redis persistence for circuit breaker state
- [ ] Cache circuit breaker check (avoid N+1 Redis fetches)
- [ ] Integrate portfolio risk checks before every trade

**Task Group 4: Orchestration** (2 hours)
- [ ] Clear state accumulation (prevent OOM)
- [ ] Fix A/B routing logic (deterministic MD5-based split)
- [ ] Add failed job tracking + Telegram alerts

**Task Group 5: ML Pipeline** (2.75 hours)
- [ ] Parallelize ensemble voting with ThreadPoolExecutor
- [ ] Cache computed features (90 min work, 25% throughput gain)

**See**: `PHASE_9_KICKOFF_GUIDE.md` for detailed implementation steps.

---

### 8. **Debugging & Troubleshooting**

#### Issue: Tests fail with "conftest: env stub missing"
**Solution**: Add missing `Settings` field to `tests/conftest.py` stub_env dict

#### Issue: Kite WebSocket keeps disconnecting
**Solution**: Use reconnect loop in `data/live_feed.py` (already implemented)

#### Issue: TimescaleDB hypertable query slow
**Solution**: Always include time dimension in WHERE clause (required for partition pruning)

#### Issue: Redis keys() blocking feed
**Solution**: Use `redis.smembers()` for set operations (Phase 9 fix #2)

#### Issue: Feature computation slow
**Solution**: Enable Redis caching + batch predictions (Phase 9 fixes #1-2)

---

### 9. **Documentation & Reference**

| Document | Purpose |
|----------|---------|
| `project.md` | Complete 16-week build roadmap + module descriptions |
| `CLAUDE.md` | MCP tools integration + project conventions |
| `TECHNICAL_REVIEW_MASTER_SUMMARY.md` | 40+ optimization opportunities (40-44% improvement potential) |
| `PHASE_9_KICKOFF_GUIDE.md` | Step-by-step implementation guide for quick wins |
| `DATA_LAYER_REVIEW.md` | Data layer deep dive (13 issues, 3 critical) |
| `SIGNAL_PIPELINE_REVIEW.md` | Feature/ML optimization opportunities |
| `ML_PIPELINE_REVIEW.md` | Ensemble model parallelization (45% latency reduction) |
| `ORCHESTRATOR_REVIEW.md` | Scheduling, state management, HA issues |
| `ARCHITECTURE_REVIEW.md` | Risk/execution race conditions, slippage |
| `README.md` | Quick start guide |
| `docs/ARCHITECTURE.md` | High-level system design |
| `backtest/STRATEGY_HARNESS_GUIDE.md` | Walk-forward validation protocol |

---

### 10. **Performance Targets**

| Metric | Current | Target | Effort |
|--------|---------|--------|--------|
| Scan time | 15 min | 9 min | 7 hours (Phase 9) |
| P99 latency | 800ms | 100ms | 1 hour (data layer fix) |
| Model inference | 40-50 min | 15-20 min | 2.75 hours (ensemble) |
| Daily scans | 4-6 | 8-12 | Throughput gain |
| Data loss | 0.3% | 0.01% | 20 min (backpressure) |
| Order fill rate | 85-90% | 95-98% | 3 hours (risk/exec) |

---

### 11. **Support & Quick Help**

**Testing**:
```bash
uv run pytest tests/ -x -q
```

**Code quality**:
```bash
uv run ruff format . && uv run ruff check . --fix && uv run mypy . --strict
```

**Run system**:
```bash
python -m orchestrator.main --market both
```

**View logs**:
```bash
tail -f logs/trading.log
```

**Check code relationships**:
Use code-review-graph MCP tools for semantic search, impact analysis, and architecture overview.

---

## Next Steps

1. **Setup** (5 min): `uv sync --frozen --dev && uv run alembic upgrade head`
2. **Verify** (10 min): `uv run pytest tests/ -q` (should pass 1100+ tests)
3. **Phase 9 Quick Wins** (17 hours): Follow `PHASE_9_KICKOFF_GUIDE.md` for 40%+ performance gain
4. **Monitor** (ongoing): Use Telegram alerts + MLflow dashboards for real-time insights
5. **Live deployment**: Only after 200+ paper-trade validation checklist

---

**Last Updated**: April 25, 2026
**Status**: ✅ Ready to use efficiently
