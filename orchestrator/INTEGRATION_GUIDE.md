# Phase 7: Orchestrator Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   OrchestratorRunner                            │
│                   (Phase 7 Integration)                         │
└─────────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
   Pre-market            Trading Loop      Post-market
   Setup                  (5-min loop)      Summary
        │                   │                   │
        ├─ Registry         ├─ Mode Gate       ├─ Final P&L
        ├─ Strategies       ├─ Risk Check      ├─ Drift Check
        ├─ Scanner          ├─ CircuitBreaker  ├─ Reconciliation
        ├─ Normalization    ├─ Execution       └─ Telegram
        ├─ Watchlist        ├─ Audit Trail
        └─ Telegram         └─ Telegram


Data Flow (Single Signal):

Strategy Dict
    │ (raw scanner result)
    ▼
Signal Router
    │ normalize_strategy_result()
    ▼
Signal Object (contracts.Signal)
    │
    ├─ signal_id, timestamp
    ├─ symbol, exchange, asset_class
    ├─ strategy_name, signal_type, direction
    ├─ confidence, score, rank
    ├─ entry/risk specs
    ├─ mode (research/watchlist/paper/live)
    └─ features, attribution, metadata
    │
    ▼
Mode Gate (orchestrator.main._mode_gate)
    │
    ├─ research  → ❌ Never execute
    ├─ watchlist → ❌ Never execute
    ├─ paper     → ✓ If paper_trade_mode=True
    └─ live      → ✓ If paper_trade_mode=False
    │
    ▼
PreExecutionRiskCheck (portfolio.risk_manager)
    │
    ├─ Sector concentration (<5% per sector)
    ├─ Single-stock concentration (<2% hard cap)
    ├─ Correlation penalty
    ├─ Liquidity check
    ├─ Turnover limit
    └─ Position count
    │
    ▼
CircuitBreaker.check()
    │
    ├─ Daily drawdown < limit (3% default)
    ├─ Weekly drawdown < limit (7% default)
    └─ Manual halt status
    │
    ▼
ExecutionAdapter.execute_order()
    │
    ├─ Paper mode  → Simulator (paper_broker)
    ├─ Kite mode   → Zerodha Kite API
    ├─ Upstox mode → Upstox API
    └─ Binance     → Binance WebSocket
    │
    ▼
TradeLogger
    │
    ├─ MLflow experiment (signal decision)
    ├─ Features, confidence, execution
    └─ Later: P&L reconciliation
    │
    ▼
TelegramNotifier
    │
    ├─ Trade execution alert
    ├─ Risk block alert
    ├─ End-of-day summary
    └─ Error notifications
```

## Data Models

### Signal Contract (signals/contracts.py)

```python
Signal
├─ signal_id: str              # UUID4
├─ timestamp: datetime         # UTC
├─ symbol: str                 # RELIANCE.NS
├─ exchange: str               # NSE, BSE, BINANCE
├─ asset_class: str            # equity, crypto, futures
│
├─ strategy_name: str          # vcp, rs_breakout, ml_long
├─ strategy_version: str       # 1.0
├─ signal_type: SignalType     # scanner_hit, ml_prediction, entry, exit
├─ direction: Direction        # long, short, neutral, exit
│
├─ confidence: float [0, 1]    # Strategy confidence
├─ score: float [0, 1]         # Overall score (ML prob, fundamental rank)
├─ rank: int                   # Rank in universe (1 = best)
│
├─ timeframe: str              # daily, 5minute
├─ entry: EntrySpec            # entry_price, stop_price, target_price
├─ risk: RiskSpec              # size_hint_pct, capital_at_risk
│
├─ features: dict              # TA indicators, sentiment, etc. (for debugging)
├─ attribution: dict           # Model version, SHAP values, weights
│
├─ mode: str                   # research, watchlist, paper, live ⭐
├─ metadata: dict              # Custom data
└─ raw_payload: dict           # Original strategy dict
```

### Risk Decision (portfolio/schema.py)

```python
RiskDecision
├─ allowed: bool               # Execution allowed?
├─ reason: str                 # Why allowed/denied
├─ capital_allocated: float    # Amount to allocate
├─ priority: int               # Risk priority (1=highest)
└─ details: dict               # Sector exposure, correlation, etc.
```

### Portfolio State (portfolio/schema.py)

```python
PortfolioState
├─ cash: float                 # Available cash
├─ positions: dict             # {symbol: Position}
├─ open_orders: dict           # {order_id: Order}
└─ timestamp: datetime         # State snapshot time
```

## Implementation: OrchestratorRunner

Located in `orchestrator/runner.py`:

```python
class OrchestratorRunner:
    """
    Phase 7 orchestrator — end-to-end integration.

    Methods:
        pre_market_setup()     → Registry → Strategies → Scan → Normalize
        trading_loop()         → Mode gate → Risk check → Execute
        post_market_summary()  → Final P&L → Drift check → Telegram
    """

    def __init__(self, market_type: str | None = None) -> None:
        """Initialize all subsystems."""

    def pre_market_setup(self) -> None:
        """
        1. Load StrategyRegistry
        2. Validate strategy backtest promotion gates
        3. Load MultibaggerWatchlist
        4. Initialize PreExecutionRiskCheck
        5. Initialize TelegramNotifier
        6. Run ScannerEngine
        7. Normalize signals to Signal objects
        8. Send pre-market Telegram summary
        """

    def trading_loop(self) -> None:
        """
        For each Signal:
            a. Apply mode gate
            b. PreExecutionRiskCheck
            c. CircuitBreaker.check()
            d. ExecutionAdapter.execute_order()
            e. Log to audit trail
            f. Send Telegram alert
        """

    def post_market_summary(self) -> None:
        """
        - Log final portfolio state to MLflow
        - Check model drift
        - Send end-of-day Telegram summary
        """
```

## Configuration Checklist

### 1. Strategies in YAML (config/strategy_params.yaml)

```yaml
strategies:
  - name: vcp
    enabled: true
    class_path: signals.strategies.vcp.VCPStrategy
    interval: day
    lookback_days: 60
    asset_classes: [equity]
    params: {}

  - name: rs_breakout
    enabled: true
    class_path: signals.strategies.rs_breakout.RSBreakoutStrategy
    interval: day
    lookback_days: 90
    asset_classes: [equity]
    params: {}
```

### 2. Risk Limits (portfolio/schema.py)

```python
DEFAULT_EQUITY_LIMITS = RiskLimits(
    max_position_pct=0.02,           # 2% hard cap per stock
    max_sector_pct=0.05,             # 5% per sector
    max_correlation=0.7,             # Filter correlated pairs
    min_liquidity_score=0.5,         # Minimum liquidity
    turnover_limit_pct=0.10,         # Max 10% daily turnover
    max_positions=10,                # Max 10 open positions
)

DEFAULT_CRYPTO_LIMITS = RiskLimits(
    max_position_pct=0.01,           # 1% per coin
    max_correlation=0.8,
    min_liquidity_score=0.3,
    turnover_limit_pct=0.05,
    max_positions=5,
)
```

### 3. Alert Channels (.env)

```bash
# Telegram (required for alerts)
TELEGRAM_BOT_TOKEN=<token>
TELEGRAM_CHANNEL_ID=<channel_id>

# Market type (equity | crypto | both)
MARKET_TYPE=equity

# Paper trading (required — default True)
PAPER_TRADE_MODE=true

# Data provider (kite | upstox | binance)
DATA_PROVIDER=kite

# Feature drift detection
FEATURE_DRIFT_THRESHOLD=0.05

# Model drift detection
MODEL_DRIFT_THRESHOLD=0.10
```

## Deployment Checklist

### Phase 1: Development (Isolated Testing)

- [ ] Run all unit tests: `uv run pytest tests/ -k "test_orchestrator" -v`
- [ ] Run integration tests: `uv run pytest tests/test_orchestrator_integration.py -v`
- [ ] Type check: `uv run mypy orchestrator/ portfolio/ signals/ --strict`
- [ ] Lint: `uv run ruff check orchestrator/`
- [ ] Test with synthetic data (no real market)

### Phase 2: Backtest Validation (Historical Data)

- [ ] Backtest all enabled strategies on 1-year history
- [ ] Verify each strategy passes promotion gates:
  - Min 20 trades
  - Annual return > 15%
  - Max drawdown < 20%
  - Profit factor > 1.5
  - Win rate > 50%
- [ ] Store backtest metrics in MLflow
- [ ] Review feature drift (compare to training data)
- [ ] Command: `uv run backtest/runner.py --strategies=all --period=1y`

### Phase 3: Paper Trading (1 Week)

- [ ] Deploy to paper_trade_mode=True
- [ ] Monitor live signal generation
- [ ] Check:
  - Strategy signals are generated
  - Risk checks are working (some rejections expected)
  - Telegram alerts are flowing
  - No crashes or memory leaks
  - Model predictions look reasonable
- [ ] Target: 20+ trades for validation
- [ ] Check daily P&L drift from backtest

### Phase 4: Live Trading (Production)

- [ ] Complete paper trading checklist
- [ ] Switch paper_trade_mode=False
- [ ] Start with market_type=equity only (smaller universe)
- [ ] Monitor:
  - Order fills on real market
  - Slippage vs paper assumptions
  - Sector concentration stays within limits
  - Daily drawdown doesn't exceed 3%
- [ ] After 1 month, evaluate:
  - Win rate vs backtest
  - Max drawdown
  - Model drift
  - Feature drift
- [ ] Expand to market_type=both if successful

## Error Handling

### Telegram Down

```python
# OrchestratorRunner._send_telegram_alert()
try:
    bot = self._init_telegram()
    if bot:
        bot.send_signal_alert(...)
except Exception as e:
    log.warning("telegram_send_failed", error=str(e))
    # ❌ Don't crash trading!
    # Trading continues with just log.info()
```

### Broker Down

```python
# orchestrator/main._execute_signal()
try:
    order_id = self._executor.place_market_order(...)
except Exception as e:
    log.error("execution_failed", error=str(e))
    self._send_alert(f"Broker error: {e}")
    # ❌ Circuit breaker halts on repeated failures
    # Check CircuitBreaker.check() for halt status
```

### Database Down

```python
# data/store.py get_engine()
# Creates connection pool with retries
# Raises on persistent failure
```

### Signal Normalization Error

```python
# orchestrator/main.pre_market_setup()
try:
    signal = normalize_strategy_result(...)
except ValueError as e:
    log.warning("signal_normalization_failed", error=str(e))
    # ❌ Skip this signal; continue scanning others
```

## Testing Strategy

### Unit Tests (orchestrator/main.py)

- Mode gating logic
- Signal normalization
- Feature fetching
- Signal execution (both old and new interfaces)

### Integration Tests (tests/test_orchestrator_integration.py) — 600+ lines

**Signal Pipeline**
- ✓ End-to-end: Signal → Normalization → Risk → Audit
- ✓ Mode gating (research, watchlist, paper, live)
- ✓ Risk gates (sector, correlation, liquidity)
- ✓ Audit trail (all decisions logged)

**Execution Modes**
- ✓ Paper mode: signals execute normally
- ✓ Live mode: restricted to paper=False
- ✓ Research mode: never execute
- ✓ Watchlist mode: never execute

**Error Resilience**
- ✓ Telegram down → trading continues
- ✓ Broker error → logged, circuit breaks eventually
- ✓ Database error → logged, system recovers
- ✓ Model missing → skip cycle, log warning

**Strategy Validation**
- ✓ Backtest validation gates
- ✓ Promotion gates (min trades, return, drawdown, etc.)
- ✓ Deprecation warnings for strategies

**Portfolio State**
- ✓ Holdings updated after execution
- ✓ Risk recalculated with each position
- ✓ Correlation penalties applied
- ✓ Sector exposure tracked

### Running Tests

```bash
# All orchestrator tests
uv run pytest tests/test_orchestrator*.py -v

# Integration tests only
uv run pytest tests/test_orchestrator_integration.py -v --tb=short

# With coverage
uv run pytest tests/test_orchestrator_integration.py --cov=orchestrator --cov=portfolio --cov=signals

# Specific test class
uv run pytest tests/test_orchestrator_integration.py::TestModeGating -v

# Specific test
uv run pytest tests/test_orchestrator_integration.py::TestModeGating::test_research_mode_never_executes -v
```

## Sequence Diagrams

### Pre-market Setup

```
OrchestratorRunner.pre_market_setup()
    │
    ├─→ TradingSystem.pre_market_setup()
    │   ├─ Refresh auth (broker)
    │   ├─ Load universe
    │   ├─ Run ScannerEngine
    │   │  ├─ VCPStrategy.scan()
    │   │  ├─ RSBreakoutStrategy.scan()
    │   │  └─ [collect raw dicts]
    │   └─ Sentiment pipeline
    │
    ├─→ StrategyRegistry.enabled_strategies()
    │   └─ [validate backtest status]
    │
    ├─→ PreExecutionRiskCheck.__init__()
    │   └─ [load limits from config]
    │
    ├─→ TelegramSignalBot.__init__()
    │   └─ [connect to bot API]
    │
    ├─→ For each strategy result dict:
    │   │
    │   ├─ normalize_strategy_result()
    │   │  └─ Signal (contracts.Signal)
    │   │
    │   ├─ Add to _pre_market_signals
    │   └─ Log signal_generated
    │
    └─→ TelegramSignalBot.send_signal_alert()
        └─ "Pre-market OK | 3 strategies | 500 universe | 25 signals"
```

### Trading Loop (Per Signal)

```
OrchestratorRunner.trading_loop()
    │
    ├─→ For each Signal in _pre_market_signals:
    │   │
    │   ├─→ _mode_gate()
    │   │   ├─ research? → SKIP
    │   │   ├─ watchlist? → SKIP
    │   │   ├─ paper?
    │   │   │  └─ paper_trade_mode=True? → OK
    │   │   │     else → SKIP
    │   │   └─ live?
    │   │      └─ paper_trade_mode=False? → OK
    │   │         else → SKIP
    │   │
    │   ├─→ PreExecutionRiskCheck.check_signal_execution(signal, portfolio)
    │   │   ├─ Sector concentration?
    │   │   ├─ Correlation penalty?
    │   │   ├─ Liquidity score?
    │   │   └─ Decision (allowed, capital_allocated)
    │   │
    │   ├─→ If rejected:
    │   │   ├─ Log rejection reason
    │   │   ├─ Telegram alert
    │   │   └─ SKIP execution
    │   │
    │   ├─→ CircuitBreaker.check()
    │   │   ├─ Is halted?
    │   │   ├─ Daily DD exceeded?
    │   │   └─ Decision (allowed, reason)
    │   │
    │   ├─→ If rejected:
    │   │   ├─ Telegram alert
    │   │   └─ SKIP execution
    │   │
    │   ├─→ ExecutionAdapter.execute_order()
    │   │   ├─ Paper mode: simulator
    │   │   ├─ Kite mode: broker API
    │   │   └─ Return order_id
    │   │
    │   ├─→ TradeLogger.log_signal()
    │   │   ├─ MLflow experiment run
    │   │   ├─ Features, confidence, action
    │   │   └─ For later P&L reconciliation
    │   │
    │   └─→ TelegramSignalBot.send_signal_alert()
    │       └─ "✅ RELIANCE.NS | vcp | conf=75% | capital=10,000"
    │
    └─ Done
```

## Migration Path (Existing to Phase 7)

### Old Interface (pre-Phase 7)

```python
# orchestrator/main.TradingSystem._execute_signal(symbol, feature_df)
for symbol in universe:
    feature_df = features_map.get(symbol)
    self._execute_signal(symbol, feature_df, asset_class="equity")
```

### New Interface (Phase 7)

```python
# orchestrator/runner.OrchestratorRunner._process_signal(signal)
for signal in signals:
    self._process_signal(signal)
```

### Compatibility

- `_execute_signal()` still accepts both `(symbol, feature_df)` and `(Signal)`
- Existing code continues to work
- New code uses Signal objects throughout
- Phase 8: Deprecate old interface entirely

## Monitoring & Observability

### MLflow Experiments

```
nse_equity_signals (MLflow experiment)
  run_20240425_093015
    ├─ params: {strategy: vcp, symbol: RELIANCE.NS, ...}
    ├─ metrics: {confidence: 0.75, signal_prob: 0.65, ...}
    ├─ artifacts: {features.json, attribution.json}
    └─ tags: {decision: EXECUTED, mode: paper}

  run_20240425_093020
    ├─ params: {strategy: rs_breakout, symbol: INFY.NS, ...}
    ├─ metrics: {confidence: 0.55, ...}
    └─ tags: {decision: SKIPPED, reason: low_confidence}
```

### Telegram Alerts

```
📊 Pre-market setup complete
Strategies: 3 enabled
Universe: 500 symbols
Signals: 25 generated
Watchlist: 150 candidates
Backtest status: 3/3 passed

✅ RELIANCE.NS | vcp | conf=75% | capital=10,000
🟡 INFY.NS | rs_breakout | WATCHLIST (mode gate)
⚠️ WIPRO.NS | vcp | Risk block: sector concentration

📈 End-of-day summary
Portfolio: ₹512,450
Daily DD: 1.2%
Weekly DD: 3.5%
```

### Logs (structlog)

```
timestamp=2024-04-25T09:30:15 level=info event=pre_market_setup_start market=equity
timestamp=2024-04-25T09:30:16 level=info event=strategy_signal_generated signal_id=abc123 symbol=RELIANCE.NS strategy=vcp confidence=0.75
timestamp=2024-04-25T09:30:17 level=warning event=strategy_backtest_validation_failed strategy=rs_breakout reason="Win rate < 50%"
timestamp=2024-04-25T09:30:18 level=info event=pre_market_setup_complete signals_collected=25
```

## Troubleshooting

### No signals being generated

1. Check `ScannerEngine.run()` output
2. Verify strategies are enabled in YAML
3. Check universe has sufficient data
4. Look for scanner errors in logs (strategy.prepare() failures)

### Signals generated but not executing

1. Check mode gate: is signal.mode="paper" but paper_trade_mode=False?
2. Check circuit breaker: is it halted?
3. Check risk checker: is there a sector concentration block?
4. Look for "signal_rejected" log entries with reason

### Telegram alerts not flowing

1. Check TELEGRAM_BOT_TOKEN in .env
2. Check TELEGRAM_CHANNEL_ID in .env
3. Look for "telegram_initialization_failed" in logs
4. Verify bot has access to channel

### High slippage in paper vs live

1. Check paper assumptions in `OrderExecutor._apply_slippage()`
2. Compare actual fills to paper fills
3. Adjust liquidity tier if needed
4. Check market impact (large orders on thin stocks)

## References

- **signals/contracts.py** — Signal data model
- **signals/registry.py** — Strategy loading and validation
- **signals/scanner_engine.py** — Parallel strategy scanning
- **signals/signal_router.py** — Signal normalization
- **portfolio/risk_manager.py** — PreExecutionRiskCheck
- **portfolio/schema.py** — RiskDecision, PortfolioState
- **execution/orders.py** — OrderExecutor
- **execution/logger.py** — TradeLogger
- **monitoring/telegram_bot.py** — TelegramNotifier
- **orchestrator/main.py** — TradingSystem (base)
- **orchestrator/runner.py** — OrchestratorRunner (Phase 7)
- **tests/test_orchestrator_integration.py** — Integration tests
- **backtest/strategy_harness.py** — StrategyBacktester (validation)

## Next Steps (Phase 8)

- [ ] Deprecate old `_execute_signal(symbol, feature_df)` interface
- [ ] Migrate all equity/crypto cycles to Signal-based pipeline
- [ ] Add sentiment scoring as Signal attribution
- [ ] Add ML model confidence → Signal confidence mapping
- [ ] Performance optimization: batch risk checks
- [ ] Add distributed scanning (Dask) for large universes
