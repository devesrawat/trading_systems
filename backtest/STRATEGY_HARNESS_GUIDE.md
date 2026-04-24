# Strategy Backtest Harness Guide

## Overview

The `StrategyBacktester` validates BaseStrategy plugins before they reach live trading. It runs historical backtests, computes performance metrics, and evaluates against promotion gates.

## Promotion Gates

A strategy must pass ALL gates to be promoted:

| Gate | Threshold | Rationale |
|------|-----------|-----------|
| Min Trades | ≥ 20 | Statistical significance |
| Annual Return | > 15% | Meaningful returns after costs |
| Max Drawdown | < -20% | Acceptable risk envelope |
| Profit Factor | > 1.5 | Win ratio sufficient |
| Expectancy | > 0 | Positive edge |

## Basic Usage

```python
from backtest.strategy_harness import StrategyBacktester
from signals.strategies.vcp import VCPStrategy
import pandas as pd

# 1. Create backtester
backtester = StrategyBacktester(
    initial_capital=100_000,
    slippage_bps=1.0,      # 1bp slippage
    commission_bps=0.5,     # 0.5bp commission
    liquidity_tier="large_cap"
)

# 2. Prepare data dictionary {symbol: OHLCV DataFrame}
data_dict = {
    "RELIANCE.NS": df_reliance,
    "INFY.NS": df_infy,
    # ... more symbols
}

# 3. Run backtest
strategy = VCPStrategy()
metrics, gate = backtester.run_backtest(
    strategy=strategy,
    symbols=["RELIANCE.NS", "INFY.NS"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    data_dict=data_dict
)

# 4. Check results
print(f"Trades: {metrics.trades}")
print(f"Annual Return: {metrics.annual_return_pct:.2f}%")
print(f"Max Drawdown: {metrics.max_drawdown_pct:.2f}%")
print(f"Sharpe: {metrics.sharpe:.2f}")
print(f"Profit Factor: {metrics.profit_factor:.2f}")
print(f"Gate Passed: {gate.passed}")
print(f"Reason: {gate.reason}")

# 5. Log to MLflow
backtester.log_to_mlflow(
    run_name=f"vcp_validation_{strategy.name}",
    params={
        "strategy": strategy.name,
        "version": "1.0",
        "symbol_count": len(["RELIANCE.NS", "INFY.NS"]),
    },
    metrics=metrics.to_dict(),
    gate_decision=gate
)
```

## Edge Cases Handled

### No Trades Generated
```
Result: All metrics = 0, fails gates
Reason: "No signals generated during backtest period"
Action: Review strategy parameters, ensure symbols are suitable
```

### All Losing Trades
```
Result: profit_factor = 0, fails gates
Max drawdown spike likely triggers gate failure too
Action: Adjust strategy setup parameters
```

### Insufficient History
```
Result: All metrics = 0, fails gates
Reason: "No trades executed"
Action: Provide at least strategy.lookback_days of data before start_date
```

### Liquidity Constraints
Trade is skipped if `apply_liquidity_constraints()` returns False.
Override this method to implement custom liquidity checks:

```python
class CustomBacktester(StrategyBacktester):
    def apply_liquidity_constraints(self, symbol: str, qty: int) -> bool:
        # Custom logic: check market cap, avg volume, bid-ask spread
        market_cap = get_market_cap(symbol)
        if market_cap < 500_cr:
            return False  # Too illiquid
        return True
```

### Survivorship Bias
Trade is skipped if `apply_survivorship_safeguards()` returns False.
By default, allows all symbols. Override to check NSE universe history:

```python
class SurvivalBiasBacktester(StrategyBacktester):
    def apply_survivorship_safeguards(self, symbol: str, date: pd.Timestamp) -> bool:
        # Check if symbol was actively listed on NSE on given date
        return was_listed_on_date(symbol, date)
```

## Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| Total Return | (1 + returns).prod() - 1 | Cumulative P&L |
| Annual Return | daily_return × 252 | Annualized P&L |
| Max Drawdown | peak-to-trough decline | Worst underwater level |
| Sharpe Ratio | (mean_return / std) × √252 | Risk-adjusted return |
| Profit Factor | sum(wins) / sum(losses) | Win quality relative to losses |
| Win Rate | wins / total_trades | % of positive trades |
| Expectancy | (avg_win × wr) - (avg_loss × lr) | Average trade value |
| Calmar Ratio | annual_return / abs(max_dd) | Return per unit drawdown |

## Cost Model

Costs include:
- **Slippage**: Market impact (1bp default for large-caps)
- **Commission**: Broker fee (0.5bp default)
- **STT**: Securities Transaction Tax (already baked into NSE costs)

Each trade pays both entry and exit costs (round-trip).

## Testing Your Strategy

Use the provided synthetic data helpers:

```python
from tests.test_backtest_harness import _make_daily

# Create synthetic uptrend suitable for trend strategies
df = _make_daily(n=500, trend="up", seed=42)

# Or create downtrend
df = _make_daily(n=500, trend="down", seed=42)
```

## Integration with MLflow

All backtests log to MLflow under experiment `strategy_validation`:

```
Experiment: strategy_validation
├── Run: vcp_validation_vcp_2024-01-15
│   ├── Params: strategy, version, symbol_count, period
│   ├── Metrics: annual_return%, max_dd%, sharpe, profit_factor, ...
│   ├── Artifacts:
│   │   ├── gate_decision.txt
│   │   └── trade_log.json (if trades executed)
```

View results:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
# http://localhost:5000
```

## Production Workflow

```
Strategy Backtest → Gate Evaluation → Promotion Decision
         ↓
    20+ trades?
    15%+ return?
    <20% drawdown?
    1.5+ PF?
    >0 expectancy?
         ↓
   PASS → Paper Trading (100 trades, 2 weeks)
   FAIL → Iterate & Re-test
```

Once paper trading succeeds, strategy is ready for live mode.
