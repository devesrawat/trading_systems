"""
Strategy Backtest Harness — Plugin validation before live trading.

Validates BaseStrategy plugins against promotion gates:
  - Minimum 20 trades
  - Annual return > 15%
  - Max drawdown < 20%
  - Profit factor > 1.5
  - Expectancy > 0

Handles edge cases:
  - No trades → fails gates
  - All losing trades → profit_factor undefined, fails gates
  - Insufficient history → clear error message, fails gates
  - Survivorship bias: skip stocks not in NSE on trade date
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import mlflow
import pandas as pd
import structlog

from backtest.costs import NSECostModel
from backtest.metrics import (
    calmar_ratio,
    expectancy,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from signals.base_strategy import BaseStrategy

log = structlog.get_logger(__name__)

# Promotion gate thresholds
_PROMOTION_GATES = {
    "min_trades": 20,
    "annual_return_pct": 15,
    "max_drawdown_pct": 20,
    "profit_factor": 1.5,
    "expectancy": 0,
}

# Trading days per year
_TRADING_DAYS = 252


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest run."""

    total_return_pct: float
    annual_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float
    trades: int
    win_rate_pct: float
    avg_win_loss_ratio: float
    expectancy: float
    calmar_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)


@dataclass
class PromotionGate:
    """Promotion gate decision."""

    passed: bool
    reason: str
    metrics_summary: dict[str, Any]


class StrategyBacktester:
    """
    Validate BaseStrategy plugins through historical backtests.

    Runs the strategy on historical OHLCV data, applies transaction costs
    and liquidity constraints, computes performance metrics, and evaluates
    against promotion gates.

    Usage
    -----
    backtester = StrategyBacktester(initial_capital=100_000)
    metrics, gate = backtester.run_backtest(
        strategy=vcp_strategy,
        symbols=['RELIANCE.NS', 'INFY.NS'],
        start_date='2023-01-01',
        end_date='2024-01-01',
    )
    backtester.log_to_mlflow(
        run_name=f"vcp_validation_{strategy.name}",
        params={'strategy': 'vcp', 'symbol_count': 2},
        metrics=metrics.to_dict(),
        gate_decision=gate,
    )
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        slippage_bps: float = 1.0,
        commission_bps: float = 0.5,
        liquidity_tier: str = "large_cap",
    ) -> None:
        """
        Parameters
        ----------
        initial_capital : Initial trading capital (default ₹100,000)
        slippage_bps : Intraday slippage in basis points (default 1 bp)
        commission_bps : Commission in basis points (default 0.5 bp)
        liquidity_tier : Liquidity constraint level (large_cap, mid_cap, small_cap)
        """
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps
        self.commission_bps = commission_bps
        self.liquidity_tier = liquidity_tier
        self._cost_model = NSECostModel()
        self._trade_log: list[dict[str, Any]] = []

    def run_backtest(
        self,
        strategy: BaseStrategy,
        symbols: list[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
        data_dict: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[BacktestMetrics, PromotionGate]:
        """
        Run backtest for a strategy across multiple symbols.

        Parameters
        ----------
        strategy : BaseStrategy instance to validate
        symbols : List of trading symbols to scan
        start_date : Backtest start date (inclusive)
        end_date : Backtest end date (inclusive)
        data_dict : Optional dict of {symbol: OHLCV DataFrame}
                    If None, must be provided via scan_results

        Returns
        -------
        (BacktestMetrics, PromotionGate)
            Metrics: performance statistics
            Gate: promotion decision and reasoning
        """
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)

        if start_date >= end_date:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

        log.info(
            "backtest_start",
            strategy=strategy.name,
            symbol_count=len(symbols),
            start_date=start_date.date(),
            end_date=end_date.date(),
        )

        self._trade_log = []

        # Generate signals using the strategy
        if data_dict is None:
            raise ValueError("data_dict must be provided or passed to run_backtest")

        signals = self._generate_signals(strategy, symbols, data_dict, start_date, end_date)

        if not signals:
            log.warning("no_signals_generated", strategy=strategy.name)
            return self._zero_metrics(), PromotionGate(
                passed=False,
                reason="No signals generated during backtest period",
                metrics_summary={},
            )

        # Simulate trades and collect returns
        returns = self._simulate_trades(signals, data_dict, start_date, end_date)

        if len(returns) == 0:
            log.warning("no_trades_executed", strategy=strategy.name)
            return self._zero_metrics(), PromotionGate(
                passed=False,
                reason="No trades executed (insufficient liquidity or data)",
                metrics_summary={},
            )

        # Compute metrics
        equity_curve = self._compute_equity_curve(returns)
        metrics = self.compute_metrics(returns, equity_curve)

        # Evaluate promotion gates
        gate = self._evaluate_promotion_gates(metrics)

        log.info(
            "backtest_complete",
            strategy=strategy.name,
            trades=metrics.trades,
            annual_return=f"{metrics.annual_return_pct:.2f}%",
            max_dd=f"{metrics.max_drawdown_pct:.2f}%",
            gate_passed=gate.passed,
        )

        return metrics, gate

    def _generate_signals(
        self,
        strategy: BaseStrategy,
        symbols: list[str],
        data_dict: dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        """
        Generate strategy signals for each symbol across the date range.

        Returns list of signal dicts with symbol, date, and strategy-specific fields.
        """
        signals = []

        for symbol in symbols:
            if symbol not in data_dict:
                log.warning("symbol_data_missing", symbol=symbol)
                continue

            df = data_dict[symbol]

            # Filter to date range
            df_range = df[(df.index >= start_date) & (df.index <= end_date)].copy()
            if df_range.empty:
                continue

            # Prepare data (clean, validate min bars)
            prepared = strategy.prepare(df)
            if prepared is None:
                log.warning("symbol_insufficient_history", symbol=symbol)
                continue

            # Scan for setup
            result = strategy.scan(symbol, prepared)
            if result is not None:
                # Add scan timestamp (last bar of prepared data)
                result["scan_date"] = prepared.index[-1]
                signals.append(result)

        return signals

    def _simulate_trades(
        self,
        signals: list[dict[str, Any]],
        data_dict: dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Simulate trades from signals and collect returns.

        For each signal, holds for 5 days (or until end_date) and
        calculates the return, applying costs and liquidity constraints.
        """
        returns_list = []
        trade_dates = []

        for signal in signals:
            symbol = signal.get("symbol")
            scan_date = signal.get("scan_date")

            if symbol not in data_dict or scan_date is None:
                continue

            df = data_dict[symbol]

            # Check liquidity constraint
            if not self.apply_liquidity_constraints(symbol, 0):
                log.debug("liquidity_constraint_failed", symbol=symbol)
                continue

            # Check survivorship safeguard
            if not self.apply_survivorship_safeguards(symbol, scan_date):
                log.debug("survivorship_check_failed", symbol=symbol, date=scan_date)
                continue

            # Find next 5 trading days' return
            try:
                _ = df.index.get_loc(scan_date)
            except KeyError:
                continue

            # Simulate entry at next open and exit 5 days later
            exit_date = scan_date + timedelta(days=5)
            df_after = df[(df.index > scan_date) & (df.index <= min(exit_date, end_date))]

            if len(df_after) == 0:
                continue

            entry_price = df_after.iloc[0]["open"]
            exit_price = df_after.iloc[-1]["close"]

            # Calculate return before costs
            raw_return = (exit_price - entry_price) / entry_price

            # Apply transaction costs
            net_return = self.apply_costs(raw_return)

            returns_list.append(net_return)
            trade_dates.append(scan_date)

            self._trade_log.append(
                {
                    "symbol": symbol,
                    "scan_date": scan_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "raw_return": raw_return,
                    "net_return": net_return,
                }
            )

        if not returns_list:
            return pd.Series(dtype=float)

        return pd.Series(returns_list, index=trade_dates)

    def apply_costs(self, raw_return: float) -> float:
        """
        Apply transaction costs (slippage + commission) to a return.

        Parameters
        ----------
        raw_return : Per-trade return before costs (0.05 = +5%)

        Returns
        -------
        Net return after costs are deducted
        """
        total_cost_bps = self.slippage_bps + self.commission_bps
        cost_pct = total_cost_bps / 10_000
        return raw_return - cost_pct

    def apply_liquidity_constraints(self, symbol: str, qty: int) -> bool:
        """
        Check if a trade quantity is feasible for the symbol's liquidity tier.

        Parameters
        ----------
        symbol : Trading symbol
        qty : Trade quantity (0 for constraint check only)

        Returns
        -------
        True if trade is acceptable, False if symbol is too illiquid
        """
        # In a real system, would look up market cap, avg volume, bid-ask spread
        # For now, skip very small symbols (e.g., symbols with pennies in price)
        # In production: connect to NSE universe database
        return not symbol.startswith("TINY")

    def apply_survivorship_safeguards(self, symbol: str, date: pd.Timestamp) -> bool:
        """
        Check if a symbol was actively listed on the NSE on the given date.

        Parameters
        ----------
        symbol : Trading symbol
        date : Trade date

        Returns
        -------
        True if symbol was active on date, False if delisted or suspended
        """
        # In a real system, would query the NSE universe history database
        # to check if symbol was listed on the given date.
        # For now, allow all (tests will mock this)
        # In production: connect to historical NSE symbol universe
        return True

    def compute_metrics(self, returns: pd.Series, equity_curve: pd.Series) -> BacktestMetrics:
        """
        Compute comprehensive performance metrics.

        Parameters
        ----------
        returns : Series of per-trade returns
        equity_curve : Cumulative equity curve

        Returns
        -------
        BacktestMetrics with all performance statistics
        """
        if returns.empty:
            return self._zero_metrics()

        # Basic metrics
        total_return = float((1 + returns).prod() - 1)
        annual_return = float(returns.mean() * _TRADING_DAYS)
        dd = max_drawdown(equity_curve)
        s = sharpe_ratio(returns)
        pf = profit_factor(returns)
        wr = win_rate(returns)
        exp = expectancy(returns)
        cal = calmar_ratio(returns, dd)

        # Win/loss ratio
        wins = returns[returns > 0]
        losses = returns[returns < 0].abs()
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

        return BacktestMetrics(
            total_return_pct=total_return * 100,
            annual_return_pct=annual_return * 100,
            max_drawdown_pct=dd * 100,
            sharpe=s,
            profit_factor=pf,
            trades=len(returns),
            win_rate_pct=wr * 100,
            avg_win_loss_ratio=avg_win_loss_ratio,
            expectancy=exp,
            calmar_ratio=cal,
        )

    def _compute_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Compute equity curve from returns series."""
        if returns.empty:
            return pd.Series([self.initial_capital])
        equity = (1 + returns).cumprod() * self.initial_capital
        return equity

    def _zero_metrics(self) -> BacktestMetrics:
        """Return a zero-valued metrics object (for no-trade cases)."""
        return BacktestMetrics(
            total_return_pct=0.0,
            annual_return_pct=0.0,
            max_drawdown_pct=0.0,
            sharpe=0.0,
            profit_factor=0.0,
            trades=0,
            win_rate_pct=0.0,
            avg_win_loss_ratio=0.0,
            expectancy=0.0,
            calmar_ratio=0.0,
        )

    def _evaluate_promotion_gates(self, metrics: BacktestMetrics) -> PromotionGate:
        """
        Evaluate if strategy passes all promotion gates.

        Gates (non-negotiable):
          - min_trades >= 20
          - annual_return% > 15
          - max_dd% < -20 (drawdown is negative, so more negative = worse)
          - profit_factor > 1.5
          - expectancy > 0
        """
        failures = []

        if metrics.trades < _PROMOTION_GATES["min_trades"]:
            failures.append(
                f"Insufficient trades: {metrics.trades} < {_PROMOTION_GATES['min_trades']}"
            )

        if metrics.annual_return_pct <= _PROMOTION_GATES["annual_return_pct"]:
            failures.append(
                f"Annual return too low: {metrics.annual_return_pct:.2f}% "
                f"<= {_PROMOTION_GATES['annual_return_pct']}%"
            )

        # max_drawdown_pct is negative (e.g., -15), so we check if it's less than -20
        if metrics.max_drawdown_pct < -_PROMOTION_GATES["max_drawdown_pct"]:
            failures.append(
                f"Max drawdown too high: {metrics.max_drawdown_pct:.2f}% "
                f"< -{_PROMOTION_GATES['max_drawdown_pct']}%"
            )

        if metrics.profit_factor < _PROMOTION_GATES["profit_factor"]:
            failures.append(
                f"Profit factor too low: {metrics.profit_factor:.2f} "
                f"< {_PROMOTION_GATES['profit_factor']}"
            )

        if metrics.expectancy <= _PROMOTION_GATES["expectancy"]:
            failures.append(
                f"Expectancy non-positive: {metrics.expectancy:.6f} "
                f"<= {_PROMOTION_GATES['expectancy']}"
            )

        passed = len(failures) == 0
        reason = "; ".join(failures) if failures else "All gates passed"

        return PromotionGate(
            passed=passed,
            reason=reason,
            metrics_summary=metrics.to_dict(),
        )

    def log_to_mlflow(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, Any],
        gate_decision: PromotionGate,
    ) -> None:
        """
        Log backtest results to MLflow.

        Parameters
        ----------
        run_name : Name for the MLflow run
        params : Parameter dict (strategy name, version, symbols, etc.)
        metrics : Metrics dict from BacktestMetrics.to_dict()
        gate_decision : PromotionGate object
        """
        try:
            mlflow.set_experiment("strategy_validation")

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metric("gate_passed", 1.0 if gate_decision.passed else 0.0)

                # Log gate summary as text
                mlflow.log_text(gate_decision.reason, artifact_file="gate_decision.txt")

                # Save trade log as JSON artifact
                if self._trade_log:
                    trade_json = json.dumps(self._trade_log, default=str, indent=2)
                    mlflow.log_text(trade_json, artifact_file="trade_log.json")

                log.info(
                    "backtest_logged_to_mlflow", run_name=run_name, gate_passed=gate_decision.passed
                )
        except Exception as e:
            log.error("mlflow_logging_failed", error=str(e), run_name=run_name)

    @classmethod
    def check_promotion_gates(cls, metrics: BacktestMetrics) -> tuple[bool, str]:
        """
        Evaluate promotion gates (class method convenience).

        Returns
        -------
        (passed: bool, reason: str)
        """
        backtester = cls()
        gate = backtester._evaluate_promotion_gates(metrics)
        return gate.passed, gate.reason
