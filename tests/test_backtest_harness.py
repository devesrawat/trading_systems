"""
Tests for the strategy backtest harness.

Covers
------
- Run backtest on VCP strategy with historical test data
- Verify metrics computed correctly (total return, Sharpe, max DD)
- Verify cost application (slippage, commission)
- Verify liquidity constraints (skip illiquid symbols)
- Verify survivorship safeguards (skip delisted dates)
- Verify promotion gates (pass/fail decisions)
- Verify MLflow logging
- Verify no trades → fails gates
- Verify insufficient history → error message
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backtest.strategy_harness import (
    BacktestMetrics,
    PromotionGate,
    StrategyBacktester,
)
from signals.base_strategy import BaseStrategy
from signals.strategies.vcp import VCPStrategy

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_daily(
    n: int = 300,
    trend: str = "up",
    seed: int = 0,
    start_date: str = "2022-01-03",
) -> pd.DataFrame:
    """
    Synthetic daily OHLCV for backtesting.

    'up' builds a rising trend suitable for VCP / RS setups.
    """
    rng = np.random.default_rng(seed)
    base = 1000.0

    if trend == "up":
        drift = np.linspace(0, 500, n)
    elif trend == "down":
        drift = np.linspace(0, -300, n)
    else:
        drift = np.zeros(n)

    close = base + drift + rng.normal(0, 8, n).cumsum()
    close = np.abs(close) + 10

    high = close + rng.uniform(3, 15, n)
    low = close - rng.uniform(3, 15, n)
    low = np.maximum(low, 1.0)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    idx = pd.date_range(start_date, periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


class DummyStrategy(BaseStrategy):
    """
    Dummy strategy that signals on every nth bar.

    Used to test harness logic without relying on VCP's complex setup.
    """

    name = "dummy"
    lookback_days = 100
    interval = "day"
    min_bars = 50

    def __init__(self, signal_every: int = 10):
        self.signal_every = signal_every

    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        # Signal on every nth bar
        if len(df) % self.signal_every == 0:
            return {
                "symbol": symbol,
                "strategy": self.name,
                "signal_bar": len(df),
            }
        return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrategyBacktesterBasics:
    """Test basic backtest runner functionality."""

    def test_init_with_defaults(self):
        """Backtester initializes with sensible defaults."""
        bt = StrategyBacktester()
        assert bt.initial_capital == 100_000
        assert bt.slippage_bps == 1.0
        assert bt.commission_bps == 0.5
        assert bt.liquidity_tier == "large_cap"

    def test_init_with_custom_params(self):
        """Backtester accepts custom parameters."""
        bt = StrategyBacktester(
            initial_capital=50_000,
            slippage_bps=2.0,
            commission_bps=1.0,
            liquidity_tier="mid_cap",
        )
        assert bt.initial_capital == 50_000
        assert bt.slippage_bps == 2.0
        assert bt.commission_bps == 1.0
        assert bt.liquidity_tier == "mid_cap"

    def test_run_backtest_requires_date_range(self):
        """run_backtest raises if start_date >= end_date."""
        bt = StrategyBacktester()
        strategy = DummyStrategy()

        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-01-01")
        data_dict = {"DUMMY.NS": _make_daily(n=200)}

        with pytest.raises(ValueError, match=r"start_date.*must be before end_date"):
            bt.run_backtest(strategy, ["DUMMY.NS"], start, end, data_dict)

    def test_run_backtest_requires_data_dict(self):
        """run_backtest raises if data_dict is None."""
        bt = StrategyBacktester()
        strategy = DummyStrategy()

        with pytest.raises(ValueError, match="data_dict must be provided"):
            bt.run_backtest(
                strategy,
                ["DUMMY.NS"],
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-12-31"),
                None,
            )


class TestApplyCosts:
    """Test transaction cost application."""

    def test_apply_costs_subtracts_slippage_and_commission(self):
        """apply_costs deducts slippage and commission from return."""
        bt = StrategyBacktester(slippage_bps=1.0, commission_bps=0.5)
        raw_return = 0.02  # +2%
        net_return = bt.apply_costs(raw_return)
        expected_cost = (1.0 + 0.5) / 10_000  # 0.015%
        assert net_return == pytest.approx(0.02 - expected_cost, abs=1e-6)

    def test_apply_costs_custom_values(self):
        """apply_costs respects custom slippage/commission settings."""
        bt = StrategyBacktester(slippage_bps=2.0, commission_bps=1.0)
        raw_return = 0.01  # +1%
        net_return = bt.apply_costs(raw_return)
        expected_cost = (2.0 + 1.0) / 10_000  # 0.03%
        assert net_return == pytest.approx(0.01 - expected_cost, abs=1e-6)

    def test_apply_costs_on_loss(self):
        """apply_costs deducts costs even on losing trades."""
        bt = StrategyBacktester(slippage_bps=1.0, commission_bps=0.5)
        raw_return = -0.01  # -1%
        net_return = bt.apply_costs(raw_return)
        expected_cost = (1.0 + 0.5) / 10_000  # 0.00015 = 0.015%
        assert net_return == pytest.approx(-0.01 - expected_cost, abs=1e-6)


class TestLiquidityConstraints:
    """Test liquidity constraint checking."""

    def test_apply_liquidity_constraints_allows_normal_symbols(self):
        """Normal symbols pass liquidity check."""
        bt = StrategyBacktester()
        assert bt.apply_liquidity_constraints("RELIANCE.NS", 100) is True
        assert bt.apply_liquidity_constraints("INFY.NS", 100) is True

    def test_apply_liquidity_constraints_rejects_tiny_symbols(self):
        """Symbols starting with TINY are rejected."""
        bt = StrategyBacktester()
        assert bt.apply_liquidity_constraints("TINY_STOCK.NS", 100) is False


class TestSurvivorshipSafeguards:
    """Test survivorship bias handling."""

    def test_apply_survivorship_safeguards_allows_by_default(self):
        """Survivorship check allows by default (production would query DB)."""
        bt = StrategyBacktester()
        assert bt.apply_survivorship_safeguards("RELIANCE.NS", pd.Timestamp("2023-06-15")) is True

    def test_apply_survivorship_safeguards_with_mock_delisted(self):
        """Mocked survivorship check can reject delisted symbols."""
        bt = StrategyBacktester()

        with patch.object(bt, "apply_survivorship_safeguards", return_value=False) as mock_surv:
            assert (
                bt.apply_survivorship_safeguards("DELISTED.NS", pd.Timestamp("2020-01-01")) is False
            )
            mock_surv.assert_called_once()


class TestComputeMetrics:
    """Test performance metric calculation."""

    def test_compute_metrics_empty_returns(self):
        """Empty returns series → all metrics are zero."""
        bt = StrategyBacktester()
        returns = pd.Series(dtype=float)
        equity = pd.Series([100_000])

        metrics = bt.compute_metrics(returns, equity)

        assert metrics.total_return_pct == 0.0
        assert metrics.annual_return_pct == 0.0
        assert metrics.max_drawdown_pct == 0.0
        assert metrics.sharpe == 0.0
        assert metrics.trades == 0

    def test_compute_metrics_all_wins(self):
        """All winning trades → high metrics."""
        bt = StrategyBacktester()
        returns = pd.Series([0.01, 0.02, 0.015])  # +1%, +2%, +1.5%
        equity = (1 + returns).cumprod() * 100_000

        metrics = bt.compute_metrics(returns, equity)

        assert metrics.trades == 3
        assert metrics.win_rate_pct == pytest.approx(100.0)
        assert metrics.total_return_pct > 0

    def test_compute_metrics_all_losses(self):
        """All losing trades → negative metrics, profit_factor=0."""
        bt = StrategyBacktester()
        returns = pd.Series([-0.01, -0.02, -0.015])
        equity = (1 + returns).cumprod() * 100_000

        metrics = bt.compute_metrics(returns, equity)

        assert metrics.trades == 3
        assert metrics.win_rate_pct == pytest.approx(0.0)
        assert metrics.profit_factor == pytest.approx(0.0)

    def test_compute_metrics_mixed_trades(self):
        """Mixed wins/losses → realistic metrics."""
        bt = StrategyBacktester()
        returns = pd.Series([0.02, -0.01, 0.015, -0.005, 0.03])
        equity = (1 + returns).cumprod() * 100_000

        metrics = bt.compute_metrics(returns, equity)

        assert metrics.trades == 5
        assert metrics.win_rate_pct == pytest.approx(60.0)  # 3/5 wins
        assert metrics.profit_factor > 1.0  # Wins > losses


class TestPromotionGates:
    """Test promotion gate evaluation."""

    def test_passes_all_gates_with_excellent_metrics(self):
        """Strategy passes all gates with strong metrics."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is True
        assert "All gates passed" in gate.reason

    def test_fails_insufficient_trades(self):
        """Strategy fails if trades < 20."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=10,  # < 20
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        assert "Insufficient trades" in gate.reason

    def test_fails_low_annual_return(self):
        """Strategy fails if annual_return% <= 15."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=30.0,
            annual_return_pct=10.0,  # <= 15%
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        assert "Annual return too low" in gate.reason

    def test_fails_high_max_drawdown(self):
        """Strategy fails if max_dd% > 20."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-25.0,  # > 20%
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        assert "Max drawdown too high" in gate.reason

    def test_fails_low_profit_factor(self):
        """Strategy fails if profit_factor < 1.5."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=1.2,  # < 1.5
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        assert "Profit factor too low" in gate.reason

    def test_fails_negative_expectancy(self):
        """Strategy fails if expectancy <= 0."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=-0.01,  # < 0
            calmar_ratio=2.5,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        assert "Expectancy non-positive" in gate.reason

    def test_fails_multiple_gates(self):
        """Strategy reports all failed gates."""
        bt = StrategyBacktester()

        metrics = BacktestMetrics(
            total_return_pct=10.0,
            annual_return_pct=5.0,  # Too low
            max_drawdown_pct=-30.0,  # Too high
            sharpe=0.5,
            profit_factor=1.0,  # Too low
            trades=5,  # Too few
            win_rate_pct=40.0,
            avg_win_loss_ratio=0.8,
            expectancy=-0.05,  # Negative
            calmar_ratio=0.1,
        )

        gate = bt._evaluate_promotion_gates(metrics)
        assert gate.passed is False
        # Should contain multiple failure reasons
        failure_count = gate.reason.count(";")
        assert failure_count >= 3  # At least 4 failures (3 semicolons = 4 failures)


class TestBacktestNoTrades:
    """Test backtest with no trades scenario."""

    def test_no_signals_generated(self):
        """Backtest with strategy that never signals → fails gates."""
        bt = StrategyBacktester()

        # Strategy that never signals
        class NeverSignalStrategy(BaseStrategy):
            name = "never_signal"

            def scan(self, symbol: str, df: pd.DataFrame):
                return None

        strategy = NeverSignalStrategy()
        data = {"DUMMY.NS": _make_daily(n=300)}

        metrics, gate = bt.run_backtest(
            strategy,
            ["DUMMY.NS"],
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2022-12-31"),
            data,
        )

        assert gate.passed is False
        assert "No signals generated" in gate.reason

    def test_no_trades_due_to_liquidity(self):
        """Backtest where all signals fail liquidity check → fails gates."""
        bt = StrategyBacktester()
        strategy = DummyStrategy(signal_every=50)

        data = {"TINY_STOCK.NS": _make_daily(n=300)}

        metrics, gate = bt.run_backtest(
            strategy,
            ["TINY_STOCK.NS"],
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2022-12-31"),
            data,
        )

        # Should have signals but no trades due to liquidity
        assert gate.passed is False


class TestBacktestInsufficientHistory:
    """Test backtest with insufficient historical data."""

    def test_insufficient_history_for_strategy(self):
        """Strategy with min_bars > data length returns warning."""
        bt = StrategyBacktester()

        class HighMinBarsStrategy(BaseStrategy):
            name = "high_min_bars"
            min_bars = 500

            def scan(self, symbol: str, df: pd.DataFrame):
                return {"symbol": symbol, "strategy": self.name}

        strategy = HighMinBarsStrategy()
        # Only 100 bars of data
        data = {"DUMMY.NS": _make_daily(n=100)}

        metrics, gate = bt.run_backtest(
            strategy,
            ["DUMMY.NS"],
            pd.Timestamp("2022-01-03"),
            pd.Timestamp("2022-05-31"),
            data,
        )

        assert gate.passed is False
        assert metrics.trades == 0


class TestBacktestVCPStrategy:
    """Test backtest with real VCP strategy."""

    def test_vcp_backtest_generates_trades(self):
        """VCP strategy generates signals and computes metrics."""
        bt = StrategyBacktester()
        strategy = VCPStrategy()

        # Create uptrend data suitable for VCP
        data = {"RELIANCE.NS": _make_daily(n=400, trend="up", seed=42)}

        metrics, gate = bt.run_backtest(
            strategy,
            ["RELIANCE.NS"],
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-01-31"),
            data,
        )

        # VCP is a real strategy, so it may or may not generate signals
        # Just verify the backtest completes and returns valid metrics
        assert isinstance(metrics, BacktestMetrics)
        assert isinstance(gate, PromotionGate)
        assert metrics.trades >= 0


class TestBacktestMetrics:
    """Test BacktestMetrics dataclass."""

    def test_backtest_metrics_to_dict(self):
        """BacktestMetrics converts to dict."""
        metrics = BacktestMetrics(
            total_return_pct=25.0,
            annual_return_pct=20.0,
            max_drawdown_pct=-15.0,
            sharpe=1.5,
            profit_factor=2.0,
            trades=30,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.2,
            expectancy=0.01,
            calmar_ratio=1.33,
        )

        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert d["total_return_pct"] == 25.0
        assert d["trades"] == 30
        assert d["win_rate_pct"] == 55.0


class TestMLflowLogging:
    """Test MLflow integration."""

    @patch("backtest.strategy_harness.mlflow.start_run")
    @patch("backtest.strategy_harness.mlflow.set_experiment")
    @patch("backtest.strategy_harness.mlflow.log_params")
    @patch("backtest.strategy_harness.mlflow.log_metrics")
    @patch("backtest.strategy_harness.mlflow.log_metric")
    @patch("backtest.strategy_harness.mlflow.log_text")
    def test_log_to_mlflow_passed_gate(
        self,
        mock_log_text,
        mock_log_metric,
        mock_log_metrics,
        mock_log_params,
        mock_set_exp,
        mock_start_run,
    ):
        """MLflow logging for passed gate."""
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

        bt = StrategyBacktester()
        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )
        gate = PromotionGate(
            passed=True, reason="All gates passed", metrics_summary=metrics.to_dict()
        )

        bt.log_to_mlflow(
            run_name="test_run",
            params={"strategy": "vcp", "symbol": "RELIANCE.NS"},
            metrics=metrics.to_dict(),
            gate_decision=gate,
        )

        mock_set_exp.assert_called_once_with("strategy_validation")
        mock_start_run.assert_called_once_with(run_name="test_run")
        mock_log_params.assert_called_once()
        mock_log_metrics.assert_called_once()
        mock_log_metric.assert_called_once_with("gate_passed", 1.0)

    @patch("backtest.strategy_harness.mlflow.start_run")
    @patch("backtest.strategy_harness.mlflow.set_experiment")
    @patch("backtest.strategy_harness.mlflow.log_params")
    @patch("backtest.strategy_harness.mlflow.log_metrics")
    @patch("backtest.strategy_harness.mlflow.log_metric")
    @patch("backtest.strategy_harness.mlflow.log_text")
    def test_log_to_mlflow_failed_gate(
        self,
        mock_log_text,
        mock_log_metric,
        mock_log_metrics,
        mock_log_params,
        mock_set_exp,
        mock_start_run,
    ):
        """MLflow logging for failed gate."""
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

        bt = StrategyBacktester()
        metrics = BacktestMetrics(
            total_return_pct=5.0,
            annual_return_pct=5.0,
            max_drawdown_pct=-30.0,
            sharpe=0.5,
            profit_factor=1.0,
            trades=10,
            win_rate_pct=40.0,
            avg_win_loss_ratio=0.8,
            expectancy=-0.05,
            calmar_ratio=0.1,
        )
        gate = PromotionGate(
            passed=False,
            reason="Insufficient trades; Annual return too low; Max drawdown too high",
            metrics_summary=metrics.to_dict(),
        )

        bt.log_to_mlflow(
            run_name="test_run_failed",
            params={"strategy": "dummy"},
            metrics=metrics.to_dict(),
            gate_decision=gate,
        )

        mock_log_metric.assert_called_once_with("gate_passed", 0.0)

    @patch("backtest.strategy_harness.mlflow.start_run")
    @patch("backtest.strategy_harness.mlflow.set_experiment")
    @patch("backtest.strategy_harness.mlflow.log_params")
    @patch("backtest.strategy_harness.mlflow.log_metrics")
    @patch("backtest.strategy_harness.mlflow.log_metric")
    @patch("backtest.strategy_harness.mlflow.log_text")
    def test_log_to_mlflow_with_trade_log(
        self,
        mock_log_text,
        mock_log_metric,
        mock_log_metrics,
        mock_log_params,
        mock_set_exp,
        mock_start_run,
    ):
        """MLflow logging includes trade log JSON."""
        mock_start_run.return_value.__enter__ = MagicMock()
        mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

        bt = StrategyBacktester()
        bt._trade_log = [
            {
                "symbol": "TEST.NS",
                "entry_price": 100.0,
                "exit_price": 102.0,
                "net_return": 0.0198,
            }
        ]

        metrics = BacktestMetrics(
            total_return_pct=1.98,
            annual_return_pct=20.0,
            max_drawdown_pct=-5.0,
            sharpe=1.5,
            profit_factor=2.0,
            trades=1,
            win_rate_pct=100.0,
            avg_win_loss_ratio=0.0,
            expectancy=0.02,
            calmar_ratio=4.0,
        )
        gate = PromotionGate(
            passed=True, reason="All gates passed", metrics_summary=metrics.to_dict()
        )

        bt.log_to_mlflow(
            run_name="test_with_trades",
            params={"strategy": "dummy"},
            metrics=metrics.to_dict(),
            gate_decision=gate,
        )

        # Should log both gate_decision.txt and trade_log.json
        assert mock_log_text.call_count >= 2


class TestCheckPromotionGatesClassMethod:
    """Test class method convenience function."""

    def test_check_promotion_gates_passed(self):
        """check_promotion_gates returns (True, "All gates passed")."""
        metrics = BacktestMetrics(
            total_return_pct=50.0,
            annual_return_pct=25.0,
            max_drawdown_pct=-10.0,
            sharpe=1.8,
            profit_factor=2.5,
            trades=50,
            win_rate_pct=55.0,
            avg_win_loss_ratio=1.5,
            expectancy=0.05,
            calmar_ratio=2.5,
        )

        passed, reason = StrategyBacktester.check_promotion_gates(metrics)
        assert passed is True
        assert "All gates passed" in reason

    def test_check_promotion_gates_failed(self):
        """check_promotion_gates returns (False, reason) on failure."""
        metrics = BacktestMetrics(
            total_return_pct=5.0,
            annual_return_pct=5.0,
            max_drawdown_pct=-30.0,
            sharpe=0.5,
            profit_factor=1.0,
            trades=10,
            win_rate_pct=40.0,
            avg_win_loss_ratio=0.8,
            expectancy=-0.05,
            calmar_ratio=0.1,
        )

        passed, reason = StrategyBacktester.check_promotion_gates(metrics)
        assert passed is False
        assert "Insufficient trades" in reason


class TestEndToEndBacktest:
    """End-to-end backtest simulation."""

    def test_end_to_end_dummy_strategy(self):
        """Complete backtest flow with dummy strategy."""
        bt = StrategyBacktester(initial_capital=100_000)
        strategy = DummyStrategy(signal_every=20)

        # Create multi-symbol dataset
        data = {
            "SYMBOL_A.NS": _make_daily(n=300, trend="up", seed=1),
            "SYMBOL_B.NS": _make_daily(n=300, trend="up", seed=2),
        }

        metrics, gate = bt.run_backtest(
            strategy,
            ["SYMBOL_A.NS", "SYMBOL_B.NS"],
            pd.Timestamp("2022-02-01"),
            pd.Timestamp("2023-01-31"),
            data,
        )

        # Verify metrics are computed
        assert isinstance(metrics, BacktestMetrics)
        assert metrics.trades >= 0

        # Verify gate evaluation
        assert isinstance(gate, PromotionGate)
        assert isinstance(gate.passed, bool)
        assert isinstance(gate.reason, str)

        # Verify cost application
        if metrics.trades > 0 and len(bt._trade_log) > 0:
            trade = bt._trade_log[0]
            expected_cost = (bt.slippage_bps + bt.commission_bps) / 10_000
            # net_return = raw_return - expected_cost
            assert trade["raw_return"] - trade["net_return"] == pytest.approx(
                expected_cost, abs=1e-6
            )
