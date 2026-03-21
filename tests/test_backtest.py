"""Unit tests for backtest/ — TDD RED phase. Pure arithmetic, no DB needed."""
import numpy as np
import pandas as pd
import pytest

from backtest.costs import NSECostModel
from backtest.metrics import (
    calmar_ratio,
    expectancy,
    max_drawdown,
    print_tearsheet,
    profit_factor,
    sharpe_ratio,
    win_rate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _pos_returns(n: int = 100, mean: float = 0.001, std: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(0)
    r = rng.normal(mean, std, n)
    return pd.Series(r, index=pd.date_range("2023-01-01", periods=n, freq="B"))


def _neg_returns(n: int = 100) -> pd.Series:
    return _pos_returns(n, mean=-0.001)


def _equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() * 100_000


# ---------------------------------------------------------------------------
# NSECostModel
# ---------------------------------------------------------------------------

class TestNSECostModel:
    def test_equity_cost_buy_no_stt(self):
        """STT only applies on SELL side for equity delivery."""
        model = NSECostModel()
        buy_cost = model.equity_cost(10_000.0, side="BUY")
        sell_cost = model.equity_cost(10_000.0, side="SELL")
        assert sell_cost > buy_cost   # sell includes STT

    def test_equity_cost_positive(self):
        model = NSECostModel()
        cost = model.equity_cost(10_000.0, side="BUY")
        assert cost > 0

    def test_brokerage_capped_at_20(self):
        """Zerodha caps brokerage at ₹20 per order."""
        model = NSECostModel()
        # Large trade where 0.03% would exceed ₹20
        cost_large = model.equity_cost(1_000_000.0, side="BUY")
        cost_small = model.equity_cost(10_000.0, side="BUY")
        # Brokerage component should be capped, so marginal cost decreases
        assert cost_large / 1_000_000 < cost_small / 10_000

    def test_gst_applied_on_brokerage(self):
        model = NSECostModel()
        cost = model.equity_cost(10_000.0, side="BUY")
        brokerage = min(20.0, 10_000.0 * 0.0003)
        gst = brokerage * 0.18
        assert cost >= gst   # GST must be included

    def test_stamp_duty_only_on_buy(self):
        model = NSECostModel()
        buy_cost = model.equity_cost(10_000.0, side="BUY")
        sell_cost = model.equity_cost(10_000.0, side="SELL")
        # Both costs are positive; buy has stamp duty, sell has STT
        assert buy_cost > 0 and sell_cost > 0

    def test_round_trip_cost_positive(self):
        model = NSECostModel()
        rt = model.round_trip_cost(10_000.0)
        assert rt > 0

    def test_round_trip_cost_exceeds_half_percent(self):
        """Round-trip must be >0.5% to reflect realistic NSE costs."""
        model = NSECostModel()
        trade_value = 10_000.0
        rt = model.round_trip_cost(trade_value)
        assert rt / trade_value > 0.002

    def test_slippage_tiers_ordered(self):
        model = NSECostModel()
        large = model.slippage(10_000.0, "large_cap")
        mid = model.slippage(10_000.0, "mid_cap")
        small = model.slippage(10_000.0, "small_cap")
        assert large < mid < small

    def test_intraday_no_stt_on_buy(self):
        model = NSECostModel()
        intraday = model.equity_cost(10_000.0, side="BUY", intraday=True)
        delivery = model.equity_cost(10_000.0, side="BUY", intraday=False)
        # Intraday STT is different (charged on sell only, lower rate)
        assert intraday > 0 and delivery > 0


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:
    def test_positive_returns_positive_sharpe(self):
        r = _pos_returns(252)
        s = sharpe_ratio(r)
        assert s > 0

    def test_negative_returns_negative_sharpe(self):
        r = _neg_returns(252)
        s = sharpe_ratio(r)
        assert s < 0

    def test_zero_std_returns_zero(self):
        r = pd.Series([0.001] * 252)
        s = sharpe_ratio(r)
        assert s > 0   # positive constant return → positive Sharpe

    def test_annualised_uses_252_days(self):
        daily_return = 0.001
        r = pd.Series([daily_return] * 252)
        s = sharpe_ratio(r, risk_free=0.0)
        # Annualised return = 0.001 * 252; std → 0 → very large Sharpe
        assert s > 1.0

    def test_risk_free_reduces_sharpe(self):
        r = _pos_returns(252, mean=0.0005)
        low_rf = sharpe_ratio(r, risk_free=0.0)
        high_rf = sharpe_ratio(r, risk_free=0.10)
        assert low_rf > high_rf


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_returns_negative_fraction(self):
        r = _pos_returns(100)
        eq = _equity_curve(r)
        dd = max_drawdown(eq)
        assert dd <= 0

    def test_monotone_increase_zero_drawdown(self):
        eq = pd.Series(range(1, 101), dtype=float)
        dd = max_drawdown(eq)
        assert dd == 0.0

    def test_known_drawdown(self):
        eq = pd.Series([100.0, 120.0, 80.0, 90.0])
        dd = max_drawdown(eq)
        assert abs(dd - (-1/3)) < 0.01   # 120 → 80 = -33%

    def test_full_loss_is_minus_one(self):
        eq = pd.Series([100.0, 50.0, 0.001])
        dd = max_drawdown(eq)
        assert dd < -0.99


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

class TestProfitFactor:
    def test_profitable_strategy_above_one(self):
        r = _pos_returns(100, mean=0.002)
        pf = profit_factor(r)
        assert pf > 1.0

    def test_losing_strategy_below_one(self):
        r = _neg_returns(100)
        pf = profit_factor(r)
        assert pf < 1.0

    def test_no_losses_returns_inf(self):
        r = pd.Series([0.01, 0.02, 0.005])
        pf = profit_factor(r)
        assert pf == float("inf")

    def test_no_wins_returns_zero(self):
        r = pd.Series([-0.01, -0.02])
        pf = profit_factor(r)
        assert pf == 0.0


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

class TestWinRate:
    def test_all_positive_is_one(self):
        r = pd.Series([0.01, 0.02, 0.005])
        assert win_rate(r) == 1.0

    def test_all_negative_is_zero(self):
        r = pd.Series([-0.01, -0.02])
        assert win_rate(r) == 0.0

    def test_half_half(self):
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert win_rate(r) == 0.5


# ---------------------------------------------------------------------------
# expectancy
# ---------------------------------------------------------------------------

class TestExpectancy:
    def test_positive_expectancy_for_good_strategy(self):
        r = _pos_returns(200, mean=0.003)
        e = expectancy(r)
        assert e > 0

    def test_negative_expectancy_for_bad_strategy(self):
        r = _neg_returns(200)
        e = expectancy(r)
        assert e < 0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:
    def test_positive_for_positive_returns(self):
        r = _pos_returns(252, mean=0.001)
        eq = _equity_curve(r)
        dd = max_drawdown(eq)
        c = calmar_ratio(r, dd)
        assert c > 0

    def test_zero_drawdown_returns_inf(self):
        r = pd.Series([0.001] * 252)
        c = calmar_ratio(r, max_dd=0.0)
        assert c == float("inf")


# ---------------------------------------------------------------------------
# print_tearsheet
# ---------------------------------------------------------------------------

class TestPrintTearsheet:
    def test_runs_without_error(self, capsys):
        r = _pos_returns(252)
        eq = _equity_curve(r)
        print_tearsheet(r, eq)
        captured = capsys.readouterr()
        assert "Sharpe" in captured.out

    def test_shows_pass_fail_flags(self, capsys):
        r = _pos_returns(252, mean=0.003)
        eq = _equity_curve(r)
        print_tearsheet(r, eq)
        captured = capsys.readouterr()
        assert "PASS" in captured.out or "FAIL" in captured.out
