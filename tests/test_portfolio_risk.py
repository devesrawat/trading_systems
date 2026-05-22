"""
Unit tests for portfolio-level risk management.

Tests cover exposure calculations, correlation penalties, liquidity checks,
turnover constraints, and integrated risk decisions.
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from portfolio.correlation import (
    apply_correlation_penalty,
    compute_pairwise_correlation,
    compute_portfolio_correlation,
    is_redundant_position,
)
from portfolio.exposure import (
    compute_sector_exposure,
    exposure_adjusted_capital_for_signal,
    get_sector_for_symbol,
    is_over_sector_limit,
)
from portfolio.limits import (
    CRYPTO_LIMITS,
    LIVE_TRADING_LIMITS,
    PAPER_TRADE_LIMITS,
    get_limits_for_mode,
)
from portfolio.liquidity import (
    check_minimum_liquidity,
    compute_liquidity_score,
    portfolio_liquidity_stress_test,
)
from portfolio.risk_manager import PreExecutionRiskCheck
from portfolio.schema import (
    PortfolioPosition,
    PortfolioState,
    RiskLimits,
)
from portfolio.turnover import (
    check_turnover_limit,
    compute_turnover,
    estimate_portfolio_turnover,
)
from signals.contracts import Direction, EntrySpec, RiskSpec, Signal, SignalType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def empty_portfolio():
    """Empty portfolio with 500k capital."""
    return PortfolioState(
        positions={},
        total_capital=500_000.0,
        cash_available=500_000.0,
    )


@pytest.fixture
def populated_portfolio():
    """Portfolio with 2 positions (INFY in IT, RELIANCE in Energy)."""
    return PortfolioState(
        positions={
            "INFY": PortfolioPosition(
                symbol="INFY",
                qty=100.0,
                entry_price=1500.0,
                current_price=1600.0,
            ),
            "RELIANCE": PortfolioPosition(
                symbol="RELIANCE",
                qty=50.0,
                entry_price=2500.0,
                current_price=2600.0,
            ),
        },
        total_capital=500_000.0,
        cash_available=450_000.0,
    )


@pytest.fixture
def sample_signal():
    """Sample ML prediction signal."""
    return Signal(
        signal_id="sig-001",
        timestamp=datetime.utcnow(),
        symbol="TCS",
        exchange="NSE",
        asset_class="equity",
        strategy_name="ml_long",
        strategy_version="1.0",
        signal_type=SignalType.ml_prediction,
        direction=Direction.long,
        confidence=0.75,
        score=0.72,
        rank=5,
        timeframe="daily",
        entry=EntrySpec(
            entry_price=3500.0,
            stop_price=3400.0,
            target_price=3700.0,
        ),
        risk=RiskSpec(
            size_hint_pct=0.01,
            liquidity_score=0.8,
            volatility_score=0.5,
        ),
        mode="paper",
    )


# ============================================================================
# Sector / Exposure Tests
# ============================================================================


class TestSectorExposure:
    def test_get_sector_for_symbol_known(self):
        """Known symbol returns correct sector."""
        assert get_sector_for_symbol("INFY") == "IT"
        assert get_sector_for_symbol("RELIANCE") == "Energy"
        assert get_sector_for_symbol("SBIN") == "Banking"

    def test_get_sector_for_symbol_unknown(self):
        """Unknown symbol returns 'Other'."""
        assert get_sector_for_symbol("UNKNOWN") == "Other"

    def test_get_sector_for_symbol_strips_suffix(self):
        """Handles .NS and .BO suffixes."""
        assert get_sector_for_symbol("INFY.NS") == "IT"
        assert get_sector_for_symbol("INFY.BO") == "IT"

    def test_compute_sector_exposure_empty_portfolio(self, empty_portfolio):
        """Empty portfolio has no exposures."""
        exposures = compute_sector_exposure(empty_portfolio)
        assert len(exposures) == 0

    def test_compute_sector_exposure_single_position(self, empty_portfolio):
        """Single position computes correctly."""
        empty_portfolio.positions["INFY"] = PortfolioPosition(
            symbol="INFY",
            qty=100.0,
            entry_price=1500.0,
            current_price=1600.0,
        )
        exposures = compute_sector_exposure(empty_portfolio)
        assert "IT" in exposures
        assert exposures["IT"].rank == 1
        assert 0 < exposures["IT"].pct_of_capital < 1

    def test_compute_sector_exposure_multiple_sectors(self, populated_portfolio):
        """Multiple positions compute sector rankings."""
        exposures = compute_sector_exposure(populated_portfolio)
        assert len(exposures) >= 2
        # INFY (160k) > RELIANCE (130k) → IT rank 1, Energy rank 2
        assert exposures["IT"].rank < exposures["Energy"].rank

    def test_is_over_sector_limit_allows_room(self, empty_portfolio):
        """Sector limit allows position with room."""
        exposures = compute_sector_exposure(empty_portfolio)
        is_over = is_over_sector_limit(
            "IT",
            new_qty=10.0,
            new_price=3500.0,
            current_exposure=exposures,
            total_capital=500_000.0,
            max_sector_pct=0.25,
        )
        assert not is_over

    def test_is_over_sector_limit_rejects_excess(self, populated_portfolio):
        """Sector limit rejects when exceeded."""
        exposures = compute_sector_exposure(populated_portfolio)
        # Current IT exposure is ~32k / 500k = 6.4%
        # Adding 130k more would be ~26.4%, still under 25% limit? No, over.
        is_over = is_over_sector_limit(
            "IT",
            new_qty=400.0,
            new_price=3500.0,
            current_exposure=exposures,
            total_capital=500_000.0,
            max_sector_pct=0.25,
        )
        assert is_over  # 160k + 1.4M = 1.56M > 125k (25% of 500k)

    def test_exposure_adjusted_capital_for_signal_no_room(self, populated_portfolio):
        """Signal capital reduced when sector full."""
        exposures = compute_sector_exposure(populated_portfolio)
        allowed = exposure_adjusted_capital_for_signal(
            "INFY",
            qty=500.0,
            price=1600.0,
            current_exposure=exposures,
            total_capital=500_000.0,
            max_sector_pct=0.06,  # Very tight limit
        )
        assert allowed < 500.0 * 1600.0


# ============================================================================
# Correlation Tests
# ============================================================================


class TestCorrelation:
    def test_compute_pairwise_correlation_stub(self):
        """Pairwise correlation returns stub value."""
        corr = compute_pairwise_correlation("INFY", "TCS", lookback_days=252)
        assert -1.0 <= corr <= 1.0

    def test_compute_portfolio_correlation_empty_portfolio(self):
        """Empty portfolio has 0 correlation."""
        avg_corr, max_pair = compute_portfolio_correlation("INFY", {})
        assert avg_corr == 0.0
        assert max_pair is None

    def test_compute_portfolio_correlation_with_positions(self):
        """Portfolio correlation computes average."""
        positions = {
            "RELIANCE": PortfolioPosition(
                symbol="RELIANCE", qty=50, entry_price=2500, current_price=2600
            ),
            "TCS": PortfolioPosition(symbol="TCS", qty=100, entry_price=3500, current_price=3600),
        }
        avg_corr, max_pair = compute_portfolio_correlation("INFY", positions, lookback_days=252)
        assert 0 <= avg_corr <= 1.0
        assert max_pair is not None

    def test_apply_correlation_penalty_low_correlation(self):
        """Low correlation (< 0.3) has no penalty."""
        adjusted = apply_correlation_penalty(confidence=0.8, avg_correlation=0.2)
        assert adjusted == 0.8  # No penalty

    def test_apply_correlation_penalty_medium_correlation(self):
        """Medium correlation (0.3-0.6) applies linear penalty."""
        adjusted = apply_correlation_penalty(confidence=0.8, avg_correlation=0.45)
        assert 0 < adjusted < 0.8  # Penalty applied
        assert adjusted > 0.4  # But not too much

    def test_apply_correlation_penalty_high_correlation(self):
        """High correlation (> 0.6) applies strong penalty."""
        adjusted = apply_correlation_penalty(confidence=0.8, avg_correlation=0.8)
        assert adjusted < 0.5  # Strong penalty (0.5x)

    def test_is_redundant_position_empty_portfolio(self):
        """Empty portfolio has no redundant positions."""
        assert not is_redundant_position("INFY", {})

    def test_is_redundant_position_high_correlation(self):
        """High-correlation symbol is redundant."""
        positions = {
            "TCS": PortfolioPosition(symbol="TCS", qty=100, entry_price=3500, current_price=3600),
        }
        # Mock correlation to be high
        with patch("portfolio.correlation.compute_portfolio_correlation") as mock_corr:
            mock_corr.return_value = (0.8, ("INFY", "TCS"))
            is_red = is_redundant_position("INFY", positions, threshold=0.6)
            assert is_red


# ============================================================================
# Liquidity Tests
# ============================================================================


class TestLiquidity:
    def test_compute_liquidity_score_high_volume(self):
        """High volume → high liquidity score."""
        score = compute_liquidity_score("RELIANCE", avg_volume=5_000_000, bid_ask_spread_pct=0.01)
        assert score > 80  # Very liquid

    def test_compute_liquidity_score_low_volume(self):
        """Low volume → low liquidity score."""
        score = compute_liquidity_score("MICROCAP", avg_volume=10_000, bid_ask_spread_pct=0.05)
        assert score < 40  # Illiquid

    def test_compute_liquidity_score_wide_spread(self):
        """Wide spread → lower score."""
        score_tight = compute_liquidity_score("INFY", avg_volume=1_000_000, bid_ask_spread_pct=0.01)
        score_wide = compute_liquidity_score("INFY", avg_volume=1_000_000, bid_ask_spread_pct=0.10)
        assert score_wide < score_tight

    def test_check_minimum_liquidity_acceptable(self):
        """Small position passes liquidity check."""
        allowed, slippage = check_minimum_liquidity(
            "INFY", qty=100, current_price=3500, bid_ask_impact_bps=10
        )
        assert allowed is True
        assert slippage == 10

    def test_check_minimum_liquidity_excessive_slippage(self):
        """Position with excessive slippage is rejected."""
        allowed, slippage = check_minimum_liquidity(
            "MICROCAP", qty=100_000, current_price=50, bid_ask_impact_bps=75
        )
        assert allowed is False
        assert slippage == 75

    def test_portfolio_liquidity_stress_test_normal(self):
        """Normal conditions allow most liquidation."""
        positions = {
            "INFY": PortfolioPosition(symbol="INFY", qty=100, entry_price=1500, current_price=1600),
        }
        same_day, five_day = portfolio_liquidity_stress_test(positions, market_condition="normal")
        assert same_day >= 0.7
        assert five_day >= 0.9

    def test_portfolio_liquidity_stress_test_crisis(self):
        """Crisis conditions reduce liquidation estimates."""
        positions = {
            "INFY": PortfolioPosition(symbol="INFY", qty=100, entry_price=1500, current_price=1600),
        }
        same_day, five_day = portfolio_liquidity_stress_test(positions, market_condition="crisis")
        assert same_day < 0.5
        assert five_day < 0.6


# ============================================================================
# Turnover Tests
# ============================================================================


class TestTurnover:
    def test_compute_turnover_returns_positive(self):
        """Turnover computation returns positive value."""
        turnover = compute_turnover(qty=100, price=3500, period_days=252)
        assert turnover > 0

    def test_estimate_portfolio_turnover_empty_portfolio(self):
        """Empty portfolio has minimal estimated turnover."""
        estimated = estimate_portfolio_turnover(
            {}, new_position_notional=50_000, total_capital=500_000
        )
        assert 0 <= estimated < 0.05  # Should be small

    def test_estimate_portfolio_turnover_with_position(self):
        """Portfolio turnover scales with position size."""
        positions = {
            "INFY": PortfolioPosition(symbol="INFY", qty=100, entry_price=1500, current_price=1600),
        }
        estimated = estimate_portfolio_turnover(
            positions, new_position_notional=50_000, total_capital=500_000
        )
        assert 0 <= estimated <= 0.5

    def test_check_turnover_limit_allows_below_limit(self):
        """Turnover below limit is allowed."""
        allowed, reason = check_turnover_limit(
            estimated_turnover_pct=1.0, max_turnover_pct_annual=2.0
        )
        assert allowed is True

    def test_check_turnover_limit_rejects_above_limit(self):
        """Turnover above limit is rejected."""
        allowed, reason = check_turnover_limit(
            estimated_turnover_pct=3.0, max_turnover_pct_annual=2.0
        )
        assert allowed is False
        assert "exceeds" in reason


# ============================================================================
# Risk Limits Tests
# ============================================================================


class TestRiskLimits:
    def test_default_equity_limits(self):
        """Default equity limits are reasonable."""
        limits = get_limits_for_mode("paper")
        assert limits.max_sector_pct == 0.15
        assert limits.max_single_stock_pct == 0.015

    def test_crypto_limits_tighter_than_equity(self):
        """Crypto limits are tighter."""
        assert CRYPTO_LIMITS.max_sector_pct < LIVE_TRADING_LIMITS.max_sector_pct
        assert CRYPTO_LIMITS.max_single_stock_pct < LIVE_TRADING_LIMITS.max_single_stock_pct

    def test_live_limits_looser_than_paper(self):
        """Live limits are more generous after validation."""
        assert LIVE_TRADING_LIMITS.max_sector_pct >= PAPER_TRADE_LIMITS.max_sector_pct

    def test_hard_cap_validator_rejects_excess(self):
        """Hard cap validator rejects > 2% single stock."""
        with pytest.raises(ValueError):
            RiskLimits(max_single_stock_pct=0.03)

    def test_sector_min_validator(self):
        """Sector limit must be >= single stock limit."""
        with pytest.raises(ValueError):
            RiskLimits(max_sector_pct=0.01, max_single_stock_pct=0.02)

    def test_limits_for_mode_research(self):
        """Research mode uses paper-trade limits."""
        limits = get_limits_for_mode("research")
        assert limits.max_sector_pct == PAPER_TRADE_LIMITS.max_sector_pct

    def test_limits_for_mode_live(self):
        """Live mode uses live-trading limits."""
        limits = get_limits_for_mode("live")
        assert limits.max_sector_pct == LIVE_TRADING_LIMITS.max_sector_pct


# ============================================================================
# Pre-Execution Risk Check (Integration Tests)
# ============================================================================


class TestPreExecutionRiskCheck:
    def test_check_signal_execution_empty_portfolio(self, empty_portfolio, sample_signal):
        """Signal passes checks in empty portfolio."""
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(sample_signal, empty_portfolio)
        assert decision.allowed is True
        assert decision.capital_allocated > 0
        assert len(decision.checks_passed) > 0

    def test_check_signal_execution_populates_capital_allocated(
        self, empty_portfolio, sample_signal
    ):
        """Capital allocated matches signal risk spec."""
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(sample_signal, empty_portfolio)
        if sample_signal.risk and sample_signal.risk.size_hint_pct:
            expected = sample_signal.risk.size_hint_pct * empty_portfolio.total_capital
            assert 0 < decision.capital_allocated <= expected

    def test_check_signal_execution_position_count_limit(self, empty_portfolio, sample_signal):
        """Rejects when max positions reached."""
        # Create portfolio with max_positions - 1 positions
        tight_limits = RiskLimits(
            max_sector_pct=0.25,
            max_single_stock_pct=0.02,
            max_positions=1,  # Only 1 position allowed
        )
        # Add 1 position to portfolio
        empty_portfolio.positions["INFY"] = PortfolioPosition(
            symbol="INFY", qty=100, entry_price=1500, current_price=1600
        )
        checker = PreExecutionRiskCheck(limits=tight_limits)
        decision = checker.check_signal_execution(sample_signal, empty_portfolio)
        assert decision.allowed is False
        assert "position_count" in list(decision.checks_failed)

    def test_check_signal_execution_sector_concentration(self, empty_portfolio, sample_signal):
        """Sector concentration check is performed."""
        tight_limits = RiskLimits(
            max_sector_pct=0.05,  # Very tight
            max_single_stock_pct=0.02,
        )
        checker = PreExecutionRiskCheck(limits=tight_limits)
        decision = checker.check_signal_execution(sample_signal, empty_portfolio)
        # Should still pass for first position, but capital may be reduced
        assert "sector_concentration" in decision.checks_passed

    def test_check_signal_execution_correlation_penalty(self, populated_portfolio, sample_signal):
        """Correlation applies penalty to confidence."""
        # Use a signal for a different sector (Banking) to avoid sector concentration issues
        signal_banking = Signal(
            signal_id="sig-banking",
            timestamp=datetime.utcnow(),
            symbol="SBIN",  # Banking sector, not IT
            mode="paper",
            strategy_name="ml_long",
            signal_type=SignalType.ml_prediction,
            direction=Direction.long,
            confidence=0.75,
            score=0.72,
            entry=EntrySpec(entry_price=500, stop_price=490, target_price=550),
            risk=RiskSpec(size_hint_pct=0.01),
        )
        with patch("portfolio.correlation.compute_portfolio_correlation") as mock_corr:
            mock_corr.return_value = (0.5, ("INFY", "SBIN"))
            checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
            decision = checker.check_signal_execution(signal_banking, populated_portfolio)
            # Should have applied a penalty
            assert "correlation_penalty_reduction" in decision.adjustments

    def test_check_signal_execution_sets_priority(self, empty_portfolio, sample_signal):
        """Priority is set based on rank and constraints."""
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(sample_signal, empty_portfolio)
        assert 1 <= decision.priority <= 5

    def test_check_signal_execution_watchlist_mode_rejected(self, empty_portfolio):
        """Watchlist mode signals are rejected."""
        watchlist_signal = Signal(
            signal_id="sig-watchlist",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            mode="watchlist",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.8,
            score=0.8,
        )
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(watchlist_signal, empty_portfolio)
        assert decision.allowed is False

    def test_check_signal_execution_paper_mode_allowed(self, empty_portfolio):
        """Paper mode signals are allowed."""
        paper_signal = Signal(
            signal_id="sig-paper",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            mode="paper",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.8,
            score=0.8,
            entry=EntrySpec(entry_price=1600, stop_price=1500, target_price=1700),
            risk=RiskSpec(size_hint_pct=0.01),
        )
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(paper_signal, empty_portfolio)
        assert decision.allowed is True

    def test_check_signal_execution_live_mode_with_low_capital(self, empty_portfolio):
        """Live mode rejects when capital depleted."""
        live_signal = Signal(
            signal_id="sig-live",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            mode="live",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.8,
            score=0.8,
            entry=EntrySpec(entry_price=1600, stop_price=1500, target_price=1700),
            risk=RiskSpec(size_hint_pct=0.01),
        )
        # Drain capital
        empty_portfolio.cash_available = 50_000
        checker = PreExecutionRiskCheck(limits=LIVE_TRADING_LIMITS)
        decision = checker.check_signal_execution(live_signal, empty_portfolio)
        assert decision.allowed is False


# ============================================================================
# Portfolio State Tests
# ============================================================================


class TestPortfolioState:
    def test_portfolio_net_worth(self, populated_portfolio):
        """Net worth = cash + positions value."""
        expected = populated_portfolio.cash_available + (
            100 * 1600 + 50 * 2600
        )  # 160k + 130k + 450k
        assert populated_portfolio.net_worth == expected

    def test_portfolio_deployed_pct(self, populated_portfolio):
        """Deployed % is positions / capital."""
        expected = (160_000 + 130_000) / 500_000
        assert populated_portfolio.deployed_pct == pytest.approx(expected)

    def test_portfolio_position_unrealized_pnl(self):
        """Position P&L is correct."""
        pos = PortfolioPosition(symbol="INFY", qty=100, entry_price=1500, current_price=1600)
        assert pos.unrealized_pnl == 10_000

    def test_portfolio_position_market_value(self):
        """Position market value is qty * price."""
        pos = PortfolioPosition(symbol="INFY", qty=100, entry_price=1500, current_price=1600)
        assert pos.market_value == 160_000


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    def test_very_small_capital_exposure(self):
        """Exposure with very small capital handles gracefully."""
        portfolio = PortfolioState(
            positions={},
            total_capital=1.0,  # Minimal capital
            cash_available=1.0,
        )
        exposures = compute_sector_exposure(portfolio)
        assert len(exposures) == 0

    def test_signal_with_missing_entry_spec(self, empty_portfolio):
        """Signal without entry spec is handled."""
        signal_no_entry = Signal(
            signal_id="sig-no-entry",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            mode="paper",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.8,
            score=0.8,
            entry=None,
        )
        checker = PreExecutionRiskCheck(limits=PAPER_TRADE_LIMITS)
        decision = checker.check_signal_execution(signal_no_entry, empty_portfolio)
        # Should still run checks
        assert len(decision.checks_passed) + len(decision.checks_failed) > 0

    def test_negative_pnl_position(self):
        """Position with negative P&L."""
        pos = PortfolioPosition(symbol="INFY", qty=100, entry_price=1600, current_price=1500)
        assert pos.unrealized_pnl == -10_000

    def test_sector_exposure_with_equal_weights(self):
        """Multiple sectors with equal weight rank correctly."""
        portfolio = PortfolioState(
            positions={
                "INFY": PortfolioPosition(
                    symbol="INFY", qty=100, entry_price=1500, current_price=2000
                ),
                "RELIANCE": PortfolioPosition(
                    symbol="RELIANCE", qty=100, entry_price=2500, current_price=2000
                ),
            },
            total_capital=500_000,
            cash_available=100_000,
        )
        exposures = compute_sector_exposure(portfolio)
        # Both should have equal value (200k each)
        assert exposures["IT"].pct_of_capital == exposures["Energy"].pct_of_capital
        # But different ranks (sorted order)
        assert exposures["IT"].rank != exposures["Energy"].rank
