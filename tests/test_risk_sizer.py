"""Unit tests for risk/sizer.py — TDD RED phase. Pure arithmetic, no mocks needed."""
import pytest

from risk.sizer import PositionSizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sizer(total_capital: float = 500_000.0, max_position_pct: float = 0.02) -> PositionSizer:
    return PositionSizer(total_capital=total_capital, max_position_pct=max_position_pct)


# ---------------------------------------------------------------------------
# size() — half-Kelly formula
# ---------------------------------------------------------------------------

class TestSize:
    def test_returns_positive_for_valid_signal(self):
        sizer = _sizer()
        result = sizer.size(signal_probability=0.65, asset_volatility=0.20, current_capital=500_000.0)
        assert result > 0

    def test_result_never_exceeds_max_position(self):
        sizer = _sizer(total_capital=500_000.0, max_position_pct=0.02)
        max_allowed = 500_000.0 * 0.02
        result = sizer.size(signal_probability=0.99, asset_volatility=0.01, current_capital=500_000.0)
        assert result <= max_allowed + 0.01   # allow ₹0.01 float tolerance

    def test_high_vol_scales_down_size(self):
        sizer = _sizer()
        low_vol = sizer.size(0.70, asset_volatility=0.10, current_capital=500_000.0)
        high_vol = sizer.size(0.70, asset_volatility=0.50, current_capital=500_000.0)
        assert low_vol > high_vol

    def test_higher_probability_gives_larger_size(self):
        sizer = _sizer()
        low_prob = sizer.size(0.55, asset_volatility=0.20, current_capital=500_000.0)
        high_prob = sizer.size(0.80, asset_volatility=0.20, current_capital=500_000.0)
        assert high_prob > low_prob

    def test_zero_edge_returns_near_zero(self):
        """signal_probability=0.5 means zero edge → Kelly = 0."""
        sizer = _sizer()
        result = sizer.size(signal_probability=0.50, asset_volatility=0.20, current_capital=500_000.0)
        assert result == 0.0

    def test_negative_edge_returns_zero(self):
        """signal_probability < 0.5 is a bad signal — no position."""
        sizer = _sizer()
        result = sizer.size(signal_probability=0.40, asset_volatility=0.20, current_capital=500_000.0)
        assert result == 0.0

    def test_result_rounded_to_two_decimal_places(self):
        sizer = _sizer()
        result = sizer.size(0.65, 0.20, 500_000.0)
        assert result == round(result, 2)

    def test_shrinks_with_lower_current_capital(self):
        sizer = _sizer()
        full = sizer.size(0.70, 0.20, current_capital=500_000.0)
        half = sizer.size(0.70, 0.20, current_capital=250_000.0)
        assert full > half

    def test_very_low_volatility_floored_not_divide_by_zero(self):
        """asset_volatility=0.0 should not raise."""
        sizer = _sizer()
        result = sizer.size(0.65, asset_volatility=0.0, current_capital=500_000.0)
        assert result >= 0


# ---------------------------------------------------------------------------
# shares() — convert ₹ amount to share count
# ---------------------------------------------------------------------------

class TestShares:
    def test_equity_floors_to_whole_shares(self):
        sizer = _sizer()
        shares = sizer.shares(rupee_amount=10_000.0, current_price=333.33)
        assert shares == 30   # floor(10000 / 333.33)

    def test_equity_returns_zero_for_tiny_amount(self):
        sizer = _sizer()
        shares = sizer.shares(rupee_amount=5.0, current_price=1000.0)
        assert shares == 0

    def test_fo_rounds_to_lot_size(self):
        sizer = _sizer()
        # ₹50,000 at price ₹100, lot_size=50 → floor(50000 / (100*50))*50 = 10
        shares = sizer.shares(rupee_amount=50_000.0, current_price=100.0, lot_size=50)
        assert shares % 50 == 0

    def test_fo_lot_size_respected(self):
        sizer = _sizer()
        shares = sizer.shares(rupee_amount=75_000.0, current_price=100.0, lot_size=50)
        assert shares == 750   # floor(75000 / 5000) * 50 = 15 * 50

    def test_negative_price_raises(self):
        sizer = _sizer()
        with pytest.raises(ValueError):
            sizer.shares(rupee_amount=10_000.0, current_price=-100.0)

    def test_zero_price_raises(self):
        sizer = _sizer()
        with pytest.raises(ValueError):
            sizer.shares(rupee_amount=10_000.0, current_price=0.0)


# ---------------------------------------------------------------------------
# max_allowed()
# ---------------------------------------------------------------------------

class TestMaxAllowed:
    def test_returns_2pct_of_capital(self):
        sizer = PositionSizer(total_capital=500_000.0, max_position_pct=0.02)
        assert sizer.max_allowed(500_000.0) == 10_000.0

    def test_scales_with_current_capital(self):
        sizer = PositionSizer(total_capital=500_000.0, max_position_pct=0.02)
        assert sizer.max_allowed(200_000.0) == 4_000.0

    def test_max_position_pct_enforced(self):
        with pytest.raises(ValueError, match="2%"):
            PositionSizer(total_capital=500_000.0, max_position_pct=0.05)
