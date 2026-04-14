"""Tests for signals/filters.py (EarningsFilter) and signals/exit_model.py (ExitModel)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from signals.exit_model import ExitModel, PositionContext
from signals.filters import EarningsFilter

# ---------------------------------------------------------------------------
# EarningsFilter
# ---------------------------------------------------------------------------


class TestEarningsFilter:
    """EarningsFilter is always disabled by default (earnings_filter_enabled=False)."""

    def test_disabled_by_default_returns_false(self):
        """is_blackout must return False when feature flag is off."""
        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = False

        with patch("signals.filters.settings", mock_settings):
            result = EarningsFilter().is_blackout("RELIANCE")

        assert result is False

    def test_disabled_never_queries_redis(self):
        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = False
        mock_redis = MagicMock()

        with (
            patch("signals.filters.settings", mock_settings),
            patch("data.store.get_redis", return_value=mock_redis),
        ):
            EarningsFilter().is_blackout("TCS")

        mock_redis.get.assert_not_called()

    def test_enabled_no_redis_data_returns_false(self):
        """Fail open — allow signal if no earnings data is available."""
        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = True
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with (
            patch("signals.filters.settings", mock_settings),
            patch("data.store.get_redis", return_value=mock_redis),
        ):
            result = EarningsFilter().is_blackout("INFY")

        assert result is False

    def test_enabled_with_blackout_flag_returns_true(self):
        from datetime import date

        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = True
        mock_redis = MagicMock()
        mock_redis.get.return_value = date.today().isoformat().encode()  # today = blackout

        with (
            patch("signals.filters.settings", mock_settings),
            patch("data.store.get_redis", return_value=mock_redis),
        ):
            result = EarningsFilter().is_blackout("INFY")

        assert result is True

    def test_enabled_not_in_blackout_returns_false(self):
        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = True
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"2020-01-01"  # far in the past

        with (
            patch("signals.filters.settings", mock_settings),
            patch("data.store.get_redis", return_value=mock_redis),
        ):
            result = EarningsFilter().is_blackout("HDFCBANK")

        assert result is False

    def test_redis_failure_fails_open(self):
        """Redis error → fail open (allow signal — do not crash the loop)."""
        mock_settings = MagicMock()
        mock_settings.earnings_filter_enabled = True
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("connection error")

        with (
            patch("signals.filters.settings", mock_settings),
            patch("data.store.get_redis", return_value=mock_redis),
        ):
            result = EarningsFilter().is_blackout("WIPRO")

        assert result is False


# ---------------------------------------------------------------------------
# ExitModel
# ---------------------------------------------------------------------------


class TestExitModel:
    def _make_context(
        self,
        entry_price: float = 1000.0,
        current_price: float = 1030.0,
        atr: float = 15.0,
        held_days: int = 3,
    ) -> PositionContext:
        return PositionContext(
            symbol="RELIANCE",
            entry_price=entry_price,
            current_price=current_price,
            atr=atr,
            held_days=held_days,
        )

    def test_hit_profit_target_triggers_exit(self):
        """Current price > entry + 2×ATR → take profit."""
        ctx = self._make_context(entry_price=1000.0, current_price=1032.0, atr=15.0)
        # 2×ATR = 30; target = 1030; 1032 > 1030 → exit
        result = ExitModel().should_exit(ctx)
        assert result is True

    def test_below_profit_target_no_exit(self):
        ctx = self._make_context(entry_price=1000.0, current_price=1010.0, atr=15.0)
        # 2×ATR = 30; 1010 < 1030 → no exit
        result = ExitModel().should_exit(ctx)
        assert result is False

    def test_trailing_stop_triggers_exit(self):
        """Current price < entry - 1×ATR → stop loss."""
        ctx = self._make_context(entry_price=1000.0, current_price=984.0, atr=15.0)
        # 1×ATR stop = 985; 984 < 985 → exit
        result = ExitModel().should_exit(ctx)
        assert result is True

    def test_within_stop_range_no_exit(self):
        ctx = self._make_context(entry_price=1000.0, current_price=990.0, atr=15.0)
        # Stop at 985; 990 > 985 → no exit
        result = ExitModel().should_exit(ctx)
        assert result is False

    def test_zero_atr_uses_fallback(self):
        """ATR of 0 must not cause ZeroDivisionError."""
        ctx = self._make_context(entry_price=1000.0, current_price=995.0, atr=0.0)
        result = ExitModel().should_exit(ctx)
        assert isinstance(result, bool)

    def test_position_context_dataclass_fields(self):
        ctx = PositionContext(
            symbol="TCS",
            entry_price=3500.0,
            current_price=3600.0,
            atr=30.0,
            held_days=5,
        )
        assert ctx.symbol == "TCS"
        assert ctx.entry_price == 3500.0
        assert ctx.held_days == 5

    def test_model_predict_returns_bool_on_missing_mlflow(self):
        """predict() falls back gracefully when no MLflow model is registered."""
        ctx = self._make_context()
        result = ExitModel().predict({"some_feature": 1.0})
        assert isinstance(result, (bool, float))
