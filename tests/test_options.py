"""Unit tests for options/ — TDD RED phase. No live Kite or market data required."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from options.greeks import compute_portfolio_delta, delta, gamma, theta
from options.iv_features import (
    IVFeatures,
    _write_iv_snapshot,
    build_fo_features,
    compute_iv_percentile,
    compute_iv_rank,
    compute_max_pain,
    compute_realized_vol,
)
from options.strategy import FoStrategyEngine, SignalType

# ---------------------------------------------------------------------------
# greeks.py
# ---------------------------------------------------------------------------


class TestDelta:
    """Black-Scholes delta: call ∈ (0,1), put ∈ (-1,0)."""

    def test_deep_itm_call_delta_near_one(self):
        # S=120, K=100, deep ITM call → delta close to 1
        d = delta(S=120, K=100, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert d > 0.9

    def test_deep_otm_call_delta_near_zero(self):
        d = delta(S=80, K=100, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert d < 0.1

    def test_atm_call_delta_near_half(self):
        d = delta(S=100, K=100, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert 0.45 < d < 0.65

    def test_deep_itm_put_delta_near_minus_one(self):
        d = delta(S=80, K=100, T=0.25, r=0.065, sigma=0.20, option_type="put")
        assert d < -0.9

    def test_atm_put_delta_near_minus_half(self):
        d = delta(S=100, K=100, T=0.25, r=0.065, sigma=0.20, option_type="put")
        assert -0.65 < d < -0.35

    def test_put_call_parity_delta(self):
        # delta_call - delta_put ≈ 1 (put-call parity)
        d_call = delta(S=100, K=100, T=0.25, r=0.065, sigma=0.25, option_type="call")
        d_put = delta(S=100, K=100, T=0.25, r=0.065, sigma=0.25, option_type="put")
        assert abs(d_call - d_put - 1.0) < 0.01

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            delta(S=100, K=100, T=0.25, r=0.065, sigma=0.20, option_type="swap")

    def test_zero_time_to_expiry_raises(self):
        with pytest.raises(ValueError):
            delta(S=100, K=100, T=0.0, r=0.065, sigma=0.20, option_type="call")


class TestGamma:
    def test_gamma_positive(self):
        g = gamma(S=100, K=100, T=0.25, r=0.065, sigma=0.20)
        assert g > 0

    def test_atm_gamma_higher_than_deep_itm(self):
        g_atm = gamma(S=100, K=100, T=0.25, r=0.065, sigma=0.20)
        g_ditm = gamma(S=150, K=100, T=0.25, r=0.065, sigma=0.20)
        assert g_atm > g_ditm

    def test_gamma_same_for_call_and_put(self):
        g_call = gamma(S=100, K=100, T=0.25, r=0.065, sigma=0.20)
        g_put = gamma(S=100, K=100, T=0.25, r=0.065, sigma=0.20)
        assert abs(g_call - g_put) < 1e-10


class TestTheta:
    def test_theta_negative_for_long_call(self):
        t = theta(S=100, K=100, T=0.25, r=0.065, sigma=0.20, option_type="call")
        assert t < 0

    def test_theta_negative_for_long_put(self):
        t = theta(S=100, K=100, T=0.25, r=0.065, sigma=0.20, option_type="put")
        assert t < 0

    def test_theta_larger_magnitude_near_expiry(self):
        t_near = theta(S=100, K=100, T=0.01, r=0.065, sigma=0.20, option_type="call")
        t_far = theta(S=100, K=100, T=1.0, r=0.065, sigma=0.20, option_type="call")
        assert abs(t_near) > abs(t_far)


class TestComputePortfolioDelta:
    def test_empty_positions_returns_zero(self):
        assert compute_portfolio_delta([]) == 0.0

    def test_single_position(self):
        positions = [{"delta": 0.5, "qty": 100, "lot_size": 1}]
        result = compute_portfolio_delta(positions)
        assert abs(result - 50.0) < 1e-6

    def test_long_call_short_put_near_zero_net_delta(self):
        # long call delta ~0.5 short put delta ~-0.5 → net ≈ 0
        positions = [
            {"delta": 0.5, "qty": 1, "lot_size": 50},  # long call → +25
            {"delta": -0.5, "qty": -1, "lot_size": 50},  # short put (sold) → +25
        ]
        result = compute_portfolio_delta(positions)
        # (0.5 * 1 * 50) + (-0.5 * -1 * 50) = 25 + 25 = 50  net long
        assert result == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# iv_features.py
# ---------------------------------------------------------------------------


class TestComputeIVRank:
    def test_current_at_high_returns_one(self):
        iv_series = pd.Series([0.10, 0.15, 0.20, 0.25, 0.30])
        rank = compute_iv_rank(iv_series)
        assert rank == pytest.approx(1.0)

    def test_current_at_low_returns_zero(self):
        iv_series = pd.Series([0.30, 0.25, 0.20, 0.15, 0.10])
        rank = compute_iv_rank(iv_series)
        assert rank == pytest.approx(0.0)

    def test_current_at_midpoint_returns_half(self):
        iv_series = pd.Series([0.10, 0.20, 0.30])  # current=0.30? no, [0.10, 0.20, 0.30]
        # Actually current = last value (0.30) → rank = (0.30 - 0.10) / (0.30 - 0.10) = 1.0
        # let's use a series where current is 0.20 (middle of 0.10..0.30)
        iv_series2 = pd.Series([0.10, 0.30, 0.20])  # current = last = 0.20
        rank = compute_iv_rank(iv_series2)
        assert rank == pytest.approx(0.5)

    def test_flat_iv_series_returns_zero(self):
        iv_series = pd.Series([0.20, 0.20, 0.20])
        # high == low → rank should return 0.0 (no range)
        rank = compute_iv_rank(iv_series)
        assert rank == 0.0

    def test_rank_bounded_zero_to_one(self):
        iv_series = pd.Series([0.10 + i * 0.005 for i in range(252)])
        rank = compute_iv_rank(iv_series)
        assert 0.0 <= rank <= 1.0


class TestComputeIVPercentile:
    def test_current_above_all_historical(self):
        iv_series = pd.Series([*range(1, 252), 300])  # current=300 > all
        pct = compute_iv_percentile(iv_series)
        assert pct >= 99.0

    def test_current_at_median_returns_near_50(self):
        iv_series = pd.Series(range(1, 101))  # 1..100, current=100
        # 100 is above all 1-99 → ~100th pct
        # use midpoint
        iv_mid = pd.Series([*range(100), 50])  # current=50
        pct = compute_iv_percentile(iv_mid)
        assert 40.0 <= pct <= 60.0


class TestComputeRealizedVol:
    def test_returns_positive_float(self):
        prices = pd.Series([100.0 + i * 0.5 for i in range(30)])
        rv = compute_realized_vol(prices, window=20)
        assert rv > 0.0

    def test_flat_price_returns_zero(self):
        prices = pd.Series([100.0] * 30)
        rv = compute_realized_vol(prices, window=20)
        assert rv == pytest.approx(0.0)

    def test_annualization(self):
        # volatile series → rv should be annualized (order of 10–300%)
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0, 0.01, 252)))
        rv = compute_realized_vol(prices, window=20)
        assert 0.05 < rv < 3.0  # annualized, reasonable range


class TestComputeMaxPain:
    def test_max_pain_is_a_valid_strike(self):
        strikes = [95, 100, 105, 110]
        call_oi = {95: 100, 100: 500, 105: 300, 110: 200}
        put_oi = {95: 200, 100: 300, 105: 500, 110: 100}
        mp = compute_max_pain(strikes, call_oi, put_oi)
        assert mp in strikes

    def test_max_pain_example(self):
        # With all OI concentrated at 100 strike, max pain = 100
        strikes = [90, 95, 100, 105, 110]
        call_oi = {90: 0, 95: 0, 100: 1000, 105: 0, 110: 0}
        put_oi = {90: 0, 95: 0, 100: 1000, 105: 0, 110: 0}
        mp = compute_max_pain(strikes, call_oi, put_oi)
        # both calls and puts have max OI at 100 — at expiry = 100, max writers win
        assert mp == 100


class TestIVFeatures:
    def test_dataclass_fields(self):
        f = IVFeatures(
            symbol="NIFTY",
            expiry_date=date(2025, 4, 24),
            iv_rank=0.75,
            iv_percentile=80.0,
            iv_premium=0.06,
            put_call_ratio=1.2,
            max_pain=22_000,
            days_to_expiry=5,
            current_iv=0.22,
            realized_vol=0.16,
        )
        assert f.iv_rank == 0.75
        assert f.days_to_expiry == 5

    def test_to_dict_contains_all_keys(self):
        f = IVFeatures(
            symbol="RELIANCE",
            expiry_date=date(2025, 4, 24),
            iv_rank=0.60,
            iv_percentile=70.0,
            iv_premium=0.04,
            put_call_ratio=0.9,
            max_pain=2_800,
            days_to_expiry=7,
            current_iv=0.25,
            realized_vol=0.21,
        )
        d = f.to_dict()
        for key in (
            "iv_rank",
            "iv_percentile",
            "iv_premium",
            "put_call_ratio",
            "max_pain",
            "days_to_expiry",
        ):
            assert key in d


class TestBuildFoFeatures:
    """build_fo_features uses a mock Kite client."""

    def _mock_kite(self):
        kite = MagicMock()
        # iv_history: 252 daily data points with 'iv' column
        iv_data = [
            {"date": f"2024-{i // 30 + 1:02d}-{(i % 30) + 1:02d}", "iv": 0.15 + i * 0.001}
            for i in range(252)
        ]
        kite.historical_data.return_value = iv_data
        # option chain data
        kite.ltp.return_value = {
            "NFO:NIFTY24APR22000CE": {"last_price": 150.0},
        }
        return kite

    def test_returns_ivfeatures_object(self):
        kite = self._mock_kite()
        with (
            patch("options.iv_features._fetch_iv_history") as mock_iv,
            patch("options.iv_features._fetch_option_chain") as mock_oc,
            patch("options.iv_features._fetch_underlying_prices") as mock_ul,
            patch("options.iv_features._write_iv_snapshot"),
        ):
            mock_iv.return_value = pd.Series([0.15 + i * 0.001 for i in range(252)])
            mock_oc.return_value = (
                {22000: 500, 22050: 300},  # call_oi
                {22000: 400, 21950: 600},  # put_oi
            )
            mock_ul.return_value = pd.Series([22000.0 + i * 2 for i in range(30)])
            result = build_fo_features("NIFTY", date(2025, 4, 24), kite)
        assert isinstance(result, IVFeatures)
        assert result.symbol == "NIFTY"
        assert 0.0 <= result.iv_rank <= 1.0
        assert result.days_to_expiry >= 0

    def test_build_fo_features_persists_snapshot(self):
        kite = self._mock_kite()
        with (
            patch("options.iv_features._fetch_iv_history") as mock_iv,
            patch("options.iv_features._fetch_option_chain") as mock_oc,
            patch("options.iv_features._fetch_underlying_prices") as mock_ul,
            patch("options.iv_features._write_iv_snapshot") as mock_write,
        ):
            mock_iv.return_value = pd.Series([0.20 + i * 0.001 for i in range(252)])
            mock_oc.return_value = ({22000: 400}, {22000: 500})
            mock_ul.return_value = pd.Series([22000.0 + i for i in range(30)])
            build_fo_features("NIFTY", date(2025, 4, 24), kite)
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# strategy.py
# ---------------------------------------------------------------------------


class TestFoStrategyEngine:
    def _engine(self, max_positions: int = 3) -> FoStrategyEngine:
        return FoStrategyEngine(max_concurrent_positions=max_positions)

    def _features(self, iv_rank: float, iv_premium: float) -> IVFeatures:
        return IVFeatures(
            symbol="NIFTY",
            expiry_date=date.today() + timedelta(days=7),
            iv_rank=iv_rank,
            iv_percentile=iv_rank * 100,
            iv_premium=iv_premium,
            put_call_ratio=1.0,
            max_pain=22_000,
            days_to_expiry=7,
            current_iv=0.20,
            realized_vol=0.20 - iv_premium,
        )

    def test_high_iv_rank_generates_sell_signal(self):
        engine = self._engine()
        features = self._features(iv_rank=0.80, iv_premium=0.07)
        signal = engine.generate_signal(features)
        assert signal is not None
        assert signal["type"] == SignalType.SELL_PREMIUM

    def test_low_iv_rank_generates_buy_signal(self):
        engine = self._engine()
        features = self._features(iv_rank=0.20, iv_premium=-0.03)
        signal = engine.generate_signal(features)
        assert signal is not None
        assert signal["type"] == SignalType.BUY_PREMIUM

    def test_mid_iv_rank_returns_none(self):
        engine = self._engine()
        features = self._features(iv_rank=0.50, iv_premium=0.02)
        signal = engine.generate_signal(features)
        assert signal is None

    def test_high_iv_rank_but_low_iv_premium_no_signal(self):
        # iv_rank > 0.7 but iv_premium <= 0.05 → no sell signal
        engine = self._engine()
        features = self._features(iv_rank=0.80, iv_premium=0.03)
        signal = engine.generate_signal(features)
        assert signal is None

    def test_max_positions_blocks_new_signals(self):
        engine = self._engine(max_positions=1)
        features = self._features(iv_rank=0.80, iv_premium=0.07)
        # First signal succeeds
        engine.add_position("NIFTY_1")
        signal = engine.generate_signal(features)
        assert signal is None  # at capacity

    def test_signal_contains_required_fields(self):
        engine = self._engine()
        features = self._features(iv_rank=0.85, iv_premium=0.09)
        signal = engine.generate_signal(features)
        assert signal is not None
        for field in ("type", "symbol", "iv_rank", "iv_premium", "days_to_expiry"):
            assert field in signal

    def test_delta_hedge_needed_when_above_threshold(self):
        engine = self._engine()
        net_delta = 0.25  # above ±0.10 threshold
        hedge = engine.compute_delta_hedge(
            net_delta=net_delta,
            underlying_price=22_000.0,
            lot_size=50,
        )
        assert hedge is not None
        assert hedge["action"] in ("BUY", "SELL")
        assert hedge["qty"] > 0

    def test_no_hedge_needed_when_within_threshold(self):
        engine = self._engine()
        hedge = engine.compute_delta_hedge(
            net_delta=0.05,
            underlying_price=22_000.0,
            lot_size=50,
        )
        assert hedge is None

    def test_add_and_remove_position(self):
        engine = self._engine(max_positions=3)
        engine.add_position("pos_1")
        assert engine.position_count == 1
        engine.remove_position("pos_1")
        assert engine.position_count == 0

    def test_position_count_cannot_exceed_max(self):
        engine = self._engine(max_positions=2)
        engine.add_position("pos_1")
        engine.add_position("pos_2")
        with pytest.raises(ValueError):
            engine.add_position("pos_3")


# ---------------------------------------------------------------------------
# _write_iv_snapshot (DB persistence)
# ---------------------------------------------------------------------------


class TestWriteIvSnapshot:
    def _sample_features(self) -> IVFeatures:
        return IVFeatures(
            symbol="NIFTY",
            expiry_date=date(2025, 4, 24),
            iv_rank=0.70,
            iv_percentile=75.0,
            iv_premium=0.05,
            put_call_ratio=1.1,
            max_pain=22_000,
            days_to_expiry=7,
            current_iv=0.22,
            realized_vol=0.17,
        )

    def _mock_engine(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine, mock_conn

    def test_executes_insert(self):
        features = self._sample_features()
        engine, conn = self._mock_engine()
        with patch("data.store.get_engine", return_value=engine):
            _write_iv_snapshot(features)
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_params_include_symbol_and_expiry(self):
        features = self._sample_features()
        engine, conn = self._mock_engine()
        with patch("data.store.get_engine", return_value=engine):
            _write_iv_snapshot(features)
        params = conn.execute.call_args[0][1]
        assert params["symbol"] == "NIFTY"
        assert params["expiry_date"] == date(2025, 4, 24)

    def test_db_error_does_not_propagate(self):
        """Snapshot write failures must never crash signal generation."""
        features = self._sample_features()
        with patch("data.store.get_engine", side_effect=Exception("DB down")):
            _write_iv_snapshot(features)  # must not raise
