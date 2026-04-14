"""
Unit tests for signals/features.py — TDD RED phase.
No external APIs, no DB. Uses synthetic OHLCV data.
"""

import numpy as np
import pandas as pd

from signals.features import (
    AUXILIARY_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_COLUMNS,
    build_features,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    """Synthetic OHLCV with realistic price behaviour."""
    rng = np.random.default_rng(0)
    close = 1000 + rng.normal(0, 15, n).cumsum()
    close = np.abs(close)  # keep positive
    high = close + rng.uniform(1, 10, n)
    low = close - rng.uniform(1, 10, n)
    low = np.maximum(low, 1.0)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# build_features — basic contract
# ---------------------------------------------------------------------------


class TestBuildFeaturesContract:
    def test_returns_dataframe(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_index_is_datetimeindex(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_no_nan_in_feature_columns(self):
        df = _make_ohlcv()
        result = build_features(df)
        # Auxiliary columns and close are intentionally NaN when not provided
        skip = set(LABEL_COLUMNS) | set(AUXILIARY_FEATURE_COLUMNS) | {"close"}
        feature_cols = [c for c in result.columns if c not in skip]
        assert not result[feature_cols].isnull().any().any(), (
            f"NaN found in features: {result[feature_cols].isnull().sum()[result[feature_cols].isnull().any()]}"
        )

    def test_all_expected_feature_columns_present(self):
        df = _make_ohlcv()
        result = build_features(df)
        missing = set(FEATURE_COLUMNS) - set(result.columns)
        assert not missing, f"Missing feature columns: {missing}"

    def test_fewer_rows_than_input_due_to_warmup(self):
        """First ~60 rows dropped due to indicator warm-up."""
        df = _make_ohlcv(300)
        result = build_features(df)
        assert len(result) < len(df)
        assert len(result) > 0

    def test_label_columns_absent_by_default(self):
        """Label columns must NOT appear in inference features."""
        df = _make_ohlcv()
        result = build_features(df, include_labels=False)
        for col in LABEL_COLUMNS:
            assert col not in result.columns, f"Label column '{col}' leaked into features"

    def test_label_columns_present_when_requested(self):
        df = _make_ohlcv()
        result = build_features(df, include_labels=True)
        assert "label" in result.columns
        assert "forward_return_5d" in result.columns

    def test_returns_new_dataframe_not_mutating_input(self):
        df = _make_ohlcv()
        original_cols = list(df.columns)
        build_features(df)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Momentum features
# ---------------------------------------------------------------------------


class TestMomentumFeatures:
    def test_rsi_14_bounded_0_to_100(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert result["rsi_14"].between(0, 100).all()

    def test_rsi_7_bounded_0_to_100(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert result["rsi_7"].between(0, 100).all()

    def test_macd_cross_only_plus1_minus1(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert set(result["macd_cross"].unique()).issubset({-1, 0, 1})

    def test_roc_5_is_percentage_change(self):
        df = _make_ohlcv()
        result = build_features(df)
        # ROC should be in a reasonable range for stock prices
        assert result["roc_5"].abs().max() < 100  # <100% 5d move is sane


# ---------------------------------------------------------------------------
# Volatility features
# ---------------------------------------------------------------------------


class TestVolatilityFeatures:
    def test_atr_pct_positive(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert (result["atr_pct"] > 0).all()

    def test_bb_position_mostly_bounded(self):
        df = _make_ohlcv()
        result = build_features(df)
        # Allow a few breakouts but most should be 0–1
        within = result["bb_position"].between(-0.5, 1.5)
        assert within.mean() > 0.90

    def test_realized_vol_positive(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert (result["realized_vol_10"] > 0).all()
        assert (result["realized_vol_20"] > 0).all()


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------


class TestVolumeFeatures:
    def test_volume_zscore_mean_near_zero(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert abs(result["volume_zscore_20"].mean()) < 1.0

    def test_vwap_dev_reasonable_range(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert result["vwap_dev"].abs().max() < 1.0  # <100% deviation from VWAP


# ---------------------------------------------------------------------------
# Trend features
# ---------------------------------------------------------------------------


class TestTrendFeatures:
    def test_ema_cross_9_21_values(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert set(result["ema_cross_9_21"].unique()).issubset({-1, 0, 1})

    def test_adx_14_bounded(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert (result["adx_14"] >= 0).all()
        assert (result["adx_14"] <= 100).all()

    def test_price_vs_ema50_is_ratio(self):
        df = _make_ohlcv()
        result = build_features(df)
        # Should be small fractional deviation, not absolute price
        assert result["price_vs_ema50"].abs().max() < 2.0


# ---------------------------------------------------------------------------
# Mean reversion features
# ---------------------------------------------------------------------------


class TestMeanReversionFeatures:
    def test_zscore_20_reasonable_range(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert result["zscore_20"].abs().max() < 10

    def test_distance_from_52w_high_non_positive(self):
        df = _make_ohlcv()
        result = build_features(df)
        # Distance from 52w high is always <= 0 (price ≤ its own 52w high)
        assert (result["distance_from_52w_high"] <= 0.001).all()

    def test_distance_from_52w_low_non_negative(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert (result["distance_from_52w_low"] >= -0.001).all()


# ---------------------------------------------------------------------------
# Label generation (training only)
# ---------------------------------------------------------------------------


class TestLabelGeneration:
    def test_label_is_binary(self):
        df = _make_ohlcv()
        result = build_features(df, include_labels=True)
        assert set(result["label"].dropna().unique()).issubset({0, 1})

    def test_forward_return_5d_has_nans_at_tail(self):
        """Last 5 rows must be NaN (no future data)."""
        df = _make_ohlcv(200)
        result = build_features(df, include_labels=True)
        assert result["forward_return_5d"].iloc[-5:].isna().all()

    def test_label_class_balance_not_extreme(self):
        """Expect 20–80% positive labels on random walk data."""
        df = _make_ohlcv(500)
        result = build_features(df, include_labels=True)
        pos_rate = result["label"].dropna().mean()
        assert 0.15 < pos_rate < 0.85


# ---------------------------------------------------------------------------
# Regime feature
# ---------------------------------------------------------------------------


class TestRegimeFeature:
    def test_vol_regime_is_binary(self):
        df = _make_ohlcv()
        result = build_features(df)
        assert set(result["vol_regime"].unique()).issubset({0, 1})

    def test_sufficient_data_required(self):
        """Too-short input should raise or return empty."""
        df = _make_ohlcv(30)  # less than 60-bar warm-up
        result = build_features(df)
        assert result.empty or len(result) == 0
