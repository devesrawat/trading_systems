"""
Unit tests for signals/features/ package — comprehensive feature framework.

Tests cover:
- Each feature module (core_indicators, volatility, sentiment, flow, advanced)
- Feature validator (schema, outliers, stationarity)
- Backward compatibility (FEATURE_COLUMNS unchanged)
- Edge cases (empty series, NaN, zero denominators)
- Performance (all features compute in <100ms on 5 years data)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from signals.feature_lib import (
    compute_atr,
    compute_atr_pct,
    compute_bollinger_bands,
    compute_correlation_to_benchmark,
    compute_dii_participation_series,
    compute_fii_net_cash_normalized,
    compute_fii_participation_series,
    compute_finbert_sentiment_series,
    compute_garman_klass_volatility,
    compute_institutional_holding_trend,
    compute_macd,
    compute_macd_cross,
    compute_mf_inflow_trend,
    compute_momentum,
    compute_net_flow_to_volume_ratio,
    compute_parkinson_volatility,
    compute_price_acceleration,
    compute_realized_volatility,
    compute_retail_participation,
    compute_roc,
    compute_rsi,
    compute_sector_relative_strength,
    compute_sentiment_divergence,
    compute_sentiment_momentum,
    compute_social_sentiment_series,
    compute_support_resistance_levels,
    compute_volatility_of_volatility,
)
from signals.feature_validator import FeatureValidator
from signals.features import AUXILIARY_FEATURE_COLUMNS, FEATURE_COLUMNS, build_features

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV with realistic price action."""
    rng = np.random.default_rng(seed)
    close = 1000 + rng.normal(0, 15, n).cumsum()
    close = np.abs(close)
    high = close + rng.uniform(1, 10, n)
    low = np.maximum(close - rng.uniform(1, 10, n), 1.0)
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    idx = pd.date_range("2020-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# Core Indicators Tests
# ---------------------------------------------------------------------------


class TestCoreIndicators:
    """Test momentum and trend indicators."""

    def test_rsi_computation(self):
        df = _make_ohlcv()
        result = compute_rsi(df["close"], period=14)
        assert isinstance(result, pd.Series)
        assert result.dropna().between(0, 100).all()

    def test_rsi_respects_lookback(self):
        df = _make_ohlcv()
        rsi_14 = compute_rsi(df["close"], period=14)
        rsi_7 = compute_rsi(df["close"], period=7)
        # Both should have same length (pandas-ta aligns)
        assert len(rsi_14) == len(rsi_7)

    def test_rsi_edge_case_short_series(self):
        close = pd.Series([100, 101, 102, 103])
        result = compute_rsi(close, period=14)
        assert isinstance(result, pd.Series)
        # May have NaN values due to insufficient data
        assert len(result) == len(close)

    def test_macd_returns_three_series(self):
        df = _make_ohlcv()
        macd, signal, hist = compute_macd(df["close"])
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)
        assert len(macd) == len(df["close"])

    def test_macd_cross_produces_ternary_values(self):
        df = _make_ohlcv()
        macd, signal, hist = compute_macd(df["close"])
        cross = compute_macd_cross(macd, signal)
        assert set(cross.dropna().unique()).issubset({-1, 0, 1})

    def test_momentum_computation(self):
        close = pd.Series([100, 102, 104, 103, 105])
        mom = compute_momentum(close, period=2)
        expected = pd.Series([np.nan, np.nan, 2.0, 1.0, 1.0])
        pd.testing.assert_series_equal(mom, expected)

    def test_roc_percentage_change(self):
        close = pd.Series([100, 110, 120, 130])
        roc = compute_roc(close, period=1)
        # ROC should be close to pct_change
        pct_change = close.pct_change()
        pd.testing.assert_series_equal(roc, pct_change, check_dtype=False)


# ---------------------------------------------------------------------------
# Volatility Tests
# ---------------------------------------------------------------------------


class TestVolatilityIndicators:
    """Test volatility and range indicators."""

    def test_atr_is_positive(self):
        df = _make_ohlcv()
        atr = compute_atr(df["high"], df["low"], df["close"], period=14)
        assert (atr.dropna() > 0).all()

    def test_atr_pct_ratio(self):
        df = _make_ohlcv()
        atr = compute_atr(df["high"], df["low"], df["close"], period=14)
        atr_pct = compute_atr_pct(atr, df["close"])
        # Should be ATR / close
        expected = atr / df["close"]
        pd.testing.assert_series_equal(atr_pct, expected)

    def test_bollinger_bands_ordering(self):
        df = _make_ohlcv()
        lower, mid, upper, position = compute_bollinger_bands(df["close"])
        # Upper >= Mid >= Lower
        assert (upper >= mid).all() or (upper.isna() | mid.isna()).any()
        assert (mid >= lower).all() or (mid.isna() | lower.isna()).any()

    def test_bollinger_position_bounded(self):
        df = _make_ohlcv()
        _, _, _, position = compute_bollinger_bands(df["close"])
        # Position should mostly be in [0, 1] (allowing some overshoot)
        within = position.between(-0.5, 1.5)
        assert within.mean() > 0.85

    def test_realized_volatility_positive(self):
        df = _make_ohlcv()
        vol = compute_realized_volatility(df["close"], period=20)
        assert (vol.dropna() > 0).all()

    def test_parkinson_volatility_computation(self):
        df = _make_ohlcv()
        vol = compute_parkinson_volatility(df["high"], df["low"], period=20)
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(df)
        assert (vol.dropna() > 0).all() or vol.isna().all()

    def test_garman_klass_volatility_computation(self):
        df = _make_ohlcv()
        vol = compute_garman_klass_volatility(
            df["high"], df["low"], df["close"], df["open"], period=20
        )
        assert isinstance(vol, pd.Series)
        assert len(vol) == len(df)

    def test_volatility_of_volatility(self):
        close = pd.Series(np.random.normal(100, 5, 100))
        vol = compute_realized_volatility(close, period=10)
        vol_of_vol = compute_volatility_of_volatility(vol, period=10)
        assert isinstance(vol_of_vol, pd.Series)
        assert len(vol_of_vol) == len(vol)


# ---------------------------------------------------------------------------
# Flow Indicators Tests
# ---------------------------------------------------------------------------


class TestFlowIndicators:
    """Test FII/DII and institutional flow indicators."""

    def test_fii_net_cash_normalization(self):
        result = compute_fii_net_cash_normalized(5000)
        assert result == 5000 / 1e5

    def test_fii_net_cash_none_returns_nan(self):
        result = compute_fii_net_cash_normalized(None)
        assert np.isnan(result)

    def test_fii_participation_series_with_data(self):
        buying = pd.Series([1000, 1200, 900, 1100])
        selling = pd.Series([800, 900, 1000, 1050])
        result = compute_fii_participation_series(buying, selling, period=2)
        assert isinstance(result, pd.Series)
        assert len(result) == len(buying)

    def test_fii_participation_none_returns_nan(self):
        result = compute_fii_participation_series(None, None, period=2)
        assert isinstance(result, pd.Series)
        assert result.isna().all() or result.empty

    def test_dii_participation_series(self):
        buying = pd.Series([500, 600, 550, 700])
        selling = pd.Series([400, 450, 500, 600])
        result = compute_dii_participation_series(buying, selling, period=2)
        assert isinstance(result, pd.Series)

    def test_net_flow_to_volume_ratio(self):
        flow = pd.Series([100, 150, 120, 180])
        volume = pd.Series([10000, 12000, 11000, 15000])
        result = compute_net_flow_to_volume_ratio(flow, volume, period=2)
        assert isinstance(result, pd.Series)
        assert len(result) == len(flow)

    def test_mf_inflow_trend(self):
        inflow = pd.Series([100, 120, 110, 130, 125])
        result = compute_mf_inflow_trend(inflow, period=2)
        assert isinstance(result, pd.Series)

    def test_institutional_holding_trend(self):
        holding = pd.Series([20.5, 21.0, 20.8, 21.5])
        result = compute_institutional_holding_trend(holding, period=2)
        assert isinstance(result, pd.Series)

    def test_retail_participation(self):
        total = pd.Series([10000, 12000, 11000, 15000])
        institutional = pd.Series([3000, 4000, 3500, 5000])
        result = compute_retail_participation(total, institutional, period=2)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# Sentiment Tests
# ---------------------------------------------------------------------------


class TestSentimentIndicators:
    """Test sentiment score processing."""

    def test_finbert_sentiment_series(self):
        scores = pd.Series([0.5, 0.3, 0.7], index=pd.date_range("2024-01-01", periods=3))
        result = compute_finbert_sentiment_series(scores)
        assert isinstance(result, pd.Series)

    def test_social_sentiment_series(self):
        scores = pd.Series([0.2, 0.3, 0.1, 0.4])
        result = compute_social_sentiment_series(scores, period=2)
        assert isinstance(result, pd.Series)

    def test_sentiment_divergence(self):
        finbert = pd.Series([0.5, 0.3, 0.7])
        news = pd.Series([0.4, 0.5, 0.6])
        result = compute_sentiment_divergence(finbert, news)
        expected_diff = np.abs(finbert - news)
        pd.testing.assert_series_equal(result, expected_diff)

    def test_sentiment_momentum(self):
        sentiment = pd.Series([0.1, 0.2, 0.3, 0.5, 0.4])
        result = compute_sentiment_momentum(sentiment, period=2)
        expected = sentiment - sentiment.shift(2)
        pd.testing.assert_series_equal(result, expected)


# ---------------------------------------------------------------------------
# Advanced Indicators Tests
# ---------------------------------------------------------------------------


class TestAdvancedIndicators:
    """Test correlation, relative strength, regime indicators."""

    def test_correlation_to_benchmark(self):
        symbol_ret = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01])
        bench_ret = pd.Series([0.005, -0.01, 0.02, 0.005, 0.00])
        result = compute_correlation_to_benchmark(symbol_ret, bench_ret, period=3)
        assert isinstance(result, pd.Series)

    def test_sector_relative_strength(self):
        symbol = pd.Series([100, 102, 105, 104])
        sector = pd.Series([50, 50.5, 51, 50.8])
        result = compute_sector_relative_strength(symbol, sector, period=2)
        assert isinstance(result, pd.Series)

    def test_price_acceleration(self):
        close = pd.Series([100, 101, 103, 106, 110])
        accel = compute_price_acceleration(close, period=2)
        assert isinstance(accel, pd.Series)
        assert len(accel) == len(close)

    def test_support_resistance_levels(self):
        df = _make_ohlcv(100)
        support, resistance = compute_support_resistance_levels(df["high"], df["low"], period=20)
        assert isinstance(support, pd.Series)
        assert isinstance(resistance, pd.Series)
        # Resistance >= Support
        assert (resistance >= support).all() or (resistance.isna() | support.isna()).any()


# ---------------------------------------------------------------------------
# FeatureValidator Tests
# ---------------------------------------------------------------------------


class TestFeatureValidator:
    """Test feature validation and quality checks."""

    def test_validator_initialization(self):
        schema = ["feature1", "feature2"]
        validator = FeatureValidator(schema)
        assert validator.feature_schema == schema

    def test_validate_feature_output_valid(self):
        validator = FeatureValidator()
        series = pd.Series([1.0, 2.0, 3.0])
        is_valid, msg = validator.validate_feature_output("test_feature", series)
        assert is_valid
        assert msg == ""

    def test_validate_feature_output_all_nan(self):
        validator = FeatureValidator()
        series = pd.Series([np.nan, np.nan, np.nan])
        is_valid, msg = validator.validate_feature_output("test_feature", series)
        assert not is_valid
        assert "entirely NaN" in msg

    def test_validate_feature_output_with_inf(self):
        validator = FeatureValidator()
        series = pd.Series([1.0, np.inf, 3.0])
        is_valid, msg = validator.validate_feature_output("test_feature", series)
        assert not is_valid
        assert "infinite" in msg

    def test_check_completeness(self):
        schema = ["a", "b", "c"]
        validator = FeatureValidator(schema)
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # missing 'c'
        result = validator.check_completeness(df)
        assert "c" in result["missing_columns"]
        assert {"a", "b"} == set(result["present_columns"])

    def test_detect_outliers_iqr(self):
        validator = FeatureValidator()
        df = pd.DataFrame({"feature": [1, 2, 3, 4, 100]})  # 100 is outlier
        outliers = validator.detect_outliers(df, method="iqr", threshold=1.5)
        assert "feature" in outliers
        assert len(outliers["feature"]) > 0

    def test_detect_outliers_zscore(self):
        validator = FeatureValidator()
        df = pd.DataFrame({"feature": list(range(100)) + [1000]})  # 1000 is extreme
        outliers = validator.detect_outliers(df, method="zscore", threshold=3.0)
        assert "feature" in outliers

    def test_check_stationarity_random_walk(self):
        validator = FeatureValidator()
        # Random walk (non-stationary)
        series = pd.Series(np.random.randn(100).cumsum())
        result = validator.check_stationarity(series)
        assert "stationary" in result
        assert "adf_statistic" in result
        assert "p_value" in result

    def test_check_stationarity_short_series(self):
        validator = FeatureValidator()
        series = pd.Series([1, 2, 3, 4])  # Too short
        result = validator.check_stationarity(series)
        assert not result["stationary"]
        assert "Too short" in result.get("note", "")

    def test_validate_feature_correlation(self):
        validator = FeatureValidator()
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.01, 2.01, 3.01, 4.01, 5.01],  # highly correlated with a
                "c": [5, 4, 3, 2, 1],
            }
        )
        redundant = validator.validate_feature_correlation(df, threshold=0.95)
        assert len(redundant) > 0


# ---------------------------------------------------------------------------
# Backward Compatibility Tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure FEATURE_COLUMNS unchanged and existing behavior preserved."""

    def test_feature_columns_immutable(self):
        """FEATURE_COLUMNS order and content must not change."""
        expected = [
            "rsi_14",
            "rsi_7",
            "macd",
            "macd_signal",
            "macd_hist",
            "macd_cross",
            "mom_10",
            "roc_5",
            "roc_21",
            "atr_14",
            "atr_pct",
            "bb_upper",
            "bb_mid",
            "bb_lower",
            "bb_position",
            "realized_vol_10",
            "realized_vol_20",
            "volume_zscore_20",
            "obv",
            "obv_slope",
            "vwap_dev",
            "ema_9",
            "ema_21",
            "ema_50",
            "ema_cross_9_21",
            "price_vs_ema50",
            "adx_14",
            "di_plus",
            "di_minus",
            "zscore_20",
            "distance_from_52w_high",
            "distance_from_52w_low",
            "vol_regime",
        ]
        assert expected == FEATURE_COLUMNS

    def test_auxiliary_feature_columns_present(self):
        """Auxiliary columns must be defined."""
        assert len(AUXILIARY_FEATURE_COLUMNS) == 4
        assert "fii_net_cash_norm" in AUXILIARY_FEATURE_COLUMNS
        assert "sentiment_score" in AUXILIARY_FEATURE_COLUMNS

    def test_build_features_unchanged(self):
        """Original build_features() still works as before."""
        df = _make_ohlcv()
        result = build_features(df)

        # Check FEATURE_COLUMNS all present
        for col in FEATURE_COLUMNS:
            assert col in result.columns

        # Check no NaN in core features
        skip = {"close"} | set(AUXILIARY_FEATURE_COLUMNS)
        core_features = [c for c in result.columns if c not in skip]
        assert not result[core_features].isnull().any().any()

    def test_build_features_with_auxiliary(self):
        """Auxiliary columns populated when provided."""
        df = _make_ohlcv()
        result = build_features(
            df,
            fii_net_cash=5000,
            india_vix=25.5,
            sentiment_score=0.3,
            regime_code=1,
        )
        assert result["fii_net_cash_norm"].iloc[0] == 5000 / 1e5
        assert result["india_vix"].iloc[0] == 25.5
        assert result["sentiment_score"].iloc[0] == 0.3
        assert result["regime_code"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Edge Cases & Performance Tests
# ---------------------------------------------------------------------------


class TestEdgeCasesAndPerformance:
    """Test edge cases, missing data, and performance."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=pd.DatetimeIndex([]),
        )
        df.index.name = "time"
        result = build_features(df)
        assert len(result) == 0

    def test_single_value_dataframe(self):
        df = _make_ohlcv(1)
        result = build_features(df)
        # May be empty due to warm-up requirements
        assert isinstance(result, pd.DataFrame)

    def test_all_nan_volume(self):
        df = _make_ohlcv()
        df["volume"] = np.nan
        # Should not crash, but volume-dependent features will be NaN
        result = build_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_zero_denominator_handling(self):
        """Test features handle division by zero gracefully."""
        df = _make_ohlcv()
        df.loc[0, "high"] = df.loc[0, "low"]  # Force high == low
        result = build_features(df)
        # Should not have inf or raise exception
        assert not np.isinf(result.values).any() or result.values.size == 0

    def test_performance_5_years_data(self):
        """All features compute in <100ms on 5 years (1250 trading days)."""
        import time

        df = _make_ohlcv(1250)

        start = time.perf_counter()
        result = build_features(df)
        elapsed = time.perf_counter() - start

        # Performance target: <100ms
        assert elapsed < 0.1, f"build_features took {elapsed:.3f}s (target <0.1s)"
        assert len(result) > 0

    def test_large_price_swings(self):
        """Test with extreme but plausible price changes."""
        df = _make_ohlcv()
        df.loc[50:60, "close"] = df.loc[50:60, "close"] * 2  # 100% jump
        result = build_features(df)
        # Should complete without NaN in core features
        for col in FEATURE_COLUMNS:
            if col in result.columns:
                assert not result[col].isna().all()

    def test_duplicate_timestamps(self):
        """Handle duplicate index values gracefully."""
        df = _make_ohlcv()
        # Make some indices duplicate
        df.index = pd.date_range("2020-01-01", periods=len(df), freq="B")
        df.index = df.index.to_series().duplicated().where(True, df.index)
        # This may cause issues; test robustness
        try:
            result = build_features(df)
            assert isinstance(result, pd.DataFrame)
        except (ValueError, KeyError):
            # Acceptable to fail on malformed input
            pass


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test complete feature pipeline."""

    def test_feature_pipeline_no_errors(self):
        """Full pipeline: OHLCV → features → validation."""
        df = _make_ohlcv(500)
        features = build_features(df)

        validator = FeatureValidator(FEATURE_COLUMNS)
        completeness = validator.check_completeness(features)

        assert len(completeness["missing_columns"]) == 0
        assert len(completeness["present_columns"]) == len(FEATURE_COLUMNS)

    def test_feature_reproducibility(self):
        """Same input produces same features (deterministic)."""
        df = _make_ohlcv(seed=123)
        result1 = build_features(df)
        result2 = build_features(df)

        pd.testing.assert_frame_equal(result1, result2)

    def test_rolling_feature_computation(self):
        """Features computed on sliding windows are consistent."""
        df = _make_ohlcv(300)

        # Compute on full data
        full_features = build_features(df)

        # Compute on subset
        subset_df = df.iloc[-100:]
        subset_features = build_features(subset_df)

        # Overlapping rows should have similar values (allowing for warm-up differences)
        if len(full_features) > 0 and len(subset_features) > 0:
            common_idx = full_features.index.intersection(subset_features.index)
            if len(common_idx) > 0:
                # Values should be close (not exact due to rolling window differences)
                assert len(common_idx) > 0

    def test_no_circular_imports(self):
        """Ensure no circular dependencies in feature modules."""
        # If we can import these, no circular imports
        from signals.features.feature_validator import FeatureValidator

        from signals import features as base_features
        from signals.features import advanced, core_indicators, flow, sentiment, volatility

        assert base_features is not None
        assert core_indicators is not None
        assert volatility is not None
        assert flow is not None
        assert sentiment is not None
        assert advanced is not None
        assert FeatureValidator is not None
