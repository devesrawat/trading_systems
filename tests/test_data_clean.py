"""Unit tests for data/clean.py — no DB, no Kite required."""

import numpy as np
import pandas as pd
import pytest

from data.clean import (
    fill_missing_bars,
    flag_circuit_limit_days,
    remove_outliers,
    validate_ohlcv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.5, n),
            "high": close + rng.uniform(0, 1, n),
            "low": close - rng.uniform(0, 1, n),
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, n),
        },
        index=idx,
    )
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------


class TestRemoveOutliers:
    def test_zscore_removes_spike(self):
        df = _make_ohlcv(50)
        df_with_spike = df.copy()
        df_with_spike.iloc[25, df_with_spike.columns.get_loc("close")] = 1_000_000
        result = remove_outliers(df_with_spike, col="close", method="zscore", threshold=4.0)
        assert len(result) < len(df_with_spike)
        assert 1_000_000 not in result["close"].values

    def test_iqr_removes_extreme_low(self):
        df = _make_ohlcv(50)
        df_extreme = df.copy()
        df_extreme.iloc[10, df_extreme.columns.get_loc("close")] = -999
        result = remove_outliers(df_extreme, col="close", method="iqr", threshold=3.0)
        assert -999 not in result["close"].values

    def test_no_outliers_returns_same_length(self):
        df = _make_ohlcv(30)
        result = remove_outliers(df, col="close", method="zscore", threshold=4.0)
        assert len(result) == len(df)

    def test_invalid_method_raises(self):
        df = _make_ohlcv(10)
        with pytest.raises(ValueError, match="Unknown method"):
            remove_outliers(df, method="bad_method")

    def test_missing_column_raises(self):
        df = _make_ohlcv(10).drop(columns=["close"])
        with pytest.raises(ValueError, match="Column 'close' not found"):
            remove_outliers(df, col="close")

    def test_returns_new_object(self):
        df = _make_ohlcv(20)
        result = remove_outliers(df)
        assert result is not df


# ---------------------------------------------------------------------------
# validate_ohlcv
# ---------------------------------------------------------------------------


class TestValidateOhlcv:
    def test_valid_df_passes(self):
        df = _make_ohlcv(10)
        # Ensure OHLC consistency
        df["high"] = df[["open", "close"]].max(axis=1) + 0.1
        df["low"] = df[["open", "close"]].min(axis=1) - 0.1
        is_valid, issues = validate_ohlcv(df)
        assert is_valid, issues

    def test_high_less_than_low_fails(self):
        df = _make_ohlcv(5)
        df.iloc[2, df.columns.get_loc("high")] = df.iloc[2]["low"] - 1
        is_valid, issues = validate_ohlcv(df)
        assert not is_valid
        assert any("high < low" in i for i in issues)

    def test_close_above_high_fails(self):
        df = _make_ohlcv(5)
        df.iloc[1, df.columns.get_loc("close")] = df.iloc[1]["high"] + 10
        is_valid, issues = validate_ohlcv(df)
        assert not is_valid
        assert any("close outside" in i for i in issues)

    def test_negative_volume_fails(self):
        df = _make_ohlcv(5)
        df.iloc[0, df.columns.get_loc("volume")] = -1
        is_valid, issues = validate_ohlcv(df)
        assert not is_valid
        assert any("Negative volume" in i for i in issues)

    def test_missing_column_fails(self):
        df = _make_ohlcv(5).drop(columns=["volume"])
        is_valid, issues = validate_ohlcv(df)
        assert not is_valid
        assert any("Missing columns" in i for i in issues)

    def test_empty_df_fails(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        is_valid, issues = validate_ohlcv(df)
        assert not is_valid


# ---------------------------------------------------------------------------
# flag_circuit_limit_days
# ---------------------------------------------------------------------------


class TestFlagCircuitLimitDays:
    def test_normal_moves_not_flagged(self):
        df = _make_ohlcv(20)
        result = flag_circuit_limit_days(df)
        assert "circuit_hit" in result.columns
        assert not result["circuit_hit"].iloc[1:].any()

    def test_large_move_flagged(self):
        df = _make_ohlcv(5)
        # Manually inject a 25% jump
        df.iloc[3, df.columns.get_loc("close")] = df.iloc[2]["close"] * 1.25
        result = flag_circuit_limit_days(df)
        assert result["circuit_hit"].iloc[3]

    def test_exactly_at_threshold_not_flagged(self):
        df = _make_ohlcv(5)
        # 18.9% move — below the 19.9% circuit threshold, must not be flagged
        df.iloc[1, df.columns.get_loc("close")] = 1000.0
        df.iloc[2, df.columns.get_loc("close")] = 1189.0
        result = flag_circuit_limit_days(df)
        assert not result["circuit_hit"].iloc[2]

    def test_returns_new_object(self):
        df = _make_ohlcv(5)
        result = flag_circuit_limit_days(df)
        assert result is not df


# ---------------------------------------------------------------------------
# fill_missing_bars
# ---------------------------------------------------------------------------


class TestFillMissingBars:
    def test_no_gaps_unchanged(self):
        df = _make_ohlcv(10)
        result = fill_missing_bars(df, interval="day")
        # Number of business days should be the same
        assert len(result) == len(df)

    def test_single_gap_filled(self):
        df = _make_ohlcv(10)
        # Remove one row to create a gap
        df_gap = df.drop(df.index[4])
        result = fill_missing_bars(df_gap, interval="day")
        assert len(result) > len(df_gap)
        assert result["is_filled"].any()

    def test_long_gap_not_filled(self):
        idx = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1_000_000},
            index=idx,
        )
        df.index.name = "time"
        # Remove 5 consecutive bars (exceeds max 3)
        df_gap = df.drop(df.index[5:10])
        result = fill_missing_bars(df_gap, interval="day")
        # Rows beyond 3-bar fill should remain NaN
        filled = result[result["is_filled"]]
        assert len(filled) <= 3

    def test_empty_df_returned_as_is(self):
        df = pd.DataFrame()
        result = fill_missing_bars(df, interval="day")
        assert result.empty
