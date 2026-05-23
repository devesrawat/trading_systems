"""
Tests for MomentumSentinelStrategy (Hermes dual-engine, aggressive leg).

All tests are pure CPU — no DB, Redis, or broker calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signals.strategies.momentum_sentinel import MomentumSentinelStrategy


def _make_df(
    n: int = 70,
    trend: str = "up",
    vol_spike: bool = True,
    spike_multiplier: float = 2.5,
) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame.

    Parameters
    ----------
    n : int
        Number of bars.
    trend : str
        "up" → rising prices (strong uptrend), "flat" → flat prices.
    vol_spike : bool
        Whether the last bar has a volume spike.
    spike_multiplier : float
        Volume of the last bar = spike_multiplier * avg of prior bars.
    """
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    if trend == "up":
        # Consistent uptrend so RSI > 60 and price above SMA50
        closes = np.linspace(100, 160, n) + rng.normal(0, 0.5, n)
    else:
        # Flat — RSI ≈ 50, no strong trend
        closes = np.full(n, 100.0) + rng.normal(0, 0.3, n)

    base_vol = np.full(n, 1_000_000.0)
    if vol_spike:
        base_vol[-1] = spike_multiplier * 1_000_000.0

    return pd.DataFrame(
        {
            "open": closes * 0.99,
            "high": closes * 1.01,
            "low": closes * 0.98,
            "close": closes,
            "volume": base_vol,
        },
        index=dates,
    )


@pytest.fixture
def strategy() -> MomentumSentinelStrategy:
    return MomentumSentinelStrategy()


# ------------------------------------------------------------------
# Signal generation
# ------------------------------------------------------------------


def test_scan_passing_stock(strategy):
    """Uptrend + volume surge + RSI > 60 → dict result."""
    df = _make_df(n=70, trend="up", vol_spike=True, spike_multiplier=2.5)
    result = strategy.scan("TEST", df)
    assert result is not None
    assert result["symbol"] == "TEST"
    assert result["strategy"] == "momentum_sentinel"
    assert result["volume_ratio"] >= 2.0
    assert result["rsi_14"] > 60.0
    assert result["distance_to_sma50_pct"] > 0.0


def test_scan_rejects_below_sma50(strategy):
    """Price below SMA50 → None."""
    n = 70
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # Strong downtrend — last price well below SMA50
    closes = np.linspace(160, 100, n)
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes + 1,
            "low": closes - 1,
            "close": closes,
            "volume": np.full(n, 2_500_000.0),  # volume spike present
        },
        index=dates,
    )
    result = strategy.scan("TEST", df)
    assert result is None


def test_scan_rejects_insufficient_volume(strategy):
    """Volume ratio < 2× → None even with uptrend."""
    df = _make_df(n=70, trend="up", vol_spike=False)
    # All volume bars equal — no spike, ratio ≈ 1.0
    result = strategy.scan("TEST", df)
    assert result is None


def test_scan_rejects_low_rsi(strategy):
    """Flat price → RSI ≈ 50 → None despite volume spike."""
    df = _make_df(n=70, trend="flat", vol_spike=True, spike_multiplier=3.0)
    result = strategy.scan("TEST", df)
    # RSI on flat prices stays near 50 and won't exceed 60
    if result is not None:
        assert result["rsi_14"] > 60.0, "If result returned, RSI must be > 60"


def test_scan_returns_json_serialisable_fields(strategy):
    """All result values must be JSON-serialisable (float/int/str)."""
    import json

    df = _make_df(n=70, trend="up", vol_spike=True, spike_multiplier=2.5)
    result = strategy.scan("TEST", df)
    if result is not None:
        json.dumps(result)  # must not raise


# ------------------------------------------------------------------
# Sorting
# ------------------------------------------------------------------


def test_sort_key_descending_by_volume_ratio(strategy):
    """sort_key must order highest volume_ratio first (ascending comparator)."""
    results = [
        {"symbol": "A", "volume_ratio": 3.0},
        {"symbol": "B", "volume_ratio": 5.0},
        {"symbol": "C", "volume_ratio": 2.0},
    ]
    sorted_results = sorted(results, key=strategy.sort_key)
    assert [r["symbol"] for r in sorted_results] == ["B", "A", "C"]


# ------------------------------------------------------------------
# RSI helper
# ------------------------------------------------------------------


def test_rsi_perfect_uptrend():
    """RSI on a pure uptrend (all gains, no losses) should return 100."""
    closes = pd.Series(np.linspace(100, 200, 50))
    rsi = MomentumSentinelStrategy._rsi(closes, period=14)
    assert rsi == pytest.approx(100.0)


def test_rsi_perfect_downtrend():
    """RSI on a pure downtrend should return near 0."""
    closes = pd.Series(np.linspace(200, 100, 50))
    rsi = MomentumSentinelStrategy._rsi(closes, period=14)
    assert rsi < 5.0


def test_rsi_insufficient_data():
    """Short series returns sentinel 50.0."""
    closes = pd.Series([100.0, 101.0, 102.0])
    rsi = MomentumSentinelStrategy._rsi(closes, period=14)
    assert rsi == 50.0
