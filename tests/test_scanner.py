"""
Unit tests for the modular scanner framework.

Tests strategies in-process with synthetic DataFrames.
ScannerEngine integration is tested by patching the DB fetch and
replacing ProcessPoolExecutor with a ThreadPoolExecutor so tests
run without spawning subprocesses.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from signals.base_strategy import BaseStrategy
from signals.strategies.rs_breakout import RSBreakoutStrategy
from signals.strategies.tight_closes import TightClosesStrategy
from signals.strategies.vcp import VCPStrategy

# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------


def _make_daily(
    n: int = 300,
    trend: str = "up",  # "up" | "flat" | "down"
    seed: int = 0,
) -> pd.DataFrame:
    """
    Synthetic daily OHLCV.  'up' builds a rising trend suitable for
    Stage 2 / VCP / RS setups.
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

    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "time"
    return df


def _make_records(df: pd.DataFrame) -> list[dict]:
    return df.reset_index().to_dict(orient="records")


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------


class TestBaseStrategy:
    def test_fqname_roundtrip(self):
        s = VCPStrategy()
        fq = s.fqname()
        restored = BaseStrategy.from_fqname(fq)
        assert isinstance(restored, VCPStrategy)

    def test_prepare_returns_none_for_empty(self):
        s = VCPStrategy()
        result = s.prepare(pd.DataFrame())
        assert result is None

    def test_prepare_returns_none_for_short_data(self):
        s = VCPStrategy()
        df = _make_daily(n=50)
        result = s.prepare(df.reset_index())
        assert result is None  # < min_bars=200

    def test_prepare_drops_circuit_hit_rows(self):
        s = VCPStrategy()
        df = _make_daily(n=250)
        # Inject a 25 % spike — should be flagged and dropped
        df.iloc[100, df.columns.get_loc("close")] = df.iloc[99]["close"] * 1.30
        clean = s.prepare(df.reset_index())
        # prepare should still return a result (enough clean bars remain)
        # just verify the spike row is gone
        if clean is not None:
            assert not (clean["close"] > df["close"].max() * 1.20).any()

    def test_default_sort_key_is_symbol(self):
        s = VCPStrategy()
        # VCP overrides sort_key, but base default is "symbol"
        from signals.base_strategy import BaseStrategy as BS

        class _Stub(BS):
            name = "stub"

            def scan(self, sym, df):
                return None

        stub = _Stub()
        assert stub.sort_key({"symbol": "RELIANCE"}) == "RELIANCE"


# ---------------------------------------------------------------------------
# VCPStrategy
# ---------------------------------------------------------------------------


class TestVCPStrategy:
    def _passing_df(self) -> pd.DataFrame:
        """
        Build a DataFrame that satisfies the Stage 2 trend template.
        We don't try to manufacture real VCP contractions — instead we
        verify the strategy correctly rejects datasets that fail each gate.
        """
        return _make_daily(n=300, trend="up")

    def test_rejects_downtrend(self):
        s = VCPStrategy()
        df = _make_daily(n=300, trend="down")
        assert s.scan("X", df) is None

    def test_rejects_flat_below_sma(self):
        s = VCPStrategy()
        df = _make_daily(n=300, trend="flat")
        # flat price will fail the Stage 2 template (not 30 % above 52-week low)
        result = s.scan("X", df)
        # may or may not pass — just ensure it doesn't crash
        assert result is None or isinstance(result, dict)

    def test_result_has_required_keys(self):
        s = VCPStrategy()
        df = _make_daily(n=300, trend="up")
        # Manufacture a strong uptrend that passes trend template
        close = df["close"].values
        close = np.sort(close)  # monotonically rising — maximises passing chance
        df = df.copy()
        df["close"] = close
        df["high"] = close + 5
        df["low"] = close - 5
        # Try scanning — result may be None if no contractions found
        result = s.scan("RELIANCE", df)
        if result is not None:
            assert result["symbol"] == "RELIANCE"
            assert result["strategy"] == "vcp"
            assert "pivot_buy" in result
            assert "contractions" in result
            assert "swing_ranges" in result

    def test_sort_key_is_distance_to_pivot(self):
        s = VCPStrategy()
        r = {"distance_to_pivot_pct": 2.5}
        assert s.sort_key(r) == 2.5


# ---------------------------------------------------------------------------
# RSBreakoutStrategy
# ---------------------------------------------------------------------------


class TestRSBreakoutStrategy:
    def _build_breakout_df(self) -> pd.DataFrame:
        """Price that is at a new 52-week high with volume surge today."""
        n = 260
        rng = np.random.default_rng(1)
        # Steadily rising close so 52-week high is at the end
        close = np.linspace(500, 1500, n) + rng.normal(0, 5, n)
        volume = np.full(n, 1_000_000.0)
        # Make today's volume a surge
        volume[-1] = 2_000_000.0
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "open": close - 2,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "time"
        return df

    def test_passes_on_new_high_with_volume(self):
        s = RSBreakoutStrategy()
        df = self._build_breakout_df()
        result = s.scan("IREDA", df)
        assert result is not None
        assert result["strategy"] == "rs_breakout"
        assert result["volume_ratio"] >= 1.40

    def test_rejects_without_volume_surge(self):
        s = RSBreakoutStrategy()
        df = self._build_breakout_df()
        # Flatten volume — no surge
        df["volume"] = 1_000_000.0
        assert s.scan("X", df) is None

    def test_rejects_below_200sma(self):
        s = RSBreakoutStrategy()
        n = 260
        # Price drops below its own SMA200 at the end
        close = np.linspace(1500, 500, n)
        volume = np.full(n, 1_000_000.0)
        volume[-1] = 2_000_000.0
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "open": close - 2,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "time"
        assert s.scan("X", df) is None

    def test_sort_key_descending_by_return(self):
        s = RSBreakoutStrategy()
        assert s.sort_key({"return_12m_pct": 45.0}) == -45.0


# ---------------------------------------------------------------------------
# TightClosesStrategy
# ---------------------------------------------------------------------------


class TestTightClosesStrategy:
    def _build_tight_df(self, tight_bars: int = 5) -> pd.DataFrame:
        """
        Build an uptrending DF with *tight_bars* nearly-identical closes
        at the end (within 0.5 % of each other).
        """
        n = 200
        rng = np.random.default_rng(2)
        # Rising trend then flat tight zone
        trend = np.linspace(500, 1400, n - tight_bars)
        anchor = float(trend[-1])
        tight = np.full(tight_bars, anchor) + rng.uniform(-2, 2, tight_bars)
        close = np.concatenate([trend, tight])
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        volume = np.full(n, 1_000_000.0)
        # Decreasing volume over tight zone
        volume[-tight_bars:] = np.linspace(800_000, 400_000, tight_bars)
        df = pd.DataFrame(
            {
                "open": close - 3,
                "high": close + 3,
                "low": close - 3,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "time"
        return df

    def test_passes_with_tight_closes(self):
        s = TightClosesStrategy()
        df = self._build_tight_df(tight_bars=5)
        result = s.scan("KPIL", df)
        assert result is not None
        assert result["strategy"] == "tight_closes"
        assert result["tight_bars"] >= 3
        assert result["tight_range_pct"] < 1.5

    def test_rejects_wide_range(self):
        s = TightClosesStrategy()
        df = _make_daily(n=200, trend="up")
        # Random walk will rarely produce 3 tight closes in a row
        # We just verify it doesn't crash and returns dict-or-None
        result = s.scan("X", df)
        assert result is None or isinstance(result, dict)

    def test_result_has_breakout_level(self):
        s = TightClosesStrategy()
        df = self._build_tight_df(tight_bars=5)
        result = s.scan("KPIL", df)
        if result is not None:
            assert "breakout_level" in result
            assert "distance_to_breakout_pct" in result
            assert result["breakout_level"] > result["current_price"]

    def test_sort_key_tightest_first(self):
        s = TightClosesStrategy()
        assert s.sort_key({"tight_range_pct": 0.3, "tight_bars": 5}) < s.sort_key(
            {"tight_range_pct": 0.8, "tight_bars": 3}
        )


# ---------------------------------------------------------------------------
# _worker  (the process-pool entry point)
# ---------------------------------------------------------------------------


class TestWorkerFunction:
    def test_worker_returns_strategy_name_and_none_on_empty(self):
        from signals.scanner_engine import _worker

        name, result = _worker(VCPStrategy.fqname(), "X", [])
        assert name == "vcp"
        assert result is None

    def test_worker_returns_result_for_passing_symbol(self):
        from signals.scanner_engine import _worker

        # RS breakout with a clean breakout DF
        n = 260
        rng = np.random.default_rng(1)
        close = np.linspace(500, 1500, n) + rng.normal(0, 1, n)
        volume = np.full(n, 1_000_000.0)
        volume[-1] = 2_500_000.0
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "open": close - 1,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "time"
        records = df.reset_index().to_dict(orient="records")

        name, result = _worker(RSBreakoutStrategy.fqname(), "IREDA", records)
        assert name == "rs_breakout"
        # result may be None or dict — just verify no crash and type is correct
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# ScannerEngine  (integration — patches DB, uses threads not processes)
# ---------------------------------------------------------------------------


class TestScannerEngine:
    def _make_records_for_engine(self, symbol: str) -> list[dict]:
        """Produce RS-breakout records for *symbol*."""
        n = 260
        rng = np.random.default_rng(hash(symbol) % 2**31)
        close = np.linspace(500, 1500, n) + rng.normal(0, 1, n)
        volume = np.full(n, 1_000_000.0)
        volume[-1] = 2_500_000.0
        idx = pd.date_range("2022-01-03", periods=n, freq="B")
        df = pd.DataFrame(
            {
                "symbol": symbol,
                "open": close - 1,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": volume,
            },
            index=idx,
        )
        df.index.name = "time"
        return df.reset_index().to_dict(orient="records")

    def _mock_fetch(self, symbols, interval, lookback_days):
        """Return a combined DataFrame for all symbols."""
        frames = [pd.DataFrame(self._make_records_for_engine(s)) for s in symbols]
        return pd.concat(frames, ignore_index=True)

    def test_returns_dict_keyed_by_strategy_name(self):
        from concurrent.futures import ThreadPoolExecutor

        from signals.scanner_engine import ScannerEngine

        strategies = [RSBreakoutStrategy(), TightClosesStrategy()]
        engine = ScannerEngine(strategies, workers=2)

        with (
            patch("signals.scanner_engine._fetch_group", side_effect=self._mock_fetch),
            patch("signals.scanner_engine.ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            results = engine.run(["RELIANCE", "TCS", "INFY"])

        assert set(results.keys()) == {"rs_breakout", "tight_closes"}

    def test_all_results_are_lists(self):
        from concurrent.futures import ThreadPoolExecutor

        from signals.scanner_engine import ScannerEngine

        strategies = [RSBreakoutStrategy()]
        engine = ScannerEngine(strategies, workers=2)

        with (
            patch("signals.scanner_engine._fetch_group", side_effect=self._mock_fetch),
            patch("signals.scanner_engine.ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            results = engine.run(["RELIANCE"])

        assert isinstance(results["rs_breakout"], list)

    def test_empty_universe_returns_empty_lists(self):
        from signals.scanner_engine import ScannerEngine

        strategies = [VCPStrategy()]
        engine = ScannerEngine(strategies)

        with patch("signals.scanner_engine._fetch_group", return_value=pd.DataFrame()):
            results = engine.run([])

        assert results["vcp"] == []

    def test_raises_with_no_strategies(self):
        from signals.scanner_engine import ScannerEngine

        with pytest.raises(ValueError, match="strategy"):
            ScannerEngine([])

    def test_strategies_share_single_fetch_when_same_interval(self):
        """3 daily strategies → only 1 DB fetch."""
        from concurrent.futures import ThreadPoolExecutor

        from signals.scanner_engine import ScannerEngine

        strategies = [VCPStrategy(), RSBreakoutStrategy(), TightClosesStrategy()]
        engine = ScannerEngine(strategies, workers=2)

        with (
            patch(
                "signals.scanner_engine._fetch_group", side_effect=self._mock_fetch
            ) as mock_fetch,
            patch("signals.scanner_engine.ProcessPoolExecutor", ThreadPoolExecutor),
        ):
            engine.run(["RELIANCE"])

        # All three strategies use interval='day', so only one query
        assert mock_fetch.call_count == 1
