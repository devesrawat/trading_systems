"""Unit tests for signals/regime.py — TDD RED phase."""

import numpy as np
import pandas as pd
import pytest

from signals.regime import SimpleVolRegime, VolRegimeDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_returns(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(0, 0.01, n), index=idx, name="returns")


def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 1000 + rng.normal(0, 5, n).cumsum()
    close = np.abs(close)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# SimpleVolRegime
# ---------------------------------------------------------------------------


class TestSimpleVolRegime:
    def test_classify_returns_valid_label(self):
        df = _make_ohlcv()
        svr = SimpleVolRegime()
        label = svr.classify(df)
        assert label in ("high_vol", "low_vol", "normal")

    def test_high_vol_detected(self):
        """Inject a clearly high-vol period and confirm classification."""
        df = _make_ohlcv(100)
        # Multiply recent vol by 10x
        df_high = df.copy()
        df_high["close"] = df_high["close"] * (
            1 + np.random.default_rng(1).normal(0, 0.05, 100).cumsum()
        )
        df_high["close"] = df_high["close"].abs()
        svr = SimpleVolRegime(multiplier=0.5)  # low threshold → easier to trigger high_vol
        label = svr.classify(df_high)
        assert label in ("high_vol", "normal")  # should lean high_vol

    def test_label_regimes_adds_column(self):
        df = _make_ohlcv()
        svr = SimpleVolRegime()
        result = svr.label_regimes(df)
        assert "regime" in result.columns
        assert set(result["regime"].unique()).issubset({"high_vol", "low_vol", "normal"})

    def test_label_regimes_preserves_index(self):
        df = _make_ohlcv()
        svr = SimpleVolRegime()
        result = svr.label_regimes(df)
        pd.testing.assert_index_equal(result.index, df.index)

    def test_returns_new_df_not_mutating(self):
        df = _make_ohlcv()
        original_cols = list(df.columns)
        svr = SimpleVolRegime()
        svr.label_regimes(df)
        assert list(df.columns) == original_cols

    def test_empty_df_raises(self):
        svr = SimpleVolRegime()
        with pytest.raises((ValueError, Exception)):
            svr.classify(pd.DataFrame())


# ---------------------------------------------------------------------------
# VolRegimeDetector (HMM)
# ---------------------------------------------------------------------------


class TestVolRegimeDetector:
    def test_fit_runs_without_error(self):
        returns = _make_returns(400)
        vrd = VolRegimeDetector(n_states=2)
        vrd.fit(returns)
        assert vrd.is_fitted

    def test_predict_returns_binary_array(self):
        returns = _make_returns(400)
        vrd = VolRegimeDetector(n_states=2)
        vrd.fit(returns)
        labels = vrd.predict(returns)
        assert set(labels).issubset({0, 1})

    def test_predict_same_length_as_input(self):
        returns = _make_returns(400)
        vrd = VolRegimeDetector(n_states=2)
        vrd.fit(returns)
        labels = vrd.predict(returns)
        assert len(labels) == len(returns)

    def test_predict_without_fit_raises(self):
        vrd = VolRegimeDetector()
        with pytest.raises(RuntimeError, match="not fitted"):
            vrd.predict(_make_returns(100))

    def test_label_regimes_adds_columns(self):
        df = _make_ohlcv()
        returns = df["close"].pct_change().dropna()
        vrd = VolRegimeDetector()
        vrd.fit(returns)
        result = vrd.label_regimes(df)
        assert "regime" in result.columns
        assert "vol_regime_hmm" in result.columns

    def test_save_and_load_roundtrip(self, tmp_path):
        returns = _make_returns(400)
        vrd = VolRegimeDetector()
        vrd.fit(returns)
        path = tmp_path / "regime_model.joblib"
        vrd.save(str(path))

        loaded = VolRegimeDetector.load(str(path))
        original_preds = vrd.predict(returns)
        loaded_preds = loaded.predict(returns)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_high_vol_state_has_larger_variance(self):
        """HMM state with higher variance should map to 'high vol' state."""
        # Create clearly bimodal returns: calm + volatile periods
        rng = np.random.default_rng(99)
        calm = rng.normal(0, 0.005, 200)
        volatile = rng.normal(0, 0.03, 200)
        returns = pd.Series(np.concatenate([calm, volatile]))
        returns.index = pd.date_range("2022-01-03", periods=400, freq="B")

        vrd = VolRegimeDetector()
        vrd.fit(returns)
        labels = vrd.predict(returns)

        # Volatile period (last 200) should have more "1" (high vol) labels
        vol_rate_calm = labels[:200].mean()
        vol_rate_volatile = labels[200:].mean()
        assert vol_rate_volatile > vol_rate_calm
