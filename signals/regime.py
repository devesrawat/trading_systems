"""
Volatility regime detection.

Two implementations:
  VolRegimeDetector  — 2-state Gaussian HMM (hmmlearn). Requires fitting.
  SimpleVolRegime    — rule-based fallback using realized vol vs rolling median.

Both expose:
  fit(returns_series)          → train
  predict(returns_series)      → array of 0/1 regime labels
  label_regimes(df)            → df copy with 'regime' + 'vol_regime_hmm' columns
  save(path) / load(path)      → joblib serialisation
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import structlog
from hmmlearn import hmm

log = structlog.get_logger(__name__)

RegimeLabel = Literal["high_vol", "low_vol", "normal"]

_DEFAULT_LOOKBACK = 20
_DEFAULT_MULTIPLIER = 1.5


# ---------------------------------------------------------------------------
# HMM-based detector
# ---------------------------------------------------------------------------

class VolRegimeDetector:
    """
    Two-state Gaussian HMM trained on log returns.

    State 0 → low volatility
    State 1 → high volatility

    The mapping is established at fit time by comparing the variance
    of each HMM state: the state with higher variance is labelled 1.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100, random_state: int = 42) -> None:
        self._n_states = n_states
        self._n_iter = n_iter
        self._random_state = random_state
        self._model: hmm.GaussianHMM | None = None
        self._high_vol_state: int = 1   # determined after fit

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, returns_series: pd.Series) -> "VolRegimeDetector":
        """Train the HMM on *returns_series* (daily or intraday log returns)."""
        X = self._to_features(returns_series)
        model = hmm.GaussianHMM(
            n_components=self._n_states,
            covariance_type="full",
            n_iter=self._n_iter,
            random_state=self._random_state,
        )
        model.fit(X)
        self._model = model
        self._high_vol_state = self._identify_high_vol_state()
        log.info("hmm_fitted", n_states=self._n_states, high_vol_state=self._high_vol_state)
        return self

    def _identify_high_vol_state(self) -> int:
        """Return the state index with the largest variance."""
        assert self._model is not None
        variances = [
            float(np.mean(np.diag(self._model.covars_[s])))
            for s in range(self._n_states)
        ]
        return int(np.argmax(variances))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, returns_series: pd.Series) -> np.ndarray:
        """
        Predict regime labels (0=low vol, 1=high vol) for *returns_series*.

        Returns an ndarray of ints with the same length as *returns_series*.
        """
        if not self.is_fitted:
            raise RuntimeError("VolRegimeDetector is not fitted. Call fit() first.")

        X = self._to_features(returns_series)
        raw_states = self._model.predict(X)  # type: ignore[union-attr]

        # Remap so that high_vol_state → 1, everything else → 0
        labels = (raw_states == self._high_vol_state).astype(int)
        return labels

    # ------------------------------------------------------------------
    # label_regimes
    # ------------------------------------------------------------------

    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'regime' (int 0/1) and 'vol_regime_hmm' (int 0/1) columns to *df*.

        Returns a new DataFrame — never mutates the input.
        """
        if not self.is_fitted:
            raise RuntimeError("VolRegimeDetector is not fitted. Call fit() first.")

        returns = df["close"].pct_change().fillna(0)
        labels = self.predict(returns)

        result = df.copy()
        result["vol_regime_hmm"] = labels
        result["regime"] = labels
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        payload = {
            "model": self._model,
            "high_vol_state": self._high_vol_state,
            "n_states": self._n_states,
        }
        joblib.dump(payload, path)
        log.info("regime_model_saved", path=path)

    @classmethod
    def load(cls, path: str) -> "VolRegimeDetector":
        """Load a previously saved VolRegimeDetector from *path*."""
        payload = joblib.load(path)
        obj = cls(n_states=payload["n_states"])
        obj._model = payload["model"]
        obj._high_vol_state = payload["high_vol_state"]
        log.info("regime_model_loaded", path=path)
        return obj

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_features(returns_series: pd.Series) -> np.ndarray:
        """Reshape returns into (n_samples, 1) float64 array for hmmlearn."""
        vals = returns_series.fillna(0).values.astype(np.float64)
        return vals.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

class SimpleVolRegime:
    """
    Classify current volatility regime without ML.

    Compares 20-day realised vol against a rolling median:
      > median * multiplier  → 'high_vol'
      < median / multiplier  → 'low_vol'
      else                   → 'normal'
    """

    def __init__(
        self,
        lookback: int = _DEFAULT_LOOKBACK,
        multiplier: float = _DEFAULT_MULTIPLIER,
    ) -> None:
        self._lookback = lookback
        self._multiplier = multiplier

    def classify(self, df: pd.DataFrame) -> RegimeLabel:
        """Return the regime label for the most recent bar in *df*."""
        if df.empty:
            raise ValueError("Cannot classify an empty DataFrame.")

        vol = self._realised_vol(df)
        rolling_median = vol.rolling(60).median()

        current_vol = vol.iloc[-1]
        median_vol = rolling_median.iloc[-1]

        if pd.isna(current_vol) or pd.isna(median_vol):
            return "normal"

        if current_vol > median_vol * self._multiplier:
            return "high_vol"
        if current_vol < median_vol / self._multiplier:
            return "low_vol"
        return "normal"

    def label_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'regime' column with per-bar classification to a copy of *df*.
        """
        if df.empty:
            raise ValueError("Cannot label an empty DataFrame.")

        vol = self._realised_vol(df)
        rolling_median = vol.rolling(60).median()

        def _classify_row(row_vol: float, med: float) -> RegimeLabel:
            if pd.isna(row_vol) or pd.isna(med):
                return "normal"
            if row_vol > med * self._multiplier:
                return "high_vol"
            if row_vol < med / self._multiplier:
                return "low_vol"
            return "normal"

        regimes = pd.Series(
            [_classify_row(v, m) for v, m in zip(vol, rolling_median)],
            index=df.index,
            name="regime",
        )
        result = df.copy()
        result["regime"] = regimes
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _realised_vol(self, df: pd.DataFrame) -> pd.Series:
        pct_chg = df["close"].astype(float).pct_change()
        return pct_chg.rolling(self._lookback).std() * sqrt(252)


# ---------------------------------------------------------------------------
# Multi-factor regime detector (Phase 1 — M3 spec)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas_ta as _pta


class RegimeState(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    CHOPPY = "choppy"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True)
class RegimeMetrics:
    state: RegimeState
    adx_14: float
    vix: float
    nifty_200dma_slope: float    # % change of 200-DMA over 20 days
    realized_vol_ratio: float    # 20d realized vol / 252d realized vol
    score: float                 # 0.0-1.0 confidence in classification


class RegimeDetector:
    """
    Multi-factor regime classification.
    Thresholds tuned on Nifty 2015-2025 per architecture M3 spec.

    Usage::

        detector = RegimeDetector()
        metrics = detector.detect(nifty_df, india_vix=14.5)
        if detector.should_suppress_new_entries(metrics.state):
            return  # skip signal emission
        size_mult = detector.get_position_size_multiplier(metrics.state)
    """

    ADX_TREND_THRESHOLD = 25.0
    ADX_CHOPPY_THRESHOLD = 20.0
    VIX_HIGH_THRESHOLD = 22.0
    VOL_EXPANSION_RATIO = 1.5

    def detect(self, nifty_df: pd.DataFrame, india_vix: float) -> RegimeMetrics:
        """nifty_df must have [open, high, low, close, volume]; minimum 252 bars."""
        adx = self._compute_adx(nifty_df)
        vol_ratio = self._compute_vol_ratio(nifty_df)
        dma_slope = self._compute_200dma_slope(nifty_df)
        state = self._classify(adx, india_vix, vol_ratio, dma_slope)
        score = self._compute_confidence(adx, india_vix, vol_ratio)
        return RegimeMetrics(
            state=state, adx_14=adx, vix=india_vix,
            nifty_200dma_slope=dma_slope, realized_vol_ratio=vol_ratio, score=score,
        )

    def get_position_size_multiplier(self, state: RegimeState) -> float:
        return {
            RegimeState.TRENDING_BULL: 1.0,
            RegimeState.TRENDING_BEAR: 0.5,
            RegimeState.CHOPPY: 0.0,
            RegimeState.HIGH_VOL: 0.5,
        }[state]

    def should_suppress_new_entries(self, state: RegimeState) -> bool:
        return state == RegimeState.CHOPPY

    def _compute_adx(self, df: pd.DataFrame) -> float:
        adx_df = _pta.adx(df["high"], df["low"], df["close"], length=14)
        if adx_df is None or adx_df.empty:
            return 0.0
        # pandas_ta returns columns like ADX_14, DMP_14, DMN_14
        adx_col = [c for c in adx_df.columns if c.startswith("ADX")]
        return float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 0.0

    def _compute_vol_ratio(self, df: pd.DataFrame) -> float:
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        vol_20 = float(log_ret.tail(20).std() * np.sqrt(252))
        vol_252 = float(log_ret.tail(252).std() * np.sqrt(252))
        return vol_20 / max(vol_252, 1e-6)

    def _compute_200dma_slope(self, df: pd.DataFrame) -> float:
        dma = df["close"].rolling(200).mean()
        valid = dma.dropna()
        if len(valid) < 20:
            return 0.0
        return float((dma.iloc[-1] - dma.iloc[-21]) / dma.iloc[-21] * 100)

    def _classify(self, adx: float, vix: float, vol_ratio: float, dma_slope: float) -> RegimeState:
        if vix > self.VIX_HIGH_THRESHOLD or vol_ratio > self.VOL_EXPANSION_RATIO:
            return RegimeState.HIGH_VOL
        if adx < self.ADX_CHOPPY_THRESHOLD:
            return RegimeState.CHOPPY
        if adx >= self.ADX_TREND_THRESHOLD:
            return RegimeState.TRENDING_BULL if dma_slope > 0 else RegimeState.TRENDING_BEAR
        return RegimeState.TRENDING_BULL if dma_slope > 0 else RegimeState.CHOPPY

    def _compute_confidence(self, adx: float, vix: float, vol_ratio: float) -> float:
        adx_score = min(adx / 40.0, 1.0)
        vol_score = 1.0 - min(abs(vol_ratio - 1.0), 1.0)
        return (adx_score + vol_score) / 2.0
