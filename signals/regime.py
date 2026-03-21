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
