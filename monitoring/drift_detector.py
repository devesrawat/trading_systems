"""
Concept drift detection for XGBoost signal model features.

ConceptDriftDetector compares current feature distributions against a
reference distribution saved at model training time using a KS test.

Workflow:
  Training time:
    detector = ConceptDriftDetector()
    detector.fit(training_feature_df)          # saves reference to Redis

  Inference time (daily post-market):
    detector = ConceptDriftDetector()
    result = detector.check(live_feature_df)   # {feature: p_value}
    if detector.is_drifting(live_feature_df):
        send_alert(...)
"""

from __future__ import annotations

import json

import pandas as pd
import structlog

log = structlog.get_logger(__name__)

_MAX_REFERENCE_ROWS = 500  # rows to save from training set
_DEFAULT_ALPHA = 0.05  # KS test significance level


class ConceptDriftDetector:
    """
    KS-test based concept drift detector.

    Uses two-sample KS test: training reference rows vs recent live rows.
    """

    def __init__(self) -> None:
        self._reference: dict[str, list[float]] = {}

    # ------------------------------------------------------------------
    # fit — call at training time to save reference distribution
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, features: list[str]) -> None:
        """
        Save a random sample of training rows as the drift reference.

        Should be called from WalkForwardTrainer after the final fold
        so the reference reflects the model's training distribution.

        Args:
            df:       Training DataFrame (DatetimeIndex, FEATURE_COLUMNS).
            features: Column names to track for drift.
        """
        n = min(_MAX_REFERENCE_ROWS, len(df))
        sample = (
            df[features].dropna().sample(n=n, random_state=42)
            if len(df) >= n
            else df[features].dropna()
        )

        self._reference = {col: sample[col].tolist() for col in features if col in sample.columns}

        try:
            from data.redis_keys import RedisKeys
            from data.store import get_redis

            payload = json.dumps(self._reference)
            get_redis().set(RedisKeys.DRIFT_REFERENCE, payload)
            log.info("drift_reference_saved", features=len(self._reference), rows=n)
        except Exception as exc:
            log.warning("drift_reference_save_failed", error=str(exc))

    # ------------------------------------------------------------------
    # check — call daily post-market
    # ------------------------------------------------------------------

    def check(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Run KS test per feature between reference and live distributions.

        Returns:
            {feature_name: p_value}  — small p → drift detected.
        """
        self._load_reference()
        if not self._reference:
            log.warning("drift_check_skipped_no_reference")
            return {}

        from scipy.stats import ks_2samp

        results: dict[str, float] = {}
        for feature, ref_values in self._reference.items():
            if feature not in df.columns:
                continue
            live_values = df[feature].dropna().tolist()
            if len(live_values) < 10:
                continue
            _, p_value = ks_2samp(ref_values, live_values)
            results[feature] = float(p_value)

        drifting = {f: p for f, p in results.items() if p < _DEFAULT_ALPHA}
        if drifting:
            log.warning("concept_drift_detected", drifting_features=list(drifting.keys()))
        else:
            log.info("drift_check_clean", features_checked=len(results))

        return results

    def is_drifting(self, df: pd.DataFrame, alpha: float = _DEFAULT_ALPHA) -> bool:
        """Return True if any feature p-value < alpha (drift detected)."""
        results = self.check(df)
        return any(p < alpha for p in results.values())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_reference(self) -> None:
        if self._reference:
            return  # already loaded

        try:
            from data.redis_keys import RedisKeys
            from data.store import get_redis

            raw = get_redis().get(RedisKeys.DRIFT_REFERENCE)
            if raw:
                self._reference = json.loads(raw)
        except Exception as exc:
            log.debug("drift_reference_load_failed", error=str(exc))
