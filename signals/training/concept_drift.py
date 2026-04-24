"""
Concept drift detection for trading models.

Detects when the distribution of features or labels has shifted significantly,
triggering emergency retraining.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

log = structlog.get_logger(__name__)


class ConceptDriftDetector:
    """
    Detects concept drift via KL divergence, feature shift, and regime change detection.

    Methods
    -------
    compute_kl_divergence(dist1, dist2) → float
        KL divergence between two distributions.
    detect_feature_shift(train_features, test_features) → dict
        Which features shifted and by how much.
    is_regime_change(kl_divergence, threshold) → bool
        Whether KL divergence exceeds threshold (triggers emergency retraining).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        """
        Initialize drift detector.

        Parameters
        ----------
        threshold : float
            KL divergence threshold for regime change detection (default 0.5).
        """
        self.threshold = threshold
        self.drift_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # KL Divergence
    # ------------------------------------------------------------------

    def compute_kl_divergence(self, dist1: np.ndarray, dist2: np.ndarray, bins: int = 20) -> float:
        """
        Compute KL divergence between two continuous distributions.

        Uses histogram-based binning to discretize continuous distributions,
        then computes KL divergence. Symmetrizes the result using the mean
        of KL(dist1||dist2) and KL(dist2||dist1).

        Parameters
        ----------
        dist1 : np.ndarray
            First distribution (samples or histogram).
        dist2 : np.ndarray
            Second distribution (samples or histogram).
        bins : int
            Number of bins for histogram (default 20).

        Returns
        -------
        float
            Symmetric KL divergence (always >= 0).
        """
        if len(dist1) < 2 or len(dist2) < 2:
            return 0.0

        # Compute joint range for both distributions
        min_val = min(dist1.min(), dist2.min())
        max_val = max(dist1.max(), dist2.max())

        # Avoid division by zero if distributions are identical
        if min_val == max_val:
            return 0.0

        # Bin both distributions with same bins
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        p_hist, _ = np.histogram(dist1, bins=bin_edges)
        q_hist, _ = np.histogram(dist2, bins=bin_edges)

        # Normalize to probability distributions
        p = p_hist / (p_hist.sum() + 1e-10)
        q = q_hist / (q_hist.sum() + 1e-10)

        # Avoid log(0)
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)

        # Symmetric KL divergence: 0.5 * [KL(P||Q) + KL(Q||P)]
        kl_pq = np.sum(p * (np.log(p) - np.log(q)))
        kl_qp = np.sum(q * (np.log(q) - np.log(p)))
        return 0.5 * (kl_pq + kl_qp)

    # ------------------------------------------------------------------
    # Feature Shift Detection
    # ------------------------------------------------------------------

    def detect_feature_shift(
        self, train_features: pd.DataFrame, test_features: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Detect which features have shifted and by how much.

        Computes KL divergence for each feature between train and test sets.
        Returns sorted list of features by shift magnitude.

        Parameters
        ----------
        train_features : pd.DataFrame
            Training features.
        test_features : pd.DataFrame
            Test features.

        Returns
        -------
        dict[str, Any]
            {
                "features_shifted": [("feature_name", kl_divergence), ...],
                "max_shift": float,
                "mean_shift": float,
                "n_features_shifted": int,
            }
        """
        shifted_features = []

        for col in train_features.columns:
            if col not in test_features.columns:
                continue

            train_data = train_features[col].dropna().values
            test_data = test_features[col].dropna().values

            if len(train_data) < 2 or len(test_data) < 2:
                continue

            kl_div = self.compute_kl_divergence(train_data, test_data)
            shifted_features.append((col, kl_div))

        shifted_features.sort(key=lambda x: x[1], reverse=True)

        mean_shift = np.mean([kl for _, kl in shifted_features]) if shifted_features else 0.0
        max_shift = shifted_features[0][1] if shifted_features else 0.0
        n_shifted = sum(1 for _, kl in shifted_features if kl > 0.1)

        return {
            "features_shifted": shifted_features,
            "max_shift": max_shift,
            "mean_shift": mean_shift,
            "n_features_shifted": n_shifted,
        }

    # ------------------------------------------------------------------
    # Regime Change Detection
    # ------------------------------------------------------------------

    def is_regime_change(self, kl_divergence: float, threshold: float | None = None) -> bool:
        """
        Detect if KL divergence indicates a regime change.

        A regime change triggers emergency retraining.

        Parameters
        ----------
        kl_divergence : float
            KL divergence value.
        threshold : float, optional
            Override the default threshold.

        Returns
        -------
        bool
            True if regime change detected (KL > threshold).
        """
        thresh = threshold if threshold is not None else self.threshold
        is_change = kl_divergence > thresh
        log.info(
            "regime_change_check",
            kl_divergence=round(kl_divergence, 4),
            threshold=thresh,
            is_regime_change=is_change,
        )
        return is_change

    # ------------------------------------------------------------------
    # Label Distribution Shift
    # ------------------------------------------------------------------

    def detect_label_shift(self, train_labels: pd.Series, test_labels: pd.Series) -> dict[str, Any]:
        """
        Detect shifts in label distribution (class imbalance changes).

        Parameters
        ----------
        train_labels : pd.Series
            Training labels (0/1).
        test_labels : pd.Series
            Test labels (0/1).

        Returns
        -------
        dict[str, Any]
            {
                "train_pos_rate": float,
                "test_pos_rate": float,
                "shift_magnitude": float,
                "is_significant_shift": bool,
            }
        """
        train_pos_rate = train_labels.mean()
        test_pos_rate = test_labels.mean()
        shift_mag = abs(train_pos_rate - test_pos_rate)

        # Chi-squared test for independence
        contingency = np.array(
            [
                [
                    (train_labels == 0).sum(),
                    (train_labels == 1).sum(),
                ],
                [
                    (test_labels == 0).sum(),
                    (test_labels == 1).sum(),
                ],
            ]
        )

        chi2_stat, p_value = self._chi2_test(contingency)

        return {
            "train_pos_rate": train_pos_rate,
            "test_pos_rate": test_pos_rate,
            "shift_magnitude": shift_mag,
            "chi2_statistic": chi2_stat,
            "p_value": p_value,
            "is_significant_shift": p_value < 0.05,
        }

    @staticmethod
    def _chi2_test(contingency: np.ndarray) -> tuple[float, float]:
        """
        Simple chi-squared test for 2x2 contingency table.

        Parameters
        ----------
        contingency : np.ndarray
            2x2 contingency table.

        Returns
        -------
        tuple[float, float]
            (chi2 statistic, p-value)
        """
        n = contingency.sum()
        expected = np.outer(contingency.sum(axis=1), contingency.sum(axis=0)) / n

        chi2 = np.sum((contingency - expected) ** 2 / (expected + 1e-10))
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return float(chi2), float(p_value)

    # ------------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------------

    def log_drift_metrics(
        self,
        timestamp: pd.Timestamp,
        feature_shift: dict[str, Any],
        label_shift: dict[str, Any],
        drift_triggered: bool,
    ) -> None:
        """
        Log drift metrics for monitoring and replay.

        Parameters
        ----------
        timestamp : pd.Timestamp
            Time of drift check.
        feature_shift : dict
            Feature shift metrics.
        label_shift : dict
            Label shift metrics.
        drift_triggered : bool
            Whether emergency retraining was triggered.
        """
        record = {
            "timestamp": timestamp,
            "max_feature_shift": feature_shift.get("max_shift", 0.0),
            "mean_feature_shift": feature_shift.get("mean_shift", 0.0),
            "label_shift_magnitude": label_shift.get("shift_magnitude", 0.0),
            "drift_triggered": drift_triggered,
        }

        self.drift_history.append(record)

        if drift_triggered:
            log.warning(
                "drift_emergency_trigger",
                timestamp=timestamp,
                max_feature_shift=round(feature_shift.get("max_shift", 0.0), 4),
                label_shift=round(label_shift.get("shift_magnitude", 0.0), 4),
            )

    def get_drift_history(self) -> pd.DataFrame:
        """
        Get drift history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Drift metrics over time.
        """
        if not self.drift_history:
            return pd.DataFrame()
        return pd.DataFrame(self.drift_history).set_index("timestamp")
