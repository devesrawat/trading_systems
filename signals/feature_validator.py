"""
Feature validation and quality checks.

FeatureValidator class provides:
- Schema validation (all expected columns present)
- NaN/outlier detection
- Stationarity testing (ADF)
- Output shape and type validation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

log = structlog.get_logger(__name__)


class FeatureValidator:
    """Validates feature quality and schema consistency."""

    def __init__(self, feature_schema: list[str] | None = None) -> None:
        """
        Initialize validator with optional feature schema.

        Parameters
        ----------
        feature_schema : list[str], optional
            Expected feature column names. If provided, validate_completeness()
            will check that all are present.
        """
        self.feature_schema = feature_schema or []

    def validate_feature_output(
        self,
        feature_name: str,
        series: pd.Series,
    ) -> tuple[bool, str]:
        """
        Validate a single feature output.

        Parameters
        ----------
        feature_name : str
            Name of the feature (for logging).
        series : pd.Series
            Feature values to validate.

        Returns
        -------
        tuple of (is_valid, error_message)
            - is_valid: True if all checks pass.
            - error_message: Human-readable error if validation fails.
        """
        # Check type
        if not isinstance(series, pd.Series):
            return False, f"{feature_name}: Expected pd.Series, got {type(series)}"

        # Check dtype is numeric
        if not np.issubdtype(series.dtype, np.number):
            return False, f"{feature_name}: Expected numeric dtype, got {series.dtype}"

        # Check all-NaN (likely computation error)
        if series.isna().all():
            return False, f"{feature_name}: Series is entirely NaN"

        # Check for infinite values
        if np.isinf(series).any():
            inf_count = np.isinf(series).sum()
            return False, f"{feature_name}: Contains {inf_count} infinite values"

        return True, ""

    def check_completeness(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Check which features are missing or have high NaN rate.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix to validate.

        Returns
        -------
        dict with keys:
            - missing_columns: list of expected columns not in df
            - high_nan_columns: dict[column_name] = nan_rate (>50%)
            - present_columns: list of feature columns present
        """
        missing = set(self.feature_schema) - set(df.columns)
        present = [c for c in self.feature_schema if c in df.columns]

        high_nan = {}
        for col in present:
            nan_rate = df[col].isna().sum() / len(df)
            if nan_rate > 0.5:
                high_nan[col] = float(nan_rate)

        return {
            "missing_columns": list(missing),
            "high_nan_columns": high_nan,
            "present_columns": present,
        }

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> dict[str, list[int]]:
        """
        Detect outliers in feature columns.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix.
        method : str
            "iqr" (Interquartile Range, default) or "zscore".
        threshold : float
            Threshold multiplier for outlier detection:
            - IQR: flag if |value| > Q1 - threshold*IQR or Q3 + threshold*IQR
            - z-score: flag if |z| > threshold (default 3.0 = ~99.7%)

        Returns
        -------
        dict[column_name] = list of outlier row indices
            Empty dict if no outliers detected.
        """
        outliers = {}

        for col in df.columns:
            if df[col].dtype not in [np.float64, np.float32, np.int64, np.int32]:
                continue

            series = df[col].dropna()
            if series.empty:
                continue

            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                if IQR == 0:
                    continue
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                mask = (df[col] < lower) | (df[col] > upper)
            elif method == "zscore":
                z_scores = np.abs(stats.zscore(series))
                outlier_indices = series.index[z_scores > threshold]
                mask = df.index.isin(outlier_indices)
            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_rows = df.index[mask].tolist()
            if outlier_rows:
                outliers[col] = outlier_rows

        return outliers

    def check_stationarity(
        self,
        series: pd.Series,
        feature_name: str = "feature",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Augmented Dickey-Fuller test for stationarity.

        Parameters
        ----------
        series : pd.Series
            Time series to test.
        feature_name : str
            Name of the feature (for logging).
        alpha : float
            Significance level (default 0.05).

        Returns
        -------
        dict with keys:
            - stationary: bool, True if series is stationary at alpha level
            - adf_statistic: float, ADF test statistic
            - p_value: float, p-value from ADF test
            - critical_values: dict of critical values at different levels
            - n_lags: int, number of lags used
        """
        from statsmodels.tsa.stattools import adfuller

        clean_series = series.dropna()
        if len(clean_series) < 10:
            return {
                "stationary": False,
                "adf_statistic": np.nan,
                "p_value": np.nan,
                "critical_values": {},
                "n_lags": 0,
                "note": "Series too short for ADF test",
            }

        try:
            result = adfuller(clean_series, autolag="AIC")
            adf_stat, p_value, n_lags, _, crit_vals, _ = result

            return {
                "stationary": p_value < alpha,
                "adf_statistic": float(adf_stat),
                "p_value": float(p_value),
                "critical_values": crit_vals,
                "n_lags": int(n_lags),
            }
        except Exception as e:
            log.warning("adf_test_failed", feature=feature_name, error=str(e))
            return {
                "stationary": False,
                "adf_statistic": np.nan,
                "p_value": np.nan,
                "critical_values": {},
                "n_lags": 0,
                "error": str(e),
            }

    def validate_feature_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        Detect highly correlated feature pairs (potential redundancy).

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix.
        threshold : float
            Correlation threshold above which to flag (default 0.95).

        Returns
        -------
        dict[feature_name] = list of (correlated_feature, correlation)
            Only pairs with |correlation| >= threshold are included.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        redundant = {}
        for i, col1 in enumerate(corr_matrix.columns):
            for col2 in corr_matrix.columns[i + 1 :]:
                corr_val = corr_matrix.loc[col1, col2]
                if corr_val >= threshold:
                    if col1 not in redundant:
                        redundant[col1] = []
                    redundant[col1].append((col2, float(corr_val)))

        return redundant
