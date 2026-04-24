"""
Feature engineering pipeline for Phase 8 ML integration.

Extracts, validates, and caches features for ensemble model predictions.
Seamlessly integrates with existing signals/features.py machinery.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import structlog

from data.store import get_redis
from signals.features import FEATURE_COLUMNS, build_features

log = structlog.get_logger(__name__)

_FEATURE_CACHE_TTL = 300  # 5 minutes
_FEATURE_CACHE_PREFIX = "trading:features"


class FeatureEngineer:
    """
    Extracts, validates, and caches features for ML model predictions.

    Methods
    -------
    extract_features(symbol, df_ohlcv) → pd.Series
        Extract all 30+ features from OHLCV data
    validate_features(feature_series) → bool
        Check for NaN, outliers, and data quality
    handle_missing_data(feature_series) → pd.Series
        Forward-fill, interpolate, or drop NaN values
    cache_features(symbol, feature_series) → None
        Store features in Redis for fast retrieval
    get_cached_features(symbol) → pd.Series | None
        Retrieve cached features if available and fresh
    """

    def __init__(self, redis_client: Any = None) -> None:
        """
        Initialize FeatureEngineer.

        Parameters
        ----------
        redis_client : optional
            Redis client (injected for testing). Defaults to shared singleton.
        """
        self._redis = redis_client or get_redis()
        self._outlier_thresholds: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, symbol: str, df_ohlcv: pd.DataFrame) -> pd.Series:
        """
        Extract all features from OHLCV data.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., "INFY", "BTC/USDT")
        df_ohlcv : pd.DataFrame
            OHLCV DataFrame with columns: open, high, low, close, volume
            Index must be DatetimeIndex

        Returns
        -------
        pd.Series
            Feature vector with 30+ features, indexed by FEATURE_COLUMNS
            Latest value for each feature.

        Raises
        ------
        ValueError
            If df_ohlcv has insufficient data (< 50 rows)
        """
        if len(df_ohlcv) < 50:
            raise ValueError(f"Insufficient data for {symbol}: {len(df_ohlcv)} rows < 50 minimum")

        try:
            # Use existing feature building logic from signals.features
            features = build_features(df_ohlcv)

            if features is None or features.empty:
                log.warning("feature_extraction_returned_empty", symbol=symbol)
                return pd.Series(dtype=float)

            # Ensure we have exactly the expected features
            missing_features = set(FEATURE_COLUMNS) - set(features.index)
            if missing_features:
                log.warning(
                    "missing_features_after_extraction",
                    symbol=symbol,
                    missing=list(missing_features),
                )

            # Return only the canonical features
            return features[FEATURE_COLUMNS].iloc[-1]

        except Exception as exc:
            log.error("feature_extraction_failed", symbol=symbol, error=str(exc))
            raise

    # ------------------------------------------------------------------
    # Feature validation
    # ------------------------------------------------------------------

    def validate_features(self, feature_series: pd.Series) -> bool:
        """
        Validate feature series for NaN, outliers, and data quality.

        Parameters
        ----------
        feature_series : pd.Series
            Feature vector to validate

        Returns
        -------
        bool
            True if valid, False if issues detected
        """
        if feature_series is None or feature_series.empty:
            log.warning("validate_features_empty_series")
            return False

        # Check for NaN values
        nan_count = feature_series.isna().sum()
        if nan_count > 0:
            log.warning(
                "validate_features_nan_values",
                nan_count=nan_count,
                total_features=len(feature_series),
            )
            return False

        # Check for inf values
        inf_count = np.isinf(feature_series).sum()
        if inf_count > 0:
            log.warning("validate_features_inf_values", inf_count=inf_count)
            return False

        # Check for outliers (beyond 3-sigma)
        mean = feature_series.mean()
        std = feature_series.std()

        if std > 0:
            z_scores = np.abs((feature_series - mean) / std)
            outlier_count = (z_scores > 3).sum()

            if outlier_count > 0:
                log.warning("validate_features_outliers_detected", outlier_count=outlier_count)
                return False

        log.debug("validate_features_passed", feature_count=len(feature_series))
        return True

    # ------------------------------------------------------------------
    # Missing data handling
    # ------------------------------------------------------------------

    def handle_missing_data(self, feature_series: pd.Series) -> pd.Series:
        """
        Handle missing data via forward-fill, interpolation, or dropping.

        Protocol
        --------
        1. If < 10% NaN → forward-fill then interpolate
        2. If 10–50% NaN → forward-fill, then drop remaining
        3. If > 50% NaN → raise ValueError

        Parameters
        ----------
        feature_series : pd.Series
            Feature vector with potential NaN values

        Returns
        -------
        pd.Series
            Cleaned feature vector

        Raises
        ------
        ValueError
            If > 50% values are NaN
        """
        if feature_series is None or feature_series.empty:
            return feature_series

        nan_pct = feature_series.isna().sum() / len(feature_series)

        if nan_pct == 0.0:
            # No missing data
            return feature_series

        if nan_pct > 0.5:
            raise ValueError(f"Too many missing values ({nan_pct:.1%}). Cannot reliably impute.")

        # Forward-fill first
        filled = feature_series.ffill()

        if nan_pct <= 0.1:
            # < 10% NaN: interpolate remaining
            filled = filled.interpolate(method="linear", limit_direction="both")
        else:
            # 10–50% NaN: drop any remaining after ffill
            filled = filled.dropna()

        log.info(
            "missing_data_handled", nan_pct_original=nan_pct, nan_pct_final=filled.isna().sum()
        )
        return filled

    # ------------------------------------------------------------------
    # Redis caching
    # ------------------------------------------------------------------

    def cache_features(self, symbol: str, feature_series: pd.Series) -> None:
        """
        Store features in Redis for fast retrieval in trading loop.

        Parameters
        ----------
        symbol : str
            Trading symbol
        feature_series : pd.Series
            Feature vector to cache
        """
        try:
            if feature_series is None or feature_series.empty:
                log.warning("cache_features_empty_series", symbol=symbol)
                return

            # Convert to JSON
            features_dict = feature_series.to_dict()
            features_json = json.dumps(features_dict)

            # Cache in Redis with TTL
            cache_key = f"{_FEATURE_CACHE_PREFIX}:{symbol}"
            self._redis.setex(cache_key, _FEATURE_CACHE_TTL, features_json)

            log.debug("features_cached", symbol=symbol, features_count=len(feature_series))

        except Exception as exc:
            log.warning("cache_features_failed", symbol=symbol, error=str(exc))

    def get_cached_features(self, symbol: str) -> pd.Series | None:
        """
        Retrieve cached features from Redis if available and fresh.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        pd.Series | None
            Cached feature vector, or None if not found or stale
        """
        try:
            cache_key = f"{_FEATURE_CACHE_PREFIX}:{symbol}"
            cached_json = self._redis.get(cache_key)

            if not cached_json:
                return None

            # Deserialize
            features_dict = json.loads(cached_json)
            features = pd.Series(features_dict)

            log.debug("features_retrieved_from_cache", symbol=symbol, features_count=len(features))
            return features

        except Exception as exc:
            log.warning("get_cached_features_failed", symbol=symbol, error=str(exc))
            return None

    # ------------------------------------------------------------------
    # Integration methods
    # ------------------------------------------------------------------

    def extract_and_validate(
        self, symbol: str, df_ohlcv: pd.DataFrame, use_cache: bool = True
    ) -> pd.Series | None:
        """
        Extract features, validate, and optionally cache in one call.

        Convenience method combining extraction → validation → caching.

        Parameters
        ----------
        symbol : str
            Trading symbol
        df_ohlcv : pd.DataFrame
            OHLCV data
        use_cache : bool
            Whether to cache features in Redis (default True)

        Returns
        -------
        pd.Series | None
            Valid feature vector, or None if extraction/validation failed
        """
        try:
            # Extract
            features = self.extract_features(symbol, df_ohlcv)

            # Validate
            if not self.validate_features(features):
                log.warning("feature_extraction_validation_failed", symbol=symbol)
                return None

            # Cache
            if use_cache:
                self.cache_features(symbol, features)

            return features

        except Exception as exc:
            log.error(
                "extract_and_validate_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None
