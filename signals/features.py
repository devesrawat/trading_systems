"""
Feature engineering pipeline.

Entry point: build_features(df) → pd.DataFrame

All features use pandas-ta. No look-ahead bias — every indicator
only uses data available at the bar's close time.

FEATURE_COLUMNS: the exact list used by XGBoost at inference.
LABEL_COLUMNS:   training-only columns — never passed to the model.
"""

from __future__ import annotations

import io
import pickle  # nosec: B403 - Used for backward-compatible caching only
from math import sqrt

import numpy as np
import pandas as pd
import pandas_ta as ta
import structlog

from data.redis_keys import RedisKeys
from data.store import get_redis

log = structlog.get_logger(__name__)

# Feature cache: {(symbol, interval): (features_df, timestamp)}
_FEATURE_CACHE: dict[tuple[str, str], tuple[pd.DataFrame, float]] = {}
_FEATURE_CACHE_TTL = 86400  # 24 hours


def _get_feature_cache_key(symbol: str, date: str) -> str:
    """Generate Redis cache key for features."""
    return RedisKeys.ml_features(symbol, date)


def _try_get_cached_features(symbol: str, date: str) -> pd.DataFrame | None:
    """
    Attempt to fetch cached features from Redis.

    Parameters
    ----------
    symbol : str
        Stock symbol
    date : str
        Date (YYYY-MM-DD format)

    Returns
    -------
    pd.DataFrame or None
        Cached features if found, None otherwise
    """
    try:
        redis = get_redis()
        key = _get_feature_cache_key(symbol, date)
        cached = redis.get(key)
        if cached:
            df = pd.read_json(io.BytesIO(cached))
            log.debug("feature_cache_hit", symbol=symbol, date=date)
            return df
    except Exception as e:
        log.warning("feature_cache_fetch_failed", symbol=symbol, error=str(e))
    return None


def _cache_features_redis(symbol: str, date: str, features_df: pd.DataFrame) -> None:
    """
    Cache computed features in Redis with 24h TTL.

    Parameters
    ----------
    symbol : str
        Stock symbol
    date : str
        Date (YYYY-MM-DD format)
    features_df : pd.DataFrame
        Computed feature DataFrame
    """
    try:
        redis = get_redis()
        key = _get_feature_cache_key(symbol, date)
        serialized = features_df.to_json().encode()
        redis.setex(key, _FEATURE_CACHE_TTL, serialized)
        log.debug("feature_cached", symbol=symbol, date=date, ttl_hours=24)
    except Exception as e:
        log.warning("feature_cache_write_failed", symbol=symbol, error=str(e))


def _set_cached_features_in_redis(cache_key: str, features: pd.DataFrame) -> None:
    """Store features in Redis cache with TTL."""
    try:
        redis = get_redis()
        redis.setex(cache_key, _FEATURE_CACHE_TTL, pickle.dumps(features))
        log.debug("feature_cache_set", cache_key=cache_key)
    except Exception as exc:
        log.debug("redis_cache_write_error", error=str(exc))


# Warm-up period: longest indicator look-back (52-week high = 252 bars)
_WARMUP = 252


# ---------------------------------------------------------------------------
# Column name registries
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    # Momentum
    "rsi_14",
    "rsi_7",
    "macd",
    "macd_signal",
    "macd_hist",
    "macd_cross",
    "mom_10",
    "roc_5",
    "roc_21",
    # Volatility
    "atr_14",
    "atr_pct",
    "bb_upper",
    "bb_mid",
    "bb_lower",
    "bb_position",
    "realized_vol_10",
    "realized_vol_20",
    # Volume
    "volume_zscore_20",
    "obv",
    "obv_slope",
    "vwap_dev",
    # Trend
    "ema_9",
    "ema_21",
    "ema_50",
    "ema_cross_9_21",
    "price_vs_ema50",
    "adx_14",
    "di_plus",
    "di_minus",
    # Mean reversion
    "zscore_20",
    "distance_from_52w_high",
    "distance_from_52w_low",
    # Regime
    "vol_regime",
]

LABEL_COLUMNS: list[str] = ["forward_return_5d", "label"]

# Auxiliary features added in Phase 1 — NOT yet part of FEATURE_COLUMNS because
# existing production models were trained without them. Merge into FEATURE_COLUMNS
# only after retraining on the augmented schema.
#
# build_features() appends these columns when the optional params are provided.
# Callers that pass all four params get an enriched DataFrame; callers that omit
# them get NaN-filled columns so downstream code can safely select either schema.
AUXILIARY_FEATURE_COLUMNS: list[str] = [
    "fii_net_cash_norm",  # FII net cash flow / 1e5 (normalised INR crore)
    "india_vix",  # India VIX level
    "sentiment_score",  # FinBERT sentiment for the symbol (-1..1)
    "regime_code",  # RegimeState ordinal (0=TRENDING_BULL, …, 3=HIGH_VOL)
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    include_labels: bool = False,
    fii_net_cash: float | None = None,
    india_vix: float | None = None,
    sentiment_score: float | None = None,
    regime_code: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Convert raw OHLCV DataFrame into the XGBoost feature vector.

    Parameters
    ----------
    df:
        DataFrame with columns open, high, low, close, volume and a
        DatetimeIndex named 'time'.
    include_labels:
        If True, append forward_return_5d and label columns (training only).
        Must be False at inference time.
    fii_net_cash:
        FII net cash flow in INR crore (from NSEDataScraper). Normalised
        internally to units of 1e5. Populates ``fii_net_cash_norm``.
    india_vix:
        India VIX level. Populates ``india_vix``.
    sentiment_score:
        FinBERT sentiment for this symbol in [-1, 1]. Populates
        ``sentiment_score``.
    regime_code:
        Ordinal encoding of RegimeState (0=TRENDING_BULL, 1=TRENDING_BEAR,
        2=CHOPPY, 3=HIGH_VOL). Populates ``regime_code``.
    use_cache:
        If True, try to fetch from Redis cache before computing (24h TTL).

    Returns
    -------
    pd.DataFrame
        Feature matrix with DatetimeIndex preserved. NaN warm-up rows dropped.
        Always includes ``close`` as a pass-through column for execution pricing.
        Auxiliary columns (AUXILIARY_FEATURE_COLUMNS) are always present —
        they contain NaN when the corresponding argument is not provided.
    """
    _validate_input(df)

    # Check Redis cache if enabled (symbol is passed via df._symbol attribute)
    symbol: str | None = getattr(df, "_symbol", None)
    cache_key: str | None = None
    if use_cache and symbol:
        last_index = df.index[-1]
        date_str = (
            last_index.strftime("%Y-%m-%d")
            if hasattr(last_index, "strftime")
            else str(last_index)[:10]
        )
        cache_key = _get_feature_cache_key(symbol, date_str)
        cached = _try_get_cached_features(symbol, date_str)
        if cached is not None:
            return cached

    out = pd.DataFrame(index=df.index)

    # Single astype() call at start — optimized vs multiple calls per column
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["volume"].astype("float64")

    # ------------------------------------------------------------------ #
    # MOMENTUM
    # ------------------------------------------------------------------ #
    out["rsi_14"] = ta.rsi(close, length=14)
    out["rsi_7"] = ta.rsi(close, length=7)

    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is None:
        out["macd"] = np.nan
        out["macd_hist"] = np.nan
        out["macd_signal"] = np.nan
    else:
        out["macd"] = macd_df.iloc[:, 0]
        out["macd_hist"] = macd_df.iloc[:, 1]
        out["macd_signal"] = macd_df.iloc[:, 2]
    out["macd_cross"] = np.sign(out["macd"] - out["macd_signal"])

    out["mom_10"] = ta.mom(close, length=10)
    out["roc_5"] = ta.roc(close, length=5)
    out["roc_21"] = ta.roc(close, length=21)

    # ------------------------------------------------------------------ #
    # VOLATILITY
    # ------------------------------------------------------------------ #
    atr = ta.atr(high, low, close, length=14)
    if atr is None:
        out["atr_14"] = np.nan
        out["atr_pct"] = np.nan
    else:
        out["atr_14"] = atr
        out["atr_pct"] = atr / close

    bb = ta.bbands(close, length=20, std=2.0)
    if bb is None:
        out["bb_lower"] = np.nan
        out["bb_mid"] = np.nan
        out["bb_upper"] = np.nan
        out["bb_position"] = np.nan
    else:
        out["bb_lower"] = bb.iloc[:, 0]
        out["bb_mid"] = bb.iloc[:, 1]
        out["bb_upper"] = bb.iloc[:, 2]
        bb_range = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
        out["bb_position"] = (close - out["bb_lower"]) / bb_range

    pct_chg = close.pct_change()
    out["realized_vol_10"] = pct_chg.rolling(10).std() * sqrt(252)
    out["realized_vol_20"] = pct_chg.rolling(20).std() * sqrt(252)

    # ------------------------------------------------------------------ #
    # VOLUME
    # ------------------------------------------------------------------ #
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std().replace(0, np.nan)
    out["volume_zscore_20"] = (volume - vol_mean) / vol_std

    out["obv"] = ta.obv(close, volume)
    out["obv_slope"] = out["obv"].diff(5) / 5.0

    vwap = _rolling_vwap(df)
    vwap_safe = vwap.replace(0, np.nan)
    out["vwap_dev"] = (close - vwap_safe) / vwap_safe

    # ------------------------------------------------------------------ #
    # TREND
    # ------------------------------------------------------------------ #
    ema_9 = ta.ema(close, length=9)
    ema_21 = ta.ema(close, length=21)
    ema_50 = ta.ema(close, length=50)
    out["ema_9"] = ema_9 if ema_9 is not None else np.nan
    out["ema_21"] = ema_21 if ema_21 is not None else np.nan
    out["ema_50"] = ema_50 if ema_50 is not None else np.nan
    ema_diff = out["ema_9"] - out["ema_21"]
    out["ema_cross_9_21"] = ema_diff.apply(lambda x: np.sign(x) if pd.notna(x) else np.nan)
    out["price_vs_ema50"] = (close - out["ema_50"]) / out["ema_50"].replace(0, np.nan)

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is None:
        out["adx_14"] = np.nan
        out["di_plus"] = np.nan
        out["di_minus"] = np.nan
    else:
        out["adx_14"] = adx_df.iloc[:, 0]
        out["di_plus"] = adx_df.iloc[:, 1]
        out["di_minus"] = adx_df.iloc[:, 2]

    # ------------------------------------------------------------------ #
    # MEAN REVERSION
    # ------------------------------------------------------------------ #
    # bb_mid IS the 20-period SMA of close; (bb_upper - bb_mid) / 2 = rolling std
    roll_mean_20 = out["bb_mid"]
    roll_std_20 = ((out["bb_upper"] - out["bb_mid"]) / 2.0).replace(0, np.nan)
    out["zscore_20"] = (close - roll_mean_20) / roll_std_20

    high_52w = high.rolling(252).max()
    low_52w = low.rolling(252).min()
    out["distance_from_52w_high"] = (close - high_52w) / high_52w.replace(0, np.nan)
    out["distance_from_52w_low"] = (close - low_52w) / low_52w.replace(0, np.nan)

    # ------------------------------------------------------------------ #
    # REGIME  (simple vol regime — no HMM dependency at feature time)
    # ------------------------------------------------------------------ #
    median_vol = out["realized_vol_20"].rolling(60).median()
    out["vol_regime"] = (out["realized_vol_20"] > median_vol).astype(int)

    # ------------------------------------------------------------------ #
    # AUXILIARY FEATURES  (Phase 1 — kept separate from FEATURE_COLUMNS
    # until models are retrained on the augmented schema)
    # ------------------------------------------------------------------ #
    out["fii_net_cash_norm"] = (fii_net_cash / 1e5) if fii_net_cash is not None else np.nan
    out["india_vix"] = india_vix if india_vix is not None else np.nan
    out["sentiment_score"] = sentiment_score if sentiment_score is not None else np.nan
    out["regime_code"] = regime_code if regime_code is not None else np.nan

    # ------------------------------------------------------------------ #
    # LABELS  (training only — never used at inference)
    # ------------------------------------------------------------------ #
    if include_labels:
        out["forward_return_5d"] = close.shift(-5) / close - 1
        out["label"] = (out["forward_return_5d"] > 0.02).astype(float)
        # Mask last 5 rows — no forward data available
        out.loc[out.index[-5:], ["forward_return_5d", "label"]] = np.nan

    # Pass raw close through so callers can use it for execution pricing
    # without requiring a separate OHLCV fetch. Not a feature — never in
    # FEATURE_COLUMNS.
    out["close"] = close

    # ------------------------------------------------------------------ #
    # Drop NaN warm-up rows (only on core FEATURE_COLUMNS — auxiliary
    # columns may legitimately be NaN when external data is unavailable)
    # ------------------------------------------------------------------ #
    feature_cols = [
        c for c in out.columns if c not in LABEL_COLUMNS + AUXILIARY_FEATURE_COLUMNS + ["close"]
    ]
    out = out.dropna(subset=feature_cols)

    # Cache to Redis if enabled
    if use_cache and symbol:
        _set_cached_features_in_redis(cache_key, out)

    log.debug("features_built", rows=len(out), cols=len(out.columns))
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_input(df: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")


def _rolling_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Rolling VWAP over *window* bars.

    VWAP = Σ(typical_price × volume) / Σ(volume)
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)
    tp_vol = typical * vol
    vwap = tp_vol.rolling(window).sum() / vol.rolling(window).sum()
    return vwap
