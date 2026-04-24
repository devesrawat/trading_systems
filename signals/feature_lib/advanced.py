"""Advanced indicators: correlation, sector relative strength, regime detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_correlation_to_benchmark(
    symbol_returns: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
    period: int = 50,
) -> pd.Series:
    if symbol_returns is None or benchmark_returns is None:
        if symbol_returns is not None:
            return pd.Series(np.nan, index=symbol_returns.index)
        elif benchmark_returns is not None:
            return pd.Series(np.nan, index=benchmark_returns.index)
        else:
            return pd.Series(np.nan)
    aligned = pd.DataFrame({"symbol": symbol_returns, "benchmark": benchmark_returns}).dropna()
    correlation = aligned["symbol"].rolling(period).corr(aligned["benchmark"])
    correlation.index = symbol_returns.index
    return correlation


def compute_sector_relative_strength(
    symbol_price: pd.Series | None = None, sector_price: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if symbol_price is None or sector_price is None:
        if symbol_price is not None:
            return pd.Series(np.nan, index=symbol_price.index)
        elif sector_price is not None:
            return pd.Series(np.nan, index=sector_price.index)
        else:
            return pd.Series(np.nan)
    symbol_perf = symbol_price.pct_change(period)
    sector_perf = sector_price.pct_change(period)
    sector_perf_safe = sector_perf.replace(0, np.nan)
    relative_strength = symbol_perf / sector_perf_safe
    return relative_strength


def compute_regime_indicator(
    close: pd.Series, vol: pd.Series, vol_median: pd.Series | None = None, vol_period: int = 60
) -> tuple[pd.Series, pd.Series]:
    sma_50 = close.rolling(50).mean()
    trend_regime = (close > sma_50).astype(int) * 2 - 1
    if vol_median is None:
        vol_median = vol.rolling(vol_period).median()
    vol_regime = (vol > vol_median).astype(int)
    return trend_regime, vol_regime


def compute_price_acceleration(close: pd.Series, period: int = 5) -> pd.Series:
    velocity = close.diff(period)
    acceleration = velocity.diff(period)
    return acceleration


def compute_atr_trailing_stop(
    atr: pd.Series, close: pd.Series, multiple: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    distance = atr * multiple
    long_stop = close - distance
    short_stop = close + distance
    return long_stop, short_stop


def compute_support_resistance_levels(
    high: pd.Series, low: pd.Series, period: int = 20
) -> tuple[pd.Series, pd.Series]:
    support = low.rolling(period).min()
    resistance = high.rolling(period).max()
    return support, resistance
