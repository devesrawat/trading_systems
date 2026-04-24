"""Volatility indicators."""

from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    result = ta.atr(high, low, close, length=period)
    return result if result is not None else pd.Series(np.nan, index=close.index)


def compute_atr_pct(atr: pd.Series, close: pd.Series) -> pd.Series:
    close_safe = close.replace(0, np.nan)
    return atr / close_safe


def compute_bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    bb = ta.bbands(close, length=period, std=std_dev)
    if bb is None or bb.empty:
        return (
            pd.Series(np.nan, index=close.index),
            pd.Series(np.nan, index=close.index),
            pd.Series(np.nan, index=close.index),
            pd.Series(np.nan, index=close.index),
        )
    lower = bb.iloc[:, 0]
    mid = bb.iloc[:, 1]
    upper = bb.iloc[:, 2]
    lower.index = close.index
    mid.index = close.index
    upper.index = close.index
    bb_range = (upper - lower).replace(0, np.nan)
    position = (close - lower) / bb_range
    return lower, mid, upper, position


def compute_realized_volatility(close: pd.Series, period: int = 20) -> pd.Series:
    pct_chg = close.pct_change()
    vol = pct_chg.rolling(period).std() * sqrt(252)
    return vol


def compute_parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
    low_safe = low.replace(0, np.nan)
    hl_ratio = high / low_safe
    hl_ratio = hl_ratio.replace(0, np.nan)
    ln_hl = np.log(hl_ratio)
    ln_hl_sq = ln_hl**2
    sum_ln_sq = ln_hl_sq.rolling(period).sum()
    variance = sum_ln_sq / (period * 4 * np.log(2))
    volatility = np.sqrt(variance) * sqrt(252)
    return volatility


def compute_garman_klass_volatility(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    open_: pd.Series | None = None,
    period: int = 20,
) -> pd.Series:
    if open_ is None:
        open_ = close.shift(1)
    low_safe = low.replace(0, np.nan)
    open_safe = open_.replace(0, np.nan)
    hl_ratio = high / low_safe
    co_ratio = close / open_safe
    hl_ratio = hl_ratio.replace(0, np.nan)
    co_ratio = co_ratio.replace(0, np.nan)
    ln_hl = np.log(hl_ratio)
    ln_co = np.log(co_ratio)
    term1 = 0.5 * ln_hl**2
    term2 = (2 * np.log(2) - 1) * ln_co**2
    sum_terms = (term1 - term2).rolling(period).sum()
    variance = sum_terms / period
    volatility = np.sqrt(variance) * sqrt(252)
    return volatility


def compute_volatility_of_volatility(vol: pd.Series, period: int = 20) -> pd.Series:
    return vol.rolling(period).std()
