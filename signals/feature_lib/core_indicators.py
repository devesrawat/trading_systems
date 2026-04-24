"""
Core momentum and trend indicators.

Functions:
- compute_rsi: Relative Strength Index
- compute_macd: Moving Average Convergence Divergence
- compute_momentum: Momentum (ROC variant)
- compute_roc: Rate of Change
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        Lookback period (default 14).

    Returns
    -------
    pd.Series
        RSI values bounded [0, 100]. NaN where computation impossible.
    """
    result = ta.rsi(close, length=period)
    return result if result is not None else pd.Series(np.nan, index=close.index)


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Moving Average Convergence Divergence (MACD).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal line EMA period (default 9).

    Returns
    -------
    tuple of (macd, signal, histogram)
        Each is a pd.Series aligned to input index.
        Values are NaN where computation is impossible.
    """
    df_macd = ta.macd(close, fast=fast, slow=slow, signal=signal)

    if df_macd is None or df_macd.empty:
        return (
            pd.Series(np.nan, index=close.index),
            pd.Series(np.nan, index=close.index),
            pd.Series(np.nan, index=close.index),
        )

    macd_series = df_macd.iloc[:, 0]
    hist_series = df_macd.iloc[:, 1]
    signal_series = df_macd.iloc[:, 2]

    # Align indices
    macd_series.index = close.index
    hist_series.index = close.index
    signal_series.index = close.index

    return macd_series, hist_series, signal_series


def compute_momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Momentum indicator (close - close[t-period]).

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        Lookback period (default 10).

    Returns
    -------
    pd.Series
        Momentum values (current - historical).
    """
    result = ta.mom(close, length=period)
    return result if result is not None else pd.Series(np.nan, index=close.index)


def compute_roc(close: pd.Series, period: int = 5) -> pd.Series:
    """
    Rate of Change — percentage change over *period* bars.

    Parameters
    ----------
    close : pd.Series
        Close prices.
    period : int
        Lookback period (default 5).

    Returns
    -------
    pd.Series
        ROC as decimal (e.g., 0.05 for +5%).
    """
    result = ta.roc(close, length=period)
    return result if result is not None else pd.Series(np.nan, index=close.index)


def compute_macd_cross(macd: pd.Series, signal: pd.Series) -> pd.Series:
    """
    MACD crossover signal: +1 if MACD > signal, -1 if MACD < signal, 0 if equal.

    Parameters
    ----------
    macd : pd.Series
        MACD line.
    signal : pd.Series
        Signal line (EMA of MACD).

    Returns
    -------
    pd.Series
        Values in {-1, 0, 1}.
    """
    return np.sign(macd - signal)
