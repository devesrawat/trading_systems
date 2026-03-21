"""
Performance metric calculators and tearsheet printer.

All functions operate on pd.Series of per-trade or per-bar returns.
Risk-free rate default: 6.5% (India 10Y G-Sec approximation).
"""
from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

_TRADING_DAYS = 252
_DEFAULT_RISK_FREE = 0.065

# Minimum thresholds from architecture spec (Section 17)
_THRESHOLDS = {
    "sharpe_min": 1.0,
    "sharpe_good": 1.5,
    "max_dd_min": -0.25,
    "max_dd_good": -0.15,
    "profit_factor_min": 1.5,
    "profit_factor_good": 2.0,
    "win_rate_min": 0.52,
    "win_rate_good": 0.58,
    "calmar_min": 0.5,
    "calmar_good": 1.0,
}


def sharpe_ratio(returns: pd.Series, risk_free: float = _DEFAULT_RISK_FREE) -> float:
    """Annualised Sharpe ratio using 252 trading days."""
    if returns.empty:
        return 0.0
    excess = returns - (risk_free / _TRADING_DAYS)
    std = returns.std()
    if std == 0:
        return float(excess.mean() * _TRADING_DAYS)
    return float((excess.mean() / std) * sqrt(_TRADING_DAYS))


def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown as a negative fraction.

    Returns 0.0 for a monotonically increasing equity curve.
    """
    if equity_curve.empty:
        return 0.0
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    """
    Gross profit / gross loss (absolute value).

    Returns inf if no losses, 0.0 if no wins.
    """
    wins = returns[returns > 0].sum()
    losses = returns[returns < 0].abs().sum()
    if losses == 0:
        return float("inf") if wins > 0 else 1.0
    if wins == 0:
        return 0.0
    return float(wins / losses)


def win_rate(returns: pd.Series) -> float:
    """Fraction of positive return periods."""
    if returns.empty:
        return 0.0
    return float((returns > 0).mean())


def expectancy(returns: pd.Series) -> float:
    """
    Average win × win_rate − average loss × loss_rate.

    Must be positive for a viable strategy.
    """
    if returns.empty:
        return 0.0
    wr = win_rate(returns)
    lr = 1.0 - wr
    avg_win = returns[returns > 0].mean() if wr > 0 else 0.0
    avg_loss = returns[returns < 0].abs().mean() if lr > 0 else 0.0
    return float(avg_win * wr - avg_loss * lr)


def calmar_ratio(returns: pd.Series, max_dd: float) -> float:
    """
    Annualised return / abs(max drawdown).

    Returns inf when max_dd is zero (no drawdown).
    """
    if max_dd == 0.0:
        return float("inf")
    ann_return = float(returns.mean() * _TRADING_DAYS)
    return ann_return / abs(max_dd)


def print_tearsheet(returns: pd.Series, equity_curve: pd.Series) -> None:
    """
    Print a formatted performance tearsheet with PASS/FAIL flags
    against the minimum thresholds defined in Section 17.
    """
    s = sharpe_ratio(returns)
    dd = max_drawdown(equity_curve)
    pf = profit_factor(returns)
    wr = win_rate(returns)
    cal = calmar_ratio(returns, dd)
    exp = expectancy(returns)
    ann_ret = returns.mean() * _TRADING_DAYS

    def _flag(value: float, good: float, minimum: float, higher_is_better: bool = True) -> str:
        if higher_is_better:
            if value >= good:
                return "PASS ✓"
            if value >= minimum:
                return "PASS (marginal)"
            return "FAIL ✗"
        else:
            if value <= good:
                return "PASS ✓"
            if value <= minimum:
                return "PASS (marginal)"
            return "FAIL ✗"

    rows = [
        ("Sharpe Ratio",     f"{s:.3f}",          _flag(s, _THRESHOLDS["sharpe_good"], _THRESHOLDS["sharpe_min"])),
        ("Max Drawdown",     f"{dd:.2%}",          _flag(dd, _THRESHOLDS["max_dd_good"], _THRESHOLDS["max_dd_min"], higher_is_better=False)),
        ("Profit Factor",    f"{pf:.3f}",          _flag(pf, _THRESHOLDS["profit_factor_good"], _THRESHOLDS["profit_factor_min"])),
        ("Win Rate",         f"{wr:.2%}",          _flag(wr, _THRESHOLDS["win_rate_good"], _THRESHOLDS["win_rate_min"])),
        ("Calmar Ratio",     f"{cal:.3f}",         _flag(cal, _THRESHOLDS["calmar_good"], _THRESHOLDS["calmar_min"])),
        ("Expectancy",       f"{exp:.4f}",         "PASS ✓" if exp > 0 else "FAIL ✗"),
        ("Ann. Return",      f"{ann_ret:.2%}",     ""),
        ("Total Trades",     f"{len(returns)}",    ""),
    ]

    col_w = [22, 12, 20]
    header = f"{'Metric':<{col_w[0]}}  {'Value':>{col_w[1]}}  {'Status':<{col_w[2]}}"
    sep = "-" * (sum(col_w) + 4)
    print("\n=== STRATEGY TEARSHEET ===")
    print(header)
    print(sep)
    for name, value, status in rows:
        print(f"{name:<{col_w[0]}}  {value:>{col_w[1]}}  {status:<{col_w[2]}}")
    print(sep)
    print()
