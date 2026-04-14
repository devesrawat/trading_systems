"""
Black-Scholes option Greeks for NSE F&O positions.

Public API
----------
delta(S, K, T, r, sigma, option_type) → float
gamma(S, K, T, r, sigma)              → float
theta(S, K, T, r, sigma, option_type) → float  (per calendar day)
compute_portfolio_delta(positions)    → float   (net delta in shares)
"""

from __future__ import annotations

import math


def _validate_inputs(T: float, sigma: float, option_type: str) -> None:
    if T <= 0:
        raise ValueError(f"T (time to expiry) must be > 0, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes delta.

    Parameters
    ----------
    S           : current underlying price
    K           : strike price
    T           : time to expiry in years (must be > 0)
    r           : risk-free rate (annualised, e.g. 0.065 for 6.5%)
    sigma       : implied volatility (annualised, e.g. 0.20 for 20%)
    option_type : "call" or "put"

    Returns
    -------
    float : call delta ∈ (0, 1), put delta ∈ (-1, 0)
    """
    _validate_inputs(T, sigma, option_type)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return _norm_cdf(d1)
    return _norm_cdf(d1) - 1.0


def gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Black-Scholes gamma (identical for calls and puts).

    Returns the rate of change of delta per unit move in S.
    """
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
) -> float:
    """
    Black-Scholes theta expressed as P&L change per calendar day.

    A long option position has negative theta (time decay hurts holder).

    Returns
    -------
    float : theta per day (negative for long positions)
    """
    _validate_inputs(T, sigma, option_type)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    common = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        return (common - r * K * math.exp(-r * T) * _norm_cdf(d2)) / 365.0
    return (common + r * K * math.exp(-r * T) * _norm_cdf(-d2)) / 365.0


def compute_portfolio_delta(positions: list[dict]) -> float:
    """
    Net portfolio delta in underlying share equivalents.

    Each position dict must have:
      - "delta"    : per-contract delta (float)
      - "qty"      : number of lots (positive = long, negative = short)
      - "lot_size" : NSE lot size for the instrument

    Returns
    -------
    float : total delta (positive = net long underlying)
    """
    if not positions:
        return 0.0
    total = 0.0
    for pos in positions:
        total += pos["delta"] * pos["qty"] * pos["lot_size"]
    return total
