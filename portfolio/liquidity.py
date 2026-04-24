"""
Liquidity constraints and stress testing.
"""

from __future__ import annotations

import structlog

from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_liquidity_score(
    symbol: str,
    avg_volume: float,
    bid_ask_spread_pct: float = 0.01,
) -> float:
    """
    Compute a liquidity score (0-100) for a symbol.

    Score combines volume and spread:
    - 0-20: Very illiquid (low volume, wide spread)
    - 20-40: Illiquid
    - 40-60: Moderate
    - 60-80: Liquid
    - 80-100: Very liquid

    Stub implementation: returns realistic value based on volume.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    avg_volume : float
        Average daily volume (in units/shares).
    bid_ask_spread_pct : float
        Bid-ask spread as % of price (0-1).

    Returns
    -------
    float
        Liquidity score [0, 100].
    """
    # Normalize volume component (arbitrary: assume Nifty-50 stocks have 1M+ daily volume)
    volume_score = min(100.0, (avg_volume / 1_000_000) * 100)

    # Spread component (wider spread = lower score)
    spread_penalty = bid_ask_spread_pct * 100  # 0.01 (1%) → -1
    spread_score = max(0.0, 100.0 - spread_penalty)

    # Weighted average
    liquidity_score = 0.7 * volume_score + 0.3 * spread_score
    return min(100.0, max(0.0, liquidity_score))


def check_minimum_liquidity(
    symbol: str,
    qty: float,
    current_price: float,
    bid_ask_impact_bps: float = 10.0,
) -> tuple[bool, float]:
    """
    Estimate slippage for a position and check if acceptable.

    Stub: returns True with minimal slippage estimate.

    Parameters
    ----------
    symbol : str
        Trading symbol.
    qty : float
        Position quantity.
    current_price : float
        Current market price.
    bid_ask_impact_bps : float
        Bid-ask impact in basis points for this size.

    Returns
    -------
    tuple[bool, float]
        (allowed, slippage_estimate_bps). allowed=True if slippage acceptable.
    """
    log.debug(
        "liquidity_check",
        symbol=symbol,
        qty=qty,
        price=current_price,
        bid_ask_impact_bps=bid_ask_impact_bps,
    )
    # Stub: assume order is acceptable for small/medium sizes
    # Threshold: if estimated slippage > 50 bps, reject
    if bid_ask_impact_bps > 50.0:
        return False, bid_ask_impact_bps
    return True, bid_ask_impact_bps


def portfolio_liquidity_stress_test(
    positions: dict[str, PortfolioPosition],
    market_condition: str = "normal",
) -> tuple[float, float]:
    """
    Stress test portfolio liquidity in adverse conditions.

    Estimate: what % of portfolio can be liquidated in 1 day, 5 days?

    Stub: returns (0.8, 0.95) meaning 80% liquidatable same-day, 95% in 5 days.

    Parameters
    ----------
    positions : dict[str, object]
        Current positions (symbol → PortfolioPosition).
    market_condition : str
        Market condition: "normal", "stressed", "crisis".

    Returns
    -------
    tuple[float, float]
        (pct_liquidatable_same_day, pct_liquidatable_5_days)
    """
    if not positions:
        return 1.0, 1.0

    # Stress adjustments
    if market_condition == "normal":
        same_day_factor = 0.80
        five_day_factor = 0.95
    elif market_condition == "stressed":
        same_day_factor = 0.50
        five_day_factor = 0.75
    else:  # crisis
        same_day_factor = 0.20
        five_day_factor = 0.40

    log.debug(
        "liquidity_stress_test",
        condition=market_condition,
        same_day_factor=same_day_factor,
        five_day_factor=five_day_factor,
    )

    return same_day_factor, five_day_factor
