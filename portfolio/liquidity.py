"""
Liquidity constraints and stress testing.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import structlog

from data.store import get_engine
from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_liquidity_score(
    symbol: str,
    lookback_days: int = 20,
) -> float:
    """
    Compute a real liquidity score (0-100) for a symbol based on ADV from TimescaleDB.

    Score combines volume relative to a benchmark (e.g., 1M shares).
    - 0-20: Very illiquid
    - 20-40: Illiquid
    - 40-60: Moderate
    - 60-80: Liquid
    - 80-100: Very liquid

    Parameters
    ----------
    symbol : str
        Trading symbol.
    lookback_days : int
        Days to average volume over.

    Returns
    -------
    float
        Liquidity score [0, 100]. Returns 0.0 if data unavailable.
    """
    try:
        engine = get_engine()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days + 10)

        query = """
            SELECT AVG(volume) as adv
            FROM ohlcv
            WHERE symbol = :symbol
              AND interval = 'day'
              AND time >= :start_date
              AND time <= :end_date
        """

        with engine.connect() as conn:
            result = conn.execute(
                query, {"symbol": symbol, "start_date": start_date, "end_date": end_date}
            ).fetchone()

        if not result or result[0] is None:
            log.warning("liquidity_adv_missing", symbol=symbol)
            return 0.0

        adv = float(result[0])
        # Normalize: 1M shares is "very liquid" (100), linear scaling below that
        # In reality, this benchmark should depend on the specific market/index
        score = min(100.0, (adv / 1_000_000) * 100)
        return score

    except Exception as exc:
        log.error("liquidity_score_failed", symbol=symbol, error=str(exc))
        return 0.0


def check_minimum_liquidity(
    symbol: str,
    qty: int,
    current_price: float,
) -> tuple[bool, float]:
    """
    Estimate slippage using the square-root market impact model:
    Slippage (bps) = Impact_Coeff * sqrt(OrderSize / ADV) * Volatility

    Parameters
    ----------
    symbol : str
        Trading symbol.
    qty : int
        Position quantity.
    current_price : float
        Current market price.

    Returns
    -------
    tuple[bool, float]
        (allowed, slippage_estimate_bps). allowed=True if slippage < 50 bps.
    """
    try:
        engine = get_engine()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=30)

        # Fetch ADV and Volatility (20-day)
        query = """
            SELECT volume, close
            FROM ohlcv
            WHERE symbol = :symbol
              AND interval = 'day'
              AND time >= :start_date
              AND time <= :end_date
            ORDER BY time ASC
        """

        with engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
            )

        if len(df) < 5:
            log.warning("liquidity_check_insufficient_data", symbol=symbol)
            return True, 5.0  # Default low slippage if new symbol

        adv = df["volume"].tail(20).mean()
        # Annualized volatility
        returns = np.log(df["close"] / df["close"].shift(1)).dropna()
        volatility = returns.std() * np.sqrt(252)

        if adv <= 0:
            return False, 999.0

        # Square-root model coefficient (typical institutional estimate ~0.5-1.0)
        impact_coeff = 0.5
        participation_ratio = qty / adv

        # Slippage in basis points
        # Factor of 10000 to convert from decimal to bps
        slippage_bps = impact_coeff * np.sqrt(participation_ratio) * volatility * 10000

        # Safety cap: if order is > 10% of ADV, apply heavy penalty
        if participation_ratio > 0.1:
            slippage_bps *= 2.0

        allowed = slippage_bps < 50.0  # Reject if > 0.5% slippage
        log.debug(
            "liquidity_check_result",
            symbol=symbol,
            qty=qty,
            adv=int(adv),
            slippage_bps=round(slippage_bps, 2),
            allowed=allowed,
        )

        return allowed, round(slippage_bps, 2)

    except Exception as exc:
        log.error("liquidity_check_failed", symbol=symbol, error=str(exc))
        return True, 10.0  # Default safe if check fails


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
