"""
Portfolio turnover tracking and constraints.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog
from sqlalchemy import text

from data.store import get_engine
from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_portfolio_turnover(
    total_capital: float,
    lookback_days: int = 30,
) -> float:
    """
    Compute actual annualized portfolio turnover using historical trade data from TimescaleDB.

    Turnover (annualized) = (Sum of Absolute Trade Notionals / average_capital) * (365 / lookback_days) / 2
    Divided by 2 because a full cycle (buy+sell) counts as one turnover.

    Parameters
    ----------
    total_capital : float
        Current portfolio capital.
    lookback_days : int
        Lookback window in days.

    Returns
    -------
    float
        Annualized turnover as fraction (e.g., 2.0 = 200% turnover/year).
    """
    if total_capital <= 0:
        return 0.0

    try:
        engine = get_engine()
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        # Sum of absolute notional of all trades in lookback
        query = text("""
            SELECT SUM(ABS(quantity * entry_price))
            FROM trades
            WHERE entry_time >= :start_date
              AND entry_time <= :end_date
        """)

        with engine.connect() as conn:
            result = conn.execute(
                query, {"start_date": start_date, "end_date": end_date}
            ).fetchone()

        if not result or result[0] is None:
            return 0.0

        total_traded_notional = float(result[0])

        # Annualize
        period_turnover = total_traded_notional / total_capital
        annualized_turnover = (period_turnover * (365 / lookback_days)) / 2.0

        log.debug(
            "portfolio_turnover_computed",
            traded_notional=int(total_traded_notional),
            annualized=round(annualized_turnover, 2),
        )
        return annualized_turnover

    except Exception as exc:
        log.error("turnover_compute_failed", error=str(exc))
        return 0.0


def estimate_new_trade_turnover_impact(
    qty: int,
    price: float,
    total_capital: float,
) -> float:
    """
    Estimate the annualized turnover contribution of a single new trade.

    Parameters
    ----------
    qty : int
        Quantity.
    price : float
        Price.
    total_capital : float
        Total capital.

    Returns
    -------
    float
        Turnover fraction contribution.
    """
    if total_capital <= 0:
        return 0.0

    notional = qty * price
    # A single trade is half of a buy/sell cycle
    return (notional / total_capital) / 2.0


def check_turnover_limit(
    total_capital: float,
    max_turnover_annual: float = 6.0,
    lookback_days: int = 30,
) -> tuple[bool, str]:
    """
    Check if the portfolio's current annualized turnover is within limits.

    Parameters
    ----------
    total_capital : float
        Current capital.
    max_turnover_annual : float
        Max allowed annual turnover (e.g. 6.0 = 600%/year).
    lookback_days : int
        Lookback window to compute current turnover.

    Returns
    -------
    tuple[bool, str]
        (allowed, message)
    """
    current_turnover = compute_portfolio_turnover(total_capital, lookback_days)

    if current_turnover > max_turnover_annual:
        msg = f"turnover {current_turnover:.1f} exceeds limit {max_turnover_annual:.1f}"
        return False, msg

    return True, f"turnover OK: {current_turnover:.1f}"


def compute_turnover(
    qty: float,
    price: float,
    period_days: int = 252,
) -> float:
    """
    Backward compatibility wrapper for estimate_new_trade_turnover_impact.
    """
    return (qty * price) / 500_000.0 / 2.0


def estimate_portfolio_turnover(
    current_positions: dict[str, PortfolioPosition],
    new_position_notional: float,
    total_capital: float,
) -> float:
    """
    Backward compatibility wrapper for estimate_new_trade_turnover_impact.
    """
    return (new_position_notional / total_capital) / 2.0 if total_capital > 0 else 0.0
