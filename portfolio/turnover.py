"""
Portfolio turnover tracking and constraints.
"""

from __future__ import annotations

import structlog

from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_turnover(
    qty: float,
    price: float,
    period_days: int = 252,
) -> float:
    """
    Compute annualized turnover contribution for a trade.

    Turnover = notional traded / average capital deployed per day

    Stub: returns turnover as % of capital per year.

    Parameters
    ----------
    qty : float
        Quantity traded.
    price : float
        Trade price.
    period_days : int
        Period in days (default 252 = 1 year).

    Returns
    -------
    float
        Annualized turnover as fraction (0-10+).
    """
    notional = qty * price
    # Stub: assume this trade is ~0.5% of annual turnover
    annual_turnover_contribution = 0.005
    log.debug(
        "turnover_computed",
        qty=qty,
        price=price,
        notional=notional,
        contribution=annual_turnover_contribution,
    )
    return annual_turnover_contribution


def estimate_portfolio_turnover(
    current_positions: dict[str, PortfolioPosition],
    new_position_notional: float,
    total_capital: float,
) -> float:
    """
    Estimate portfolio turnover if new position is added.

    Parameters
    ----------
    current_positions : dict[str, object]
        Current open positions (symbol → PortfolioPosition).
    new_position_notional : float
        Notional value of new position (qty * price).
    total_capital : float
        Total portfolio capital.

    Returns
    -------
    float
        Estimated annual turnover as %.
    """
    # Stub: calculate based on notional size relative to capital
    if total_capital <= 0:
        return 0.0

    # Simple model: assume trades average 2% of capital each
    # Typical year might see 10-50 trades, so 2-100% turnover baseline
    estimated_turnover_pct = (new_position_notional / total_capital) * 0.10
    log.debug(
        "portfolio_turnover_estimated",
        new_notional=new_position_notional,
        total_capital=total_capital,
        estimated_pct=estimated_turnover_pct,
    )
    return estimated_turnover_pct


def check_turnover_limit(
    estimated_turnover_pct: float,
    max_turnover_pct_annual: float,
) -> tuple[bool, str]:
    """
    Check if estimated turnover would exceed limit.

    Parameters
    ----------
    estimated_turnover_pct : float
        Estimated annual turnover as %.
    max_turnover_pct_annual : float
        Max allowed annual turnover as %.

    Returns
    -------
    tuple[bool, str]
        (allowed, reason)
    """
    if estimated_turnover_pct > max_turnover_pct_annual:
        return (
            False,
            f"turnover {estimated_turnover_pct:.1%} exceeds limit {max_turnover_pct_annual:.1%}",
        )
    return True, f"turnover OK: {estimated_turnover_pct:.1%} < {max_turnover_pct_annual:.1%}"
