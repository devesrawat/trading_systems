"""
Correlation analysis for portfolio positions.
"""

from __future__ import annotations

import structlog

from portfolio.schema import PortfolioPosition

log = structlog.get_logger(__name__)


def compute_pairwise_correlation(symbol1: str, symbol2: str, lookback_days: int = 252) -> float:
    """
    Compute pairwise correlation between two symbols.

    NOTE: This is a skeleton implementation. In production, would fetch
    historical price data from TimescaleDB and compute Pearson correlation.

    Parameters
    ----------
    symbol1 : str
        First symbol.
    symbol2 : str
        Second symbol.
    lookback_days : int
        Lookback window in days.

    Returns
    -------
    float
        Correlation coefficient [-1, 1]. Stub returns 0.3.
    """
    # TODO: Implement with real data from TimescaleDB
    # For now, return a realistic stub value
    log.debug(
        "pairwise_correlation_stub",
        symbol1=symbol1,
        symbol2=symbol2,
        lookback_days=lookback_days,
    )
    return 0.3


def compute_portfolio_correlation(
    symbol: str,
    current_positions: dict[str, PortfolioPosition],
    lookback_days: int = 252,
) -> tuple[float, tuple[str, str] | None]:
    """
    Compute average correlation of a symbol to current portfolio.

    Parameters
    ----------
    symbol : str
        Symbol being evaluated.
    current_positions : dict[str, object]
        Current open positions (key: symbol).
    lookback_days : int
        Lookback window in days.

    Returns
    -------
    tuple[float, tuple[str, str] | None]
        (average_correlation, (symbol1, symbol2) with highest correlation)
        Returns (0.0, None) if no open positions.
    """
    if not current_positions:
        return 0.0, None

    correlations = []
    max_corr = -2.0
    max_pair: tuple[str, str] | None = None

    for existing_symbol in current_positions:
        corr = compute_pairwise_correlation(symbol, existing_symbol, lookback_days)
        correlations.append(corr)
        if corr > max_corr:
            max_corr = corr
            max_pair = (symbol, existing_symbol)

    avg_corr = sum(correlations) / len(correlations) if correlations else 0.0
    return avg_corr, max_pair if max_corr > -2.0 else None


def apply_correlation_penalty(confidence: float, avg_correlation: float) -> float:
    """
    Apply a penalty to signal confidence based on portfolio correlation.

    Logic:
    - If avg_correlation < 0.3: no penalty
    - If avg_correlation 0.3-0.6: linear penalty (0-0.5x confidence)
    - If avg_correlation > 0.6: strong penalty (0.5x confidence)

    Parameters
    ----------
    confidence : float
        Original signal confidence [0, 1].
    avg_correlation : float
        Average correlation to portfolio [-1, 1].

    Returns
    -------
    float
        Adjusted confidence [0, 1].
    """
    if avg_correlation < 0.3:
        penalty = 0.0
    elif avg_correlation < 0.6:
        # Linear from 0 to 0.5 as correlation goes from 0.3 to 0.6
        penalty = ((avg_correlation - 0.3) / 0.3) * 0.5
    else:
        # Strong penalty
        penalty = 0.5

    adjusted = confidence * (1.0 - penalty)
    return adjusted


def is_redundant_position(
    symbol: str,
    current_positions: dict[str, PortfolioPosition],
    threshold: float = 0.6,
) -> bool:
    """
    Check if a symbol would be redundant (highly correlated to existing positions).

    Parameters
    ----------
    symbol : str
        Symbol being evaluated.
    current_positions : dict[str, object]
        Current open positions.
    threshold : float
        Correlation threshold above which position is considered redundant.

    Returns
    -------
    bool
        True if symbol is redundant.
    """
    if not current_positions:
        return False

    avg_corr, _ = compute_portfolio_correlation(symbol, current_positions)
    return avg_corr > threshold
