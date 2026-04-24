"""
Portfolio risk limits configuration and helpers.
"""

from __future__ import annotations

from portfolio.schema import RiskLimits

# Defaults suitable for institutional trading
DEFAULT_EQUITY_LIMITS = RiskLimits(
    max_sector_pct=0.25,
    max_single_stock_pct=0.02,
    max_correlation_to_portfolio=0.70,
    min_liquidity_score=70,
    max_turnover_pct_annual=2.0,
    max_positions=20,
    max_intraday_trades_per_symbol=3,
)

# More conservative for crypto
CRYPTO_LIMITS = RiskLimits(
    max_sector_pct=0.15,
    max_single_stock_pct=0.01,
    max_correlation_to_portfolio=0.60,
    min_liquidity_score=75,
    max_turnover_pct_annual=1.5,
    max_positions=15,
    max_intraday_trades_per_symbol=2,
)

# Tight paper-trading limits for validation
PAPER_TRADE_LIMITS = RiskLimits(
    max_sector_pct=0.15,
    max_single_stock_pct=0.015,
    max_correlation_to_portfolio=0.65,
    min_liquidity_score=75,
    max_turnover_pct_annual=1.5,
    max_positions=15,
    max_intraday_trades_per_symbol=2,
)

# Aggressive live-trading limits (only after 200+ paper trades)
LIVE_TRADING_LIMITS = RiskLimits(
    max_sector_pct=0.25,
    max_single_stock_pct=0.02,
    max_correlation_to_portfolio=0.70,
    min_liquidity_score=60,
    max_turnover_pct_annual=2.5,
    max_positions=25,
    max_intraday_trades_per_symbol=5,
)


def get_limits_for_mode(mode: str) -> RiskLimits:
    """
    Get appropriate limits based on trading mode.

    Parameters
    ----------
    mode : str
        Trading mode: "research", "watchlist", "paper", "live"

    Returns
    -------
    RiskLimits
        Limits configuration for the mode.
    """
    if mode == "research" or mode == "watchlist" or mode == "paper":
        return PAPER_TRADE_LIMITS
    elif mode == "live":
        return LIVE_TRADING_LIMITS
    else:
        return DEFAULT_EQUITY_LIMITS
