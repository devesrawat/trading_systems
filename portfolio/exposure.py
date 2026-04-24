"""
Sector and asset class exposure calculations.
"""

from __future__ import annotations

import structlog

from portfolio.schema import PortfolioState, SectorExposure

log = structlog.get_logger(__name__)

# Sector mapping — maps NSE symbols to sectors
# This is a simplified mapping; in production, would come from a config file or database
SECTOR_MAP = {
    # Banking & Finance
    "SBIN": "Banking",
    "HDFC": "Banking",
    "HDFC Bank": "Banking",
    "ICICIBANK": "Banking",
    "INFY": "IT",
    "TCS": "IT",
    "WIPRO": "IT",
    "HCL": "IT",
    "LTIM": "IT",
    "TECHM": "IT",
    "RELIANCE": "Energy",
    "POWERGRID": "Energy",
    "COALINDIA": "Energy",
    "ITC": "FMCG",
    "NESTLEIND": "FMCG",
    "HINDUNILVR": "FMCG",
    "MARICO": "FMCG",
    "MARUTI": "Automobiles",
    "HYUNDAI": "Automobiles",
    "TATAMOTORS": "Automobiles",
    "EICHERMOT": "Automobiles",
    "BHARTIARTL": "Telecom",
    "JIOFINANCE": "Telecom",
    "TATA": "Conglomerates",
    "BAJAJFINSV": "Financials",
    "BAJAJHLDNG": "Financials",
    "L&TFH": "Financials",
    "ICICIPRULI": "Financials",
    "HDFCLIFE": "Financials",
    "SBILIFE": "Financials",
    "ADANIPORTS": "Ports & Logistics",
    "ADANIPOWER": "Energy",
    "ADANIGREEN": "Energy",
}


def get_sector_for_symbol(symbol: str) -> str:
    """
    Get sector for a symbol.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. INFY, RELIANCE)

    Returns
    -------
    str
        Sector name, or "Other" if not found.
    """
    # Remove .NS or .BO suffix if present
    clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
    return SECTOR_MAP.get(clean_symbol, "Other")


def compute_sector_exposure(portfolio: PortfolioState) -> dict[str, SectorExposure]:
    """
    Compute sector exposure for the current portfolio.

    Parameters
    ----------
    portfolio : PortfolioState
        Current portfolio state.

    Returns
    -------
    dict[str, SectorExposure]
        Sector → SectorExposure mapping.
    """
    sector_totals: dict[str, float] = {}
    sector_symbols: dict[str, list[str]] = {}

    for symbol, position in portfolio.positions.items():
        sector = get_sector_for_symbol(symbol)
        exposure = position.market_value
        if sector not in sector_totals:
            sector_totals[sector] = 0.0
            sector_symbols[sector] = []
        sector_totals[sector] += exposure
        sector_symbols[sector].append(symbol)

    # Convert to percentages and rank
    exposures: dict[str, SectorExposure] = {}
    sorted_sectors = sorted(sector_totals.items(), key=lambda x: x[1], reverse=True)

    for rank, (sector, total_value) in enumerate(sorted_sectors, 1):
        pct = total_value / portfolio.total_capital if portfolio.total_capital > 0 else 0.0
        exposures[sector] = SectorExposure(
            sector=sector,
            pct_of_capital=pct,
            rank=rank,
            symbols=sector_symbols[sector],
        )

    return exposures


def is_over_sector_limit(
    sector: str,
    new_qty: float,
    new_price: float,
    current_exposure: dict[str, SectorExposure],
    total_capital: float,
    max_sector_pct: float,
) -> bool:
    """
    Check if adding a position would exceed sector limit.

    Parameters
    ----------
    sector : str
        Sector name.
    new_qty : float
        Quantity of new position.
    new_price : float
        Price of new position.
    current_exposure : dict[str, SectorExposure]
        Current sector exposures.
    total_capital : float
        Total portfolio capital.
    max_sector_pct : float
        Maximum sector exposure as fraction.

    Returns
    -------
    bool
        True if adding position would exceed limit.
    """
    new_position_value = new_qty * new_price
    current_sector_exposure = current_exposure.get(
        sector, SectorExposure(sector=sector, pct_of_capital=0.0, rank=999)
    )

    total_after = (current_sector_exposure.pct_of_capital * total_capital) + new_position_value
    pct_after = total_after / total_capital if total_capital > 0 else 0.0

    return pct_after > max_sector_pct


def exposure_adjusted_capital_for_signal(
    symbol: str,
    qty: float,
    price: float,
    current_exposure: dict[str, SectorExposure],
    total_capital: float,
    max_sector_pct: float,
) -> float:
    """
    Calculate capital allowed for a signal after sector constraint.

    If adding the full qty would exceed sector limit, reduce qty to fit limit.

    Parameters
    ----------
    symbol : str
        Symbol being evaluated.
    qty : float
        Requested quantity.
    price : float
        Current price.
    current_exposure : dict[str, SectorExposure]
        Current sector exposures.
    total_capital : float
        Total capital.
    max_sector_pct : float
        Max sector % limit.

    Returns
    -------
    float
        Capital allowed (rupees). May be < requested amount.
    """
    sector = get_sector_for_symbol(symbol)
    current_sector_pct = current_exposure.get(
        sector, SectorExposure(sector=sector, pct_of_capital=0.0, rank=999)
    ).pct_of_capital

    # Room left in sector
    room_pct = max_sector_pct - current_sector_pct
    if room_pct <= 0:
        log.warning("sector_full", sector=sector, current=current_sector_pct, max=max_sector_pct)
        return 0.0

    # Capital allowed
    room_capital = room_pct * total_capital
    requested_capital = qty * price

    allowed_capital = min(room_capital, requested_capital)
    return allowed_capital
