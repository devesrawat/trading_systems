"""
Fundamentals package for multibagger discovery.

Composite scoring of growth companies based on:
- Growth: Revenue/net income CAGR
- Quality: ROE, ROCE, profit margins
- Balance sheet: Debt ratios, interest coverage, liquidity
- Valuation: PE relative to growth (PEG), enterprise value
- Momentum: Institutional ownership, price momentum, sector relative strength

All scores deterministic, calibrated for Indian small-cap multibaggers.
"""

from __future__ import annotations

__all__ = [
    "BaseFundamentalsProvider",
    "FundamentalsScores",
    "MultibaggerWatchlist",
    "NSEProvider",
    "QuarterlyFinancials",
    "ScreenerProvider",
    "Shareholding",
    "TrendlyneProvider",
    "Valuations",
    "compute_balance_sheet_score",
    "compute_composite_rank",
    "compute_growth_score",
    "compute_momentum_score",
    "compute_quality_score",
    "compute_valuation_score",
    "fetch_fundamentals",
    "get_cached",
    "store_in_db",
]

from fundamentals.ingest import fetch_fundamentals, get_cached, store_in_db
from fundamentals.providers import (
    BaseFundamentalsProvider,
    NSEProvider,
    ScreenerProvider,
    TrendlyneProvider,
)
from fundamentals.ranking import compute_composite_rank
from fundamentals.schema import (
    FundamentalsScores,
    QuarterlyFinancials,
    Shareholding,
    Valuations,
)
from fundamentals.scoring import (
    compute_balance_sheet_score,
    compute_growth_score,
    compute_momentum_score,
    compute_quality_score,
    compute_valuation_score,
)
from fundamentals.watchlist import MultibaggerWatchlist
