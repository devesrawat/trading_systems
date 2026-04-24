"""
Composite ranking from individual fundamentals scores.

Weighted combination of growth, quality, balance sheet, valuation, momentum, and institutional conviction.
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)


def compute_composite_rank(
    growth_score: float,
    quality_score: float,
    balance_sheet_score: float,
    valuation_score: float,
    momentum_score: float,
    institutional_conviction_score: float,
    weights: dict[str, float] | None = None,
) -> tuple[float, float | None, float | None]:
    """
    Compute composite multibagger rank from individual scores (6 dimensions).

    Formula:
    - Default weights (can be overridden):
      - growth: 25% (primary signal for multibaggers)
      - quality: 20% (earnings stability)
      - balance_sheet: 15% (financial health)
      - valuation: 15% (entry point)
      - momentum: 5% (confirmation)
      - institutional_conviction: 20% (smart money accumulation)
    - Returns: (composite_rank 0-100, growth_weighted, conviction_with_low_rally)
    - growth_weighted: 40% growth, 20% quality, 15% balance sheet, 10% valuation, 10% momentum, 5% conviction
    - conviction_with_low_rally: boost composite by 15 points if conviction >70 AND price_change <15%
      (This score is computed externally - see watchlist.py)

    Args:
        growth_score: Growth score (0-100)
        quality_score: Quality score (0-100)
        balance_sheet_score: Balance sheet score (0-100)
        valuation_score: Valuation score (0-100)
        momentum_score: Momentum score (0-100)
        institutional_conviction_score: Institutional conviction score (0-100)
        weights: Optional override of default weights

    Returns:
        Tuple of (composite_rank, growth_weighted_rank, conviction_with_low_rally)
    """

    # Default weights optimized for multibagger discovery with institutional conviction
    if weights is None:
        weights = {
            "growth": 0.25,
            "quality": 0.20,
            "balance_sheet": 0.15,
            "valuation": 0.15,
            "momentum": 0.05,
            "institutional_conviction": 0.20,
        }

    # Validate weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 0.01:
        log.warning(
            "weights_do_not_sum_to_one",
            total_weight=total_weight,
            normalizing=True,
        )
        # Normalize
        for key in weights:
            weights[key] = weights[key] / total_weight

    # Composite rank: standard weights (6 dimensions)
    composite = (
        weights["growth"] * growth_score
        + weights["quality"] * quality_score
        + weights["balance_sheet"] * balance_sheet_score
        + weights["valuation"] * valuation_score
        + weights["momentum"] * momentum_score
        + weights["institutional_conviction"] * institutional_conviction_score
    )

    # Growth-weighted: emphasize growth even more (40%)
    growth_weights = {
        "growth": 0.40,
        "quality": 0.20,
        "balance_sheet": 0.15,
        "valuation": 0.10,
        "momentum": 0.10,
        "institutional_conviction": 0.05,
    }

    growth_weighted = (
        growth_weights["growth"] * growth_score
        + growth_weights["quality"] * quality_score
        + growth_weights["balance_sheet"] * balance_sheet_score
        + growth_weights["valuation"] * valuation_score
        + growth_weights["momentum"] * momentum_score
        + growth_weights["institutional_conviction"] * institutional_conviction_score
    )

    # Conviction with low rally: boost composite if conviction strong + price not rallied much
    # (Note: this is computed in watchlist.py, here we return placeholder)
    conviction_with_low_rally = None

    # Clamp to 0-100
    composite = min(100.0, max(0.0, composite))
    growth_weighted = min(100.0, max(0.0, growth_weighted))

    return (composite, growth_weighted, conviction_with_low_rally)


def compute_percentile(rank: float, all_ranks: list[float]) -> float | None:
    """
    Compute percentile rank within a universe.

    Args:
        rank: Score to rank
        all_ranks: All scores in universe

    Returns:
        Percentile (0-100) or None if not enough data
    """
    if not all_ranks or len(all_ranks) < 2:
        return None

    sorted_ranks = sorted(all_ranks)
    position = sum(1 for r in sorted_ranks if r < rank)

    percentile = (position / len(sorted_ranks)) * 100.0
    return min(100.0, max(0.0, percentile))
