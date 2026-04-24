"""
Deterministic scoring formulas for multibagger identification.

All formulas are pure functions with no side effects, calibrated for Indian
small-cap multibaggers. Scores are 0-100 with explicit documentation.
"""

from __future__ import annotations

import math

import structlog

log = structlog.get_logger(__name__)


def compute_growth_score(
    revenue_cagr_3y: float | None,
    revenue_cagr_5y: float | None,
    net_income_cagr_3y: float | None,
    net_income_cagr_5y: float | None,
) -> float:
    """
    Growth score based on revenue and net income CAGR.

    Formula:
    - Weight 3-year CAGR at 60%, 5-year at 40%
    - Revenue CAGR: 0-40% maps to 0-100 score (sigmoid)
    - Net income CAGR: 0-35% maps to 0-100 score (higher growth required)
    - Final: average of revenue and net income scores (if both available)
    - Missing data: use available metric, floor at 0

    Args:
        revenue_cagr_3y: 3-year revenue CAGR (%)
        revenue_cagr_5y: 5-year revenue CAGR (%)
        net_income_cagr_3y: 3-year net income CAGR (%)
        net_income_cagr_5y: 5-year net income CAGR (%)

    Returns:
        Score 0-100
    """

    def sigmoid_growth(cagr: float, inflection: float = 25.0, steepness: float = 0.15) -> float:
        """Sigmoid growth calibrated for Indian equities."""
        if cagr is None or cagr < 0:
            return 0.0
        # Sigmoid: 100 / (1 + exp(-k * (x - c)))
        try:
            exponent = -steepness * (cagr - inflection)
            if exponent < -700:  # Prevent overflow
                return 100.0
            if exponent > 700:
                return 0.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 0.0

    scores = []

    # Revenue score: weighted average of 3y and 5y CAGR
    if revenue_cagr_3y is not None or revenue_cagr_5y is not None:
        rev_score = 0.0
        if revenue_cagr_3y is not None:
            rev_score += 0.6 * sigmoid_growth(revenue_cagr_3y)
        if revenue_cagr_5y is not None:
            rev_score += 0.4 * sigmoid_growth(revenue_cagr_5y)
        scores.append(rev_score)

    # Net income score: slightly higher bar (requires 35% growth for 100)
    if net_income_cagr_3y is not None or net_income_cagr_5y is not None:
        ni_score = 0.0
        if net_income_cagr_3y is not None:
            ni_score += 0.6 * sigmoid_growth(net_income_cagr_3y, inflection=20.0)
        if net_income_cagr_5y is not None:
            ni_score += 0.4 * sigmoid_growth(net_income_cagr_5y, inflection=20.0)
        scores.append(ni_score)

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def compute_quality_score(
    roe: float | None,
    roce: float | None,
    profit_margin: float | None,
    ebitda_margin: float | None,
    consistency_score: float | None = None,
) -> float:
    """
    Quality score based on profitability and consistency.

    Formula:
    - ROE > 15% and ROCE > 15% → high quality
    - Map to 0-100:
      - ROE: 0-30% maps to 0-100 (25% = 50)
      - ROCE: 0-35% maps to 0-100 (25% = ~60)
      - Profit margin: 0-25% maps to 0-100 (10% = 40)
      - EBITDA margin: 0-40% maps to 0-100 (20% = 50)
    - Consistency: penalty if capital is being destroyed (negative growth years)
    - Average of available metrics

    Args:
        roe: Return on Equity (%)
        roce: Return on Capital Employed (%)
        profit_margin: Net profit margin (%)
        ebitda_margin: EBITDA margin (%)
        consistency_score: Historical consistency (0-1)

    Returns:
        Score 0-100
    """

    def sigmoid_quality(value: float, inflection: float, steepness: float = 0.08) -> float:
        """Sigmoid calibrated for quality metrics."""
        if value is None or value < 0:
            return 0.0
        try:
            exponent = -steepness * (value - inflection)
            if exponent < -700:
                return 100.0
            if exponent > 700:
                return 0.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 0.0

    scores = []

    # ROE: 15% is benchmark, 25% is excellent
    if roe is not None:
        scores.append(sigmoid_quality(roe, inflection=15.0))

    # ROCE: 15% is benchmark, 25% is excellent
    if roce is not None:
        scores.append(sigmoid_quality(roce, inflection=15.0))

    # Profit margin: 5% is acceptable, 15% is good
    if profit_margin is not None:
        scores.append(sigmoid_quality(profit_margin, inflection=8.0))

    # EBITDA margin: 15% is acceptable, 30% is good
    if ebitda_margin is not None:
        scores.append(sigmoid_quality(ebitda_margin, inflection=15.0))

    if not scores:
        return 0.0

    avg_score = sum(scores) / len(scores)

    # Apply consistency penalty if provided
    if consistency_score is not None and consistency_score < 0.8:
        penalty = (1.0 - consistency_score) * 20  # Up to 20-point penalty
        avg_score = max(0.0, avg_score - penalty)

    return min(100.0, avg_score)


def compute_balance_sheet_score(
    debt_to_equity: float | None,
    debt_to_revenue: float | None,
    current_ratio: float | None,
    interest_coverage: float | None,
) -> float:
    """
    Balance sheet health score.

    Formula:
    - Debt/Equity: 0-1.5 is healthy → inverted sigmoid (lower is better)
    - Debt/Revenue: 0-3.0 is healthy → inverted sigmoid
    - Current ratio: 1.0-2.5 is ideal → gaussian around 1.5
    - Interest coverage: 2.0+ is safe → sigmoid (higher is better)
    - Average of available metrics

    Args:
        debt_to_equity: Debt to Equity ratio
        debt_to_revenue: Debt to Revenue ratio
        current_ratio: Current Ratio
        interest_coverage: Interest Coverage Ratio

    Returns:
        Score 0-100
    """

    def inverted_sigmoid(value: float, inflection: float, steepness: float = 0.1) -> float:
        """Lower values are better (inverted sigmoid)."""
        if value is None or value < 0:
            return 0.0
        try:
            exponent = steepness * (value - inflection)  # Note: positive for inversion
            if exponent > 700:
                return 0.0
            if exponent < -700:
                return 100.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 0.0

    def gaussian_liquidity(value: float, center: float = 1.5, width: float = 0.5) -> float:
        """Gaussian around ideal current ratio."""
        if value is None or value <= 0:
            return 0.0
        try:
            score = 100.0 * math.exp(-((value - center) ** 2) / (2 * width**2))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 0.0

    def sigmoid_coverage(value: float, inflection: float = 2.0, steepness: float = 0.5) -> float:
        """Coverage ratio sigmoid."""
        if value is None or value < 0:
            return 0.0
        try:
            exponent = -steepness * (value - inflection)
            if exponent < -700:
                return 100.0
            if exponent > 700:
                return 0.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 0.0

    scores = []

    # Debt/Equity: lower is better, 1.0 is good, >2.0 is risky
    if debt_to_equity is not None:
        scores.append(inverted_sigmoid(debt_to_equity, inflection=1.0, steepness=0.15))

    # Debt/Revenue: lower is better
    if debt_to_revenue is not None:
        scores.append(inverted_sigmoid(debt_to_revenue, inflection=2.0, steepness=0.1))

    # Current ratio: gaussian around 1.5
    if current_ratio is not None:
        scores.append(gaussian_liquidity(current_ratio, center=1.5, width=0.8))

    # Interest coverage: higher is better, 2.0x is baseline
    if interest_coverage is not None:
        scores.append(sigmoid_coverage(interest_coverage, inflection=2.5))

    if not scores:
        return 0.0

    return sum(scores) / len(scores)


def compute_valuation_score(
    pe: float | None,
    peg: float | None,
    pb: float | None,
    ps: float | None,
    revenue_cagr: float | None = None,
) -> float:
    """
    Valuation score: lower multiples relative to growth = higher score.

    Formula:
    - PEG (P/E to Growth): <1.0 is undervalued, >2.0 is overvalued
      Sigmoid with inflection at 1.2
    - PE: penalize if PE > 40 or if PE > 2x PEG-implied multiple
    - PB: <3.0 is attractive, >5.0 is expensive
    - PS: <2.0 is attractive, >5.0 is expensive
    - Weight PEG highest (60%), PE (25%), PB+PS (15%)

    Args:
        pe: P/E ratio
        peg: PEG ratio
        pb: P/B ratio
        ps: P/S ratio
        revenue_cagr: Revenue CAGR (%) for supplementary valuation check

    Returns:
        Score 0-100
    """

    def peg_sigmoid(peg_val: float, inflection: float = 1.2) -> float:
        """PEG sigmoid: <1.2 is undervalued, >1.5 is overvalued."""
        if peg_val is None or peg_val <= 0:
            return 50.0  # Neutral if missing
        try:
            exponent = 2.0 * (peg_val - inflection)  # Steeper curve
            if exponent > 700:
                return 0.0
            if exponent < -700:
                return 100.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 50.0

    def pe_sigmoid(pe_val: float) -> float:
        """PE sigmoid: 15-30 is fair, >40 is expensive."""
        if pe_val is None or pe_val <= 0:
            return 50.0
        try:
            if pe_val < 10:
                return 90.0
            elif pe_val > 50:
                return 20.0
            exponent = 0.1 * (pe_val - 25.0)
            if exponent > 700:
                return 20.0
            if exponent < -700:
                return 80.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 50.0

    def pb_ps_sigmoid(val: float, inflection: float) -> float:
        """PB/PS sigmoid."""
        if val is None or val <= 0:
            return 50.0
        try:
            exponent = 0.15 * (val - inflection)
            if exponent > 700:
                return 0.0
            if exponent < -700:
                return 100.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 50.0

    scores = []

    # PEG: primary valuation metric (weight 60%)
    if peg is not None:
        scores.append((0.6, peg_sigmoid(peg)))
    elif pe is not None and revenue_cagr is not None:
        # Derive PEG if missing
        implied_peg = pe / max(revenue_cagr, 1.0)
        scores.append((0.6, peg_sigmoid(implied_peg)))

    # PE: secondary (weight 25%)
    if pe is not None:
        scores.append((0.25, pe_sigmoid(pe)))

    # PB: supplementary (weight 7.5%)
    if pb is not None:
        scores.append((0.075, pb_ps_sigmoid(pb, inflection=3.0)))

    # PS: supplementary (weight 7.5%)
    if ps is not None:
        scores.append((0.075, pb_ps_sigmoid(ps, inflection=2.5)))

    if not scores:
        return 50.0  # Neutral if all missing

    total_weight = sum(w for w, _ in scores)
    total_score = sum(w * s for w, s in scores)

    # Normalize by actual weights used
    if total_weight > 0:
        return total_score / total_weight
    return 50.0


def compute_momentum_score(
    institutional_pct: float | None,
    fii_change_pct: float | None,
    dii_change_pct: float | None,
    price_momentum_3m: float | None = None,
) -> float:
    """
    Momentum score based on institutional buying and price trends.

    Formula:
    - Institutional ownership > 30% → positive signal
    - FII/DII buying (positive change) → positive signal
    - Price momentum positive in last 3 months → signal
    - Avoid penny stocks (<5% institutional) → penalty
    - Average of available metrics

    Args:
        institutional_pct: Institutional ownership (%)
        fii_change_pct: FII change in last quarter (%)
        dii_change_pct: DII change in last quarter (%)
        price_momentum_3m: Price momentum in last 3 months (%)

    Returns:
        Score 0-100
    """

    def institutional_sigmoid(inst_pct: float) -> float:
        """Institutional ownership: >30% is positive."""
        if inst_pct is None or inst_pct < 0:
            return 30.0  # Penalty for missing/low ownership
        try:
            exponent = -0.15 * (inst_pct - 25.0)
            if exponent < -700:
                return 100.0
            if exponent > 700:
                return 0.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 50.0

    def change_sigmoid(change_pct: float, inflection: float = 2.0) -> float:
        """FII/DII change: positive is good."""
        if change_pct is None:
            return 50.0
        try:
            exponent = -0.1 * (change_pct - inflection)
            if exponent < -700:
                return 100.0
            if exponent > 700:
                return 0.0
            score = 100.0 / (1.0 + math.exp(exponent))
            return min(100.0, max(0.0, score))
        except (ValueError, OverflowError):
            return 50.0

    scores = []

    # Institutional ownership
    if institutional_pct is not None:
        scores.append(institutional_sigmoid(institutional_pct))

    # FII change
    if fii_change_pct is not None:
        scores.append(change_sigmoid(fii_change_pct, inflection=1.0))

    # DII change
    if dii_change_pct is not None:
        scores.append(change_sigmoid(dii_change_pct, inflection=2.0))

    # Price momentum (if available)
    if price_momentum_3m is not None:
        if price_momentum_3m > 10:
            scores.append(80.0)
        elif price_momentum_3m > 5:
            scores.append(65.0)
        elif price_momentum_3m > 0:
            scores.append(55.0)
        else:
            scores.append(40.0)

    if not scores:
        return 50.0  # Neutral if all missing

    return sum(scores) / len(scores)
