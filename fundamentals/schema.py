"""
Pydantic models for fundamentals: quarterly financials, valuations, shareholding.

All models track source, timestamp, and confidence level for data provenance.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfidenceLevel(StrEnum):
    """Data source confidence level."""

    high = "high"  # Official filings (NSE, BSE)
    medium = "medium"  # Aggregator APIs (Screener, Trendlyne)
    low = "low"  # Derived or estimated


class QuarterlyFinancials(BaseModel):
    """Quarterly financial snapshot from company filings."""

    symbol: str = Field(..., description="Stock symbol (e.g. 'INFY')")
    timestamp: datetime = Field(..., description="Date of filing")
    quarter: str = Field(..., description="Quarter (e.g. 'Q1FY24')")
    source: str = Field(..., description="Data source (e.g. 'NSE', 'Screener', 'Trendlyne')")
    confidence: ConfidenceLevel = Field(
        default=ConfidenceLevel.medium, description="Data confidence level"
    )

    # Income statement (in INR millions)
    revenue: float | None = Field(None, description="Total revenue")
    net_income: float | None = Field(None, description="Net profit/loss")
    ebitda: float | None = Field(None, description="EBITDA")
    gross_profit: float | None = Field(None, description="Gross profit")
    operating_income: float | None = Field(None, description="Operating income")
    tax_expense: float | None = Field(None, description="Tax expense")
    interest_expense: float | None = Field(None, description="Interest expense")

    # Cash flow (in INR millions)
    fcf: float | None = Field(None, description="Free cash flow")
    operating_cf: float | None = Field(None, description="Operating cash flow")
    investing_cf: float | None = Field(None, description="Investing cash flow")

    # Balance sheet (in INR millions)
    equity: float | None = Field(None, description="Total shareholders' equity")
    debt: float | None = Field(None, description="Total debt (short + long term)")
    cash: float | None = Field(None, description="Cash and cash equivalents")
    current_assets: float | None = Field(None, description="Current assets")
    current_liabilities: float | None = Field(None, description="Current liabilities")
    inventory: float | None = Field(None, description="Inventory")
    receivables: float | None = Field(None, description="Trade receivables")
    payables: float | None = Field(None, description="Trade payables")

    @field_validator("revenue", "net_income", "equity", mode="before")
    @classmethod
    def non_negative(cls, v: float | None) -> float | None:
        """Revenue, income, equity should be positive or None."""
        if v is not None and v < 0:
            raise ValueError(f"Value must be non-negative, got {v}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "INFY",
                "timestamp": "2024-01-15T00:00:00Z",
                "quarter": "Q3FY24",
                "source": "NSE",
                "confidence": "high",
                "revenue": 250000.0,
                "net_income": 45000.0,
                "ebitda": 65000.0,
                "equity": 180000.0,
                "debt": 30000.0,
            }
        }
    )


class Valuations(BaseModel):
    """Valuation metrics snapshot."""

    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Date of snapshot")
    source: str = Field(..., description="Data source")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.medium)

    # Multiples
    pe: float | None = Field(None, description="P/E ratio")
    pb: float | None = Field(None, description="P/B ratio")
    ps: float | None = Field(None, description="P/S ratio")
    peg: float | None = Field(None, description="PEG ratio (P/E to growth)")
    pcf: float | None = Field(None, description="P/CF ratio")

    # Profitability
    roe: float | None = Field(None, description="Return on Equity (%)")
    roce: float | None = Field(None, description="Return on Capital Employed (%)")
    roa: float | None = Field(None, description="Return on Assets (%)")
    profit_margin: float | None = Field(None, description="Net profit margin (%)")
    operating_margin: float | None = Field(None, description="Operating margin (%)")
    ebitda_margin: float | None = Field(None, description="EBITDA margin (%)")

    # Leverage & liquidity
    debt_to_equity: float | None = Field(None, description="Debt/Equity ratio")
    debt_to_assets: float | None = Field(None, description="Debt/Assets ratio")
    debt_to_revenue: float | None = Field(None, description="Debt/Revenue ratio")
    current_ratio: float | None = Field(None, description="Current ratio")
    quick_ratio: float | None = Field(None, description="Quick ratio")
    interest_coverage: float | None = Field(None, description="Interest coverage ratio")

    @field_validator("pe", "pb", "ps", "peg", "pcf", mode="before")
    @classmethod
    def positive_multiples(cls, v: float | None) -> float | None:
        """Multiples should be positive or None."""
        if v is not None and v < 0:
            raise ValueError(f"Multiple must be positive, got {v}")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "INFY",
                "timestamp": "2024-01-15T00:00:00Z",
                "source": "Screener",
                "confidence": "high",
                "pe": 25.5,
                "pb": 4.2,
                "roe": 18.5,
                "roce": 22.3,
                "debt_to_equity": 0.15,
            }
        }
    )


class Shareholding(BaseModel):
    """Shareholding pattern snapshot."""

    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Date of snapshot")
    source: str = Field(..., description="Data source")
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.medium)

    # Percentage ownership
    promoter_pct: float | None = Field(None, ge=0, le=100, description="Promoter ownership %")
    institutional_pct: float | None = Field(
        None, ge=0, le=100, description="Institutional ownership %"
    )
    public_pct: float | None = Field(None, ge=0, le=100, description="Public ownership %")

    # Volume and momentum
    fii_qty: float | None = Field(None, description="FII holding quantity")
    dii_qty: float | None = Field(None, description="DII holding quantity")
    fii_change_pct: float | None = Field(None, description="FII change in last quarter (%)")
    dii_change_pct: float | None = Field(None, description="DII change in last quarter (%)")

    # Pledged
    pledged_pct: float | None = Field(None, ge=0, le=100, description="Pledged shareholding %")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "INFY",
                "timestamp": "2024-01-15T00:00:00Z",
                "source": "NSE",
                "confidence": "high",
                "promoter_pct": 45.5,
                "institutional_pct": 35.2,
                "public_pct": 19.3,
                "fii_qty": 125000000,
            }
        }
    )


class FundamentalsScores(BaseModel):
    """Composite fundamentals scores for multibagger ranking."""

    symbol: str = Field(..., description="Stock symbol")
    timestamp: datetime = Field(..., description="Timestamp of scoring")
    source: str = Field(..., description="Scoring source")

    # Individual scores (0-100)
    growth_score: float = Field(..., ge=0, le=100, description="Growth score")
    quality_score: float = Field(..., ge=0, le=100, description="Quality score")
    balance_sheet_score: float = Field(..., ge=0, le=100, description="Balance sheet score")
    valuation_score: float = Field(..., ge=0, le=100, description="Valuation score")
    momentum_score: float = Field(..., ge=0, le=100, description="Momentum score")

    # Composite rank (0-100)
    composite_rank: float = Field(..., ge=0, le=100, description="Weighted composite rank")
    percentile: float | None = Field(None, ge=0, le=100, description="Rank percentile in universe")

    # Data confidence
    confidence: ConfidenceLevel = Field(default=ConfidenceLevel.medium)

    # Metadata
    growth_weighted: float | None = Field(None, description="Composite rank with growth weighting")
    data_completeness: float = Field(
        default=0.0, ge=0, le=1, description="Fraction of available data used (0-1)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "symbol": "INFY",
                "timestamp": "2024-01-15T00:00:00Z",
                "source": "fundamentals",
                "growth_score": 75,
                "quality_score": 82,
                "balance_sheet_score": 88,
                "valuation_score": 65,
                "momentum_score": 72,
                "composite_rank": 76,
                "percentile": 85,
            }
        }
    )
