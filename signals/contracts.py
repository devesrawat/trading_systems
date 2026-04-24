"""
Unified signal contract layer.

Defines the canonical Signal dataclass that all strategies, ML models,
and orchestrator flows must normalize to. Supports JSON round-tripping
and strong validation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SignalType(StrEnum):
    """Signal classification."""

    scanner_hit = "scanner_hit"
    ml_prediction = "ml_prediction"
    fundamental_rank = "fundamental_rank"
    entry = "entry"
    exit = "exit"
    risk_alert = "risk_alert"


class Direction(StrEnum):
    """Trade direction."""

    long = "long"
    short = "short"
    neutral = "neutral"
    exit = "exit"


class EntrySpec(BaseModel):
    """Entry specification for a signal."""

    entry_price: float = Field(..., description="Recommended entry price")
    stop_price: float = Field(..., description="Stop-loss price")
    target_price: float = Field(..., description="Profit target price")
    invalidation_price: float | None = Field(
        None, description="Price above/below which the setup is invalidated"
    )

    @field_validator("entry_price", "stop_price", "target_price", "invalidation_price")
    @classmethod
    def positive_prices(cls, v: float | None) -> float | None:
        """Prices must be positive."""
        if v is not None and v <= 0:
            raise ValueError(f"Price must be positive, got {v}")
        return v


class RiskSpec(BaseModel):
    """Risk specification for a signal."""

    size_hint_pct: float = Field(
        ..., ge=0, le=2, description="Suggested position size as % of capital (0-2%)"
    )
    capital_at_risk: float | None = Field(None, description="Absolute capital at risk")
    liquidity_score: float = Field(default=0.5, ge=0, le=1, description="Liquidity score (0-1)")
    volatility_score: float = Field(default=0.5, ge=0, le=1, description="Volatility score (0-1)")


class Signal(BaseModel):
    """
    Unified signal contract.

    All strategies, ML models, and risk/execution layers normalize to this.
    Designed for JSON serialization and persistence.
    """

    # ------------------------------------------------------------------
    # Core identity
    # ------------------------------------------------------------------

    signal_id: str = Field(..., description="Unique signal identifier (e.g. uuid4)")
    timestamp: datetime = Field(..., description="Signal generation timestamp (UTC)")
    symbol: str = Field(..., description="Trading symbol (e.g. INFY.NS, RELIANCE.NS)")
    exchange: str = Field(default="NSE", description="Exchange code (NSE, BSE, BINANCE, etc.)")
    asset_class: str = Field(
        default="equity", description="Asset class (equity, crypto, futures, etc.)"
    )

    # ------------------------------------------------------------------
    # Strategy / model metadata
    # ------------------------------------------------------------------

    strategy_name: str = Field(
        ..., description="Strategy identifier (e.g. vcp, rs_breakout, ml_long)"
    )
    strategy_version: str = Field(default="1.0", description="Strategy version for reproducibility")
    signal_type: SignalType = Field(..., description="Type of signal")
    direction: Direction = Field(..., description="Trade direction")

    # ------------------------------------------------------------------
    # Confidence & scoring
    # ------------------------------------------------------------------

    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    score: float = Field(
        ..., ge=0, le=1, description="Overall signal score (0-1, typically ML prob or rank)"
    )
    rank: int | None = Field(None, description="Rank in universe (1 = best)")

    # ------------------------------------------------------------------
    # Setup specification
    # ------------------------------------------------------------------

    timeframe: str = Field(default="daily", description="Timeframe (daily, 5minute, etc.)")
    entry: EntrySpec | None = Field(
        None, description="Entry specification (prices, stops, targets)"
    )
    risk: RiskSpec | None = Field(None, description="Risk specification")

    # ------------------------------------------------------------------
    # Features & attribution
    # ------------------------------------------------------------------

    features: dict[str, Any] = Field(
        default_factory=dict, description="Feature dict for debugging/attribution"
    )
    attribution: dict[str, Any] = Field(
        default_factory=dict,
        description="Model/config attribution (version, hash, SHAP values, etc.)",
    )

    # ------------------------------------------------------------------
    # Operational modes
    # ------------------------------------------------------------------

    mode: str = Field(
        default="research",
        description="Execution mode (research, watchlist, paper, live)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata (scanner_extra, notes, etc.)",
    )

    # ------------------------------------------------------------------
    # Raw payload (for debugging)
    # ------------------------------------------------------------------

    raw_payload: dict[str, Any] | None = Field(
        None, description="Original dict from strategy/ML for debugging"
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @field_validator("mode")
    @classmethod
    def valid_mode(cls, v: str) -> str:
        """Mode must be one of the allowed values."""
        allowed = {"research", "watchlist", "paper", "live"}
        if v not in allowed:
            raise ValueError(f"Invalid mode: {v}. Must be one of {allowed}")
        return v

    @field_validator("symbol")
    @classmethod
    def valid_symbol(cls, v: str) -> str:
        """Symbol must be non-empty."""
        if not v or len(v) < 1:
            raise ValueError("symbol cannot be empty")
        return v.upper()

    @field_validator("confidence", "score")
    @classmethod
    def valid_score(cls, v: float) -> float:
        """Scores must be valid floats in [0, 1]."""
        if not (0 <= v <= 1):
            raise ValueError(f"Score must be in [0, 1], got {v}")
        return v

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Signal:
        """Create Signal from dict (validates schema)."""
        return cls.model_validate(data)
