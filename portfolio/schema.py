"""
Portfolio-level risk schema and models.

Defines Pydantic models for portfolio state, risk metrics, and limits.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PortfolioPosition(BaseModel):
    """A single position in the portfolio."""

    symbol: str = Field(..., description="Trading symbol")
    qty: float = Field(..., description="Quantity held")
    entry_price: float = Field(..., gt=0, description="Entry price")
    current_price: float = Field(..., gt=0, description="Current market price")

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L in rupees."""
        return self.qty * (self.current_price - self.entry_price)

    @property
    def market_value(self) -> float:
        """Market value of the position."""
        return self.qty * self.current_price


class PortfolioState(BaseModel):
    """Current portfolio state snapshot."""

    positions: dict[str, PortfolioPosition] = Field(
        default_factory=dict, description="Current open positions"
    )
    total_capital: float = Field(..., gt=0, description="Total account capital")
    cash_available: float = Field(..., ge=0, description="Available cash")

    @property
    def gross_position_value(self) -> float:
        """Sum of all position market values."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def net_worth(self) -> float:
        """Total portfolio value = cash + positions."""
        return self.cash_available + self.gross_position_value

    @property
    def deployed_pct(self) -> float:
        """Percentage of capital deployed in positions."""
        if self.total_capital <= 0:
            return 0.0
        return self.gross_position_value / self.total_capital


class SectorExposure(BaseModel):
    """Sector concentration metrics."""

    sector: str = Field(..., description="Sector name")
    pct_of_capital: float = Field(..., ge=0, le=1, description="Exposure as % of total capital")
    rank: int = Field(..., ge=1, description="Rank by exposure (1 = highest)")
    symbols: list[str] = Field(default_factory=list, description="Symbols in this sector")


class CorrelationMetrics(BaseModel):
    """Correlation analysis for a new position."""

    symbol: str = Field(..., description="Symbol being evaluated")
    avg_correlation: float = Field(ge=-1, le=1, description="Average correlation to open positions")
    max_correlation: float = Field(ge=-1, le=1, description="Highest pairwise correlation")
    correlated_symbols: list[str] = Field(
        default_factory=list, description="Symbols with high correlation (>0.6)"
    )


class LiquidityMetrics(BaseModel):
    """Liquidity analysis for a position."""

    symbol: str = Field(..., description="Symbol")
    liquidity_score: float = Field(ge=0, le=100, description="Liquidity score (0-100)")
    bid_ask_spread_bps: float = Field(ge=0, description="Bid-ask spread in basis points")
    avg_volume_qty: float = Field(ge=0, description="Average daily volume (units)")
    slippage_estimate_bps: float = Field(
        ge=0, description="Estimated slippage for position size (bps)"
    )


class RiskLimits(BaseModel):
    """Hard and soft limits for portfolio risk."""

    max_sector_pct: float = Field(
        default=0.25, ge=0.01, le=1.0, description="Max sector exposure as % of capital"
    )
    max_single_stock_pct: float = Field(
        default=0.02, ge=0.001, le=0.1, description="Max single stock % (hard-capped at 2%)"
    )
    max_correlation_to_portfolio: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Max avg correlation to existing positions"
    )
    min_liquidity_score: float = Field(
        default=70, ge=0, le=100, description="Minimum liquidity score to trade"
    )
    max_turnover_pct_annual: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Max annual turnover as % of capital"
    )
    max_positions: int = Field(
        default=20, ge=1, le=100, description="Maximum number of open positions"
    )
    max_intraday_trades_per_symbol: int = Field(
        default=3, ge=1, le=10, description="Max intraday trades per symbol"
    )

    @field_validator("max_single_stock_pct")
    @classmethod
    def validate_max_single_stock(cls, v: float) -> float:
        """Hard cap at 2%."""
        if v > 0.02:
            raise ValueError(f"max_single_stock_pct {v:.1%} exceeds hard limit of 2%")
        return v

    @field_validator("max_sector_pct")
    @classmethod
    def validate_max_sector(cls, v: float) -> float:
        """Sector limit must be >= single stock limit."""
        if v < 0.02:
            raise ValueError(f"max_sector_pct {v:.1%} must be >= 2% (single stock limit)")
        return v


class RiskDecision(BaseModel):
    """Pre-execution risk check decision."""

    allowed: bool = Field(..., description="Whether to allow the order")
    reason: str = Field(..., description="Reason for decision (allow or deny)")
    capital_allocated: float = Field(..., ge=0, description="Capital approved for execution")
    priority: int = Field(..., ge=1, le=5, description="Order priority (1=high, 5=low)")
    checks_passed: list[str] = Field(default_factory=list, description="Checks that passed")
    checks_failed: list[str] = Field(default_factory=list, description="Checks that failed")
    adjustments: dict[str, float] = Field(
        default_factory=dict, description="Risk-driven capital adjustments (key: adjustment_name)"
    )

    def add_passed_check(self, check_name: str) -> None:
        """Record a passed check."""
        if check_name not in self.checks_passed:
            self.checks_passed.append(check_name)

    def add_failed_check(self, check_name: str, reason: str | None = None) -> None:
        """Record a failed check and update reason."""
        if check_name not in self.checks_failed:
            self.checks_failed.append(check_name)
        if reason:
            self.reason = reason

    def set_denied(self, reason: str) -> None:
        """Mark decision as denied with reason."""
        self.allowed = False
        self.reason = reason
