"""
Phase 6: Audit schema — Pydantic models for Signal, Trade, Risk decision logs.

These models define the shape of audit events. Persistence is handled
in audit/persistence.py. Full database schema creation flagged for Phase 7.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SignalLogEntry(BaseModel):
    """Audit log entry for a signal."""

    signal_id: str = Field(..., description="Unique signal identifier")
    timestamp: datetime = Field(..., description="Signal generation timestamp (UTC)")
    symbol: str
    strategy_name: str
    direction: str  # "long", "short", "neutral"
    confidence: float  # 0-1
    score: float  # 0-1
    signal_type: str  # "scanner_hit", "ml_prediction", etc.
    mode: str  # "research", "watchlist", "paper", "live"
    features: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    raw_payload: dict[str, Any] | None = None


class TradeLogEntry(BaseModel):
    """Audit log entry for a trade (entry or exit)."""

    trade_id: str = Field(..., description="Unique trade identifier")
    timestamp: datetime = Field(..., description="Trade execution timestamp (UTC)")
    symbol: str
    direction: str  # "long", "short"
    order_type: str  # "entry", "exit", "reduce"
    quantity: int
    price: float
    broker_order_id: str | None = None
    status: str  # "pending", "executed", "rejected", "cancelled"
    error_msg: str | None = None
    metadata: dict[str, Any] | None = None


class RiskDecisionLog(BaseModel):
    """Audit log entry for risk decisions."""

    decision_id: str = Field(..., description="Unique decision identifier")
    timestamp: datetime = Field(..., description="Decision timestamp (UTC)")
    decision_type: str  # "position_rejected", "risk_halted", "dd_limit_exceeded", etc.
    reason: str
    symbol: str | None = None  # None for system-wide decisions
    details: dict[str, Any] | None = None


class OrderLogEntry(BaseModel):
    """Audit log entry for order lifecycle."""

    order_id: str = Field(..., description="Unique order identifier")
    timestamp: datetime = Field(..., description="Order event timestamp (UTC)")
    event_type: str  # "submitted", "acknowledged", "filled", "rejected", "cancelled"
    symbol: str
    quantity: int
    price: float
    broker_order_id: str | None = None
    status_msg: str | None = None
    metadata: dict[str, Any] | None = None


class CircuitBreakerLog(BaseModel):
    """Audit log entry for circuit breaker events."""

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    event_type: str  # "triggered", "reset", "manual_reset"
    reason: str  # "daily_dd_limit", "weekly_dd_limit", etc.
    drawdown_pct: float | None = None
    capital: float | None = None
    details: dict[str, Any] | None = None
