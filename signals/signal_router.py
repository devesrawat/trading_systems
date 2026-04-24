"""
Signal router — normalize strategy results and ML predictions into unified Signal objects.

Handles:
- Strategy scan() dict → Signal
- XGBoost probability → Signal
- Fundamental ranks → Signal (skeleton)
- Timestamp normalization
- Symbol validation
- Error handling with structured logging
"""

from __future__ import annotations

import contextlib
import hashlib
from datetime import UTC, datetime
from typing import Any

import structlog

from signals.contracts import Direction, EntrySpec, RiskSpec, Signal, SignalType

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core normalizers
# ---------------------------------------------------------------------------


def normalize_strategy_result(
    strategy_result: dict[str, Any],
    symbol: str,
    strategy_name: str,
    confidence: float = 0.5,
    mode: str = "research",
    entry_spec: EntrySpec | None = None,
    risk_spec: RiskSpec | None = None,
) -> Signal:
    """
    Convert a BaseStrategy.scan() result dict into a unified Signal.

    Parameters
    ----------
    strategy_result : dict
        Output from BaseStrategy.scan(). Must contain:
        - symbol: trading symbol
        - strategy: strategy name
        - (optional) current_price, pivot_buy, distance_to_pivot_pct, etc.

    symbol : str
        Trading symbol (for validation/override).

    strategy_name : str
        Strategy name (e.g. "vcp", "rs_breakout").

    confidence : float
        Confidence score (0-1). Default: 0.5 if strategy result doesn't specify.

    mode : str
        Execution mode (research, watchlist, paper, live).

    entry_spec : EntrySpec | None
        Entry specification. If None, attempted to infer from strategy_result.

    risk_spec : RiskSpec | None
        Risk specification. If None, defaults are used.

    Returns
    -------
    Signal object, JSON-serializable.

    Raises
    ------
    ValueError
        If strategy_result is missing required fields or invalid.
    """
    # Validate inputs
    if not strategy_result:
        raise ValueError("strategy_result cannot be empty")

    if not symbol or not strategy_name:
        raise ValueError("symbol and strategy_name are required")

    # Extract / normalize fields
    result_symbol = strategy_result.get("symbol", symbol).upper()
    if result_symbol != symbol.upper():
        log.warning(
            "symbol_mismatch",
            strategy_symbol=result_symbol,
            param_symbol=symbol,
        )
        result_symbol = symbol.upper()

    # Generate signal ID
    signal_id = f"{strategy_name}_{symbol}_{_make_timestamp_key()}"

    # Timestamp (use provided, fall back to now)
    timestamp = _normalize_timestamp(strategy_result.get("timestamp"))

    # Compute confidence / score from strategy_result if available
    strategy_confidence = confidence
    strategy_score = confidence

    # Try to extract from result dict (e.g., rs_breakout might have return_12m_pct)
    if "score" in strategy_result:
        with contextlib.suppress(ValueError, TypeError):
            strategy_score = float(strategy_result["score"])

    if "confidence" in strategy_result:
        with contextlib.suppress(ValueError, TypeError):
            strategy_confidence = float(strategy_result["confidence"])

    # Direction (scanner hits are typically LONG entries, unless signal says otherwise)
    direction = Direction.long
    if strategy_result.get("direction"):
        try:
            direction = Direction(strategy_result["direction"])
        except ValueError:
            log.warning(
                "invalid_direction",
                direction=strategy_result["direction"],
                defaulting_to="long",
            )
            direction = Direction.long

    # Entry specification
    if entry_spec is None:
        entry_spec = _infer_entry_spec(strategy_result)

    # Risk specification
    if risk_spec is None:
        risk_spec = _infer_risk_spec(strategy_result)

    # Build signal
    signal = Signal(
        signal_id=signal_id,
        timestamp=timestamp,
        symbol=result_symbol,
        exchange=strategy_result.get("exchange", "NSE"),
        asset_class=strategy_result.get("asset_class", "equity"),
        strategy_name=strategy_name,
        strategy_version=strategy_result.get("strategy_version", "1.0"),
        signal_type=SignalType.scanner_hit,
        direction=direction,
        confidence=strategy_confidence,
        score=strategy_score,
        rank=strategy_result.get("rank"),
        timeframe=strategy_result.get("timeframe", "daily"),
        entry=entry_spec,
        risk=risk_spec,
        features=strategy_result.get("features", {}),
        attribution={
            "strategy_version": strategy_result.get("strategy_version", "1.0"),
            "raw_strategy_fields": list(strategy_result.keys()),
        },
        mode=mode,
        metadata=strategy_result.get("metadata", {}),
        raw_payload=strategy_result,
    )

    log.info(
        "strategy_result_normalized",
        signal_id=signal.signal_id,
        symbol=symbol,
        strategy=strategy_name,
        score=signal.score,
    )

    return signal


def normalize_ml_signal(
    probability: float,
    symbol: str,
    features: dict[str, Any],
    model_version: str = "1.0",
    timestamp: datetime | None = None,
    mode: str = "research",
    entry_spec: EntrySpec | None = None,
    risk_spec: RiskSpec | None = None,
) -> Signal:
    """
    Convert XGBoost probability into a unified Signal.

    Parameters
    ----------
    probability : float
        Model output P(label=1). Must be in [0, 1].

    symbol : str
        Trading symbol.

    features : dict
        Feature dict used in model (for attribution).

    model_version : str
        Model version identifier.

    timestamp : datetime | None
        Signal timestamp. Default: now (UTC).

    mode : str
        Execution mode (research, watchlist, paper, live).

    entry_spec : EntrySpec | None
        Entry specification. If None, defaults to None (no specific entry).

    risk_spec : RiskSpec | None
        Risk specification. If None, derived from probability.

    Returns
    -------
    Signal object representing the ML prediction.

    Raises
    ------
    ValueError
        If probability not in [0, 1] or symbol is invalid.
    """
    if not (0 <= probability <= 1):
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    if not symbol:
        raise ValueError("symbol is required")

    symbol = symbol.upper()

    # Timestamp
    if timestamp is None:
        timestamp = datetime.now(tz=UTC)

    # Generate signal ID
    signal_id = f"ml_{symbol}_{_make_timestamp_key()}"

    # Determine direction based on probability threshold
    # Threshold = 0.65 (from strategy params)
    threshold = 0.65
    direction = Direction.long if probability >= threshold else Direction.neutral

    # Risk spec: higher probability → slightly higher size hint (capped at 2%)
    if risk_spec is None:
        size_hint = min(0.02, max(0.01, probability * 0.02))
        risk_spec = RiskSpec(
            size_hint_pct=size_hint,
            capital_at_risk=None,
            liquidity_score=0.7,
            volatility_score=0.5,
        )

    # Build signal
    signal = Signal(
        signal_id=signal_id,
        timestamp=timestamp,
        symbol=symbol,
        exchange="NSE",
        asset_class="equity",
        strategy_name="ml_signal",
        strategy_version=model_version,
        signal_type=SignalType.ml_prediction,
        direction=direction,
        confidence=probability,
        score=probability,
        rank=None,
        timeframe="daily",
        entry=entry_spec,
        risk=risk_spec,
        features=features,
        attribution={
            "model_version": model_version,
            "features_hash": _hash_dict(features),
        },
        mode=mode,
        metadata={},
        raw_payload={"probability": probability},
    )

    log.info(
        "ml_signal_normalized",
        signal_id=signal.signal_id,
        symbol=symbol,
        probability=probability,
        direction=direction.value,
    )

    return signal


def normalize_fundamental_rank(
    symbol: str,
    rank: int,
    multibagger_score: float,
    timestamp: datetime | None = None,
    mode: str = "research",
) -> Signal:
    """
    Convert a fundamental rank (multibagger score) into a Signal.

    Skeleton for future implementation. Currently returns a Signal
    with signal_type=fundamental_rank for awareness filtering.

    Parameters
    ----------
    symbol : str
        Trading symbol.

    rank : int
        Rank in universe (1 = best).

    multibagger_score : float
        Multibagger likelihood score (0-1).

    timestamp : datetime | None
        Signal timestamp. Default: now (UTC).

    mode : str
        Execution mode.

    Returns
    -------
    Signal object (currently for filtering/awareness only).

    Raises
    ------
    ValueError
        If rank <= 0 or multibagger_score not in [0, 1].
    """
    if rank <= 0:
        raise ValueError(f"rank must be > 0, got {rank}")

    if not (0 <= multibagger_score <= 1):
        raise ValueError(f"multibagger_score must be in [0, 1], got {multibagger_score}")

    if not symbol:
        raise ValueError("symbol is required")

    symbol = symbol.upper()

    if timestamp is None:
        timestamp = datetime.now(tz=UTC)

    signal_id = f"fundamental_{symbol}_{_make_timestamp_key()}"

    # Direction: neutral for now (fundamental ranks are awareness, not entry signals)
    direction = Direction.neutral

    # Risk spec: lower sizes for fundamental ranks (awareness signal)
    risk_spec = RiskSpec(
        size_hint_pct=0.005,
        capital_at_risk=None,
        liquidity_score=0.6,
        volatility_score=0.5,
    )

    signal = Signal(
        signal_id=signal_id,
        timestamp=timestamp,
        symbol=symbol,
        exchange="NSE",
        asset_class="equity",
        strategy_name="fundamental_rank",
        strategy_version="1.0",
        signal_type=SignalType.fundamental_rank,
        direction=direction,
        confidence=multibagger_score,
        score=multibagger_score,
        rank=rank,
        timeframe="daily",
        entry=None,
        risk=risk_spec,
        features={},
        attribution={},
        mode=mode,
        metadata={"universe_rank": rank},
        raw_payload={"rank": rank, "multibagger_score": multibagger_score},
    )

    log.info(
        "fundamental_rank_normalized",
        signal_id=signal.signal_id,
        symbol=symbol,
        rank=rank,
        score=multibagger_score,
    )

    return signal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_timestamp(ts: Any) -> datetime:
    """
    Normalize various timestamp formats to UTC datetime.

    Accepts:
    - datetime objects
    - ISO 8601 strings
    - Unix timestamps (float/int)
    - None (defaults to now)
    """
    if ts is None:
        return datetime.now(tz=UTC)

    if isinstance(ts, datetime):
        # Ensure UTC
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC)

    if isinstance(ts, str):
        try:
            # Try ISO 8601 parsing
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError as e:
            log.warning("failed_to_parse_timestamp", value=ts, error=str(e))
            return datetime.now(tz=UTC)

    if isinstance(ts, (int, float)):
        try:
            # Assume Unix timestamp
            return datetime.fromtimestamp(ts, tz=UTC)
        except (ValueError, OSError) as e:
            log.warning("failed_to_parse_unix_timestamp", value=ts, error=str(e))
            return datetime.now(tz=UTC)

    log.warning("unknown_timestamp_type", type=type(ts).__name__)
    return datetime.now(tz=UTC)


def _make_timestamp_key() -> str:
    """Create a compact timestamp key for signal IDs."""
    now = datetime.now(tz=UTC)
    return now.strftime("%Y%m%d_%H%M%S")


def _hash_dict(d: dict[str, Any]) -> str:
    """Hash a dict for attribution purposes."""
    import json

    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def _infer_entry_spec(strategy_result: dict[str, Any]) -> EntrySpec | None:
    """
    Attempt to infer entry spec from strategy result dict.

    Looks for common keys like current_price, pivot_buy, stop, target, etc.
    Returns None if insufficient data.
    """
    current = strategy_result.get("current_price")
    pivot = strategy_result.get("pivot_buy")
    stop = strategy_result.get("stop_price")
    target = strategy_result.get("target_price")
    invalidation = strategy_result.get("invalidation_price")

    # If we have entry and stop, compute a target
    if current and pivot and stop:
        # Simple 1:2 RR ratio
        risk = pivot - stop
        computed_target = pivot + (risk * 2)

        return EntrySpec(
            entry_price=float(pivot),
            stop_price=float(stop),
            target_price=float(target or computed_target),
            invalidation_price=float(invalidation) if invalidation else None,
        )

    # If we have all four, use them
    if current and stop and target:
        entry = pivot if pivot else current
        return EntrySpec(
            entry_price=float(entry),
            stop_price=float(stop),
            target_price=float(target),
            invalidation_price=float(invalidation) if invalidation else None,
        )

    return None


def _infer_risk_spec(strategy_result: dict[str, Any]) -> RiskSpec:
    """
    Attempt to infer risk spec from strategy result dict.

    Looks for size_hint_pct, capital_at_risk, liquidity_score, volatility_score.
    Returns a default RiskSpec with safe defaults.
    """
    size_hint = strategy_result.get("size_hint_pct", 0.01)
    capital_at_risk = strategy_result.get("capital_at_risk")
    liquidity_score = strategy_result.get("liquidity_score", 0.7)
    volatility_score = strategy_result.get("volatility_score", 0.5)

    # Ensure size_hint is in [0, 2]
    size_hint = max(0.001, min(0.02, float(size_hint)))

    return RiskSpec(
        size_hint_pct=size_hint,
        capital_at_risk=capital_at_risk,
        liquidity_score=float(liquidity_score),
        volatility_score=float(volatility_score),
    )
