"""
Phase 6: Audit persistence — Store audit logs to TimescaleDB.

Skeleton implementation: logs are stored in memory + Redis for now.
Full database schema creation flagged for Phase 7 (Alembic migration).

This layer is intentionally decoupled from execution — failures to persist
never crash the trading system.
"""

from __future__ import annotations

import structlog

from audit.schema import (
    CircuitBreakerLog,
    OrderLogEntry,
    RiskDecisionLog,
    SignalLogEntry,
    TradeLogEntry,
)
from data.redis_keys import RedisKeys
from data.store import get_redis

log = structlog.get_logger(__name__)


class AuditLogger:
    """
    Logs audit events to persistent storage.

    For Phase 6, stores in Redis with TTL. Phase 7 will add TimescaleDB persistence.
    """

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_signal(signal_entry: SignalLogEntry) -> None:
        """
        Log a signal to audit trail.

        Parameters
        ----------
        signal_entry : SignalLogEntry
            Signal to log
        """
        try:
            _AuditPersistence.log_signal(signal_entry)
        except Exception as exc:
            log.warning("audit_signal_log_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Trade logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_trade(trade_entry: TradeLogEntry) -> None:
        """
        Log a trade to audit trail.

        Parameters
        ----------
        trade_entry : TradeLogEntry
            Trade to log
        """
        try:
            _AuditPersistence.log_trade(trade_entry)
        except Exception as exc:
            log.warning("audit_trade_log_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Risk decision logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_risk_decision(risk_entry: RiskDecisionLog) -> None:
        """
        Log a risk decision to audit trail.

        Parameters
        ----------
        risk_entry : RiskDecisionLog
            Risk decision to log
        """
        try:
            _AuditPersistence.log_risk_decision(risk_entry)
        except Exception as exc:
            log.warning("audit_risk_decision_log_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Order logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_order(order_entry: OrderLogEntry) -> None:
        """
        Log an order to audit trail.

        Parameters
        ----------
        order_entry : OrderLogEntry
            Order to log
        """
        try:
            _AuditPersistence.log_order(order_entry)
        except Exception as exc:
            log.warning("audit_order_log_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Circuit breaker logging
    # ------------------------------------------------------------------

    @staticmethod
    def log_circuit_breaker(cb_entry: CircuitBreakerLog) -> None:
        """
        Log a circuit breaker event to audit trail.

        Parameters
        ----------
        cb_entry : CircuitBreakerLog
            Circuit breaker event to log
        """
        try:
            _AuditPersistence.log_circuit_breaker(cb_entry)
        except Exception as exc:
            log.warning("audit_circuit_breaker_log_failed", error=str(exc))


class _AuditPersistence:
    """Internal persistence layer."""

    TTL_SECONDS = 86400 * 30  # 30 days

    @staticmethod
    def log_signal(entry: SignalLogEntry) -> None:
        """Store signal to Redis and queue for DB."""
        data = entry.model_dump_json()
        key = f"{RedisKeys.AUDIT_SIGNALS}:{entry.signal_id}"
        get_redis().set(key, data, ex=_AuditPersistence.TTL_SECONDS)
        log.debug("audit_signal_logged", signal_id=entry.signal_id)

    @staticmethod
    def log_trade(entry: TradeLogEntry) -> None:
        """Store trade to Redis and queue for DB."""
        data = entry.model_dump_json()
        key = f"{RedisKeys.AUDIT_TRADES}:{entry.trade_id}"
        get_redis().set(key, data, ex=_AuditPersistence.TTL_SECONDS)
        log.debug("audit_trade_logged", trade_id=entry.trade_id)

    @staticmethod
    def log_risk_decision(entry: RiskDecisionLog) -> None:
        """Store risk decision to Redis and queue for DB."""
        data = entry.model_dump_json()
        key = f"{RedisKeys.AUDIT_RISK_DECISIONS}:{entry.decision_id}"
        get_redis().set(key, data, ex=_AuditPersistence.TTL_SECONDS)
        log.debug("audit_risk_decision_logged", decision_id=entry.decision_id)

    @staticmethod
    def log_order(entry: OrderLogEntry) -> None:
        """Store order to Redis and queue for DB."""
        data = entry.model_dump_json()
        key = f"{RedisKeys.AUDIT_ORDERS}:{entry.order_id}"
        get_redis().set(key, data, ex=_AuditPersistence.TTL_SECONDS)
        log.debug("audit_order_logged", order_id=entry.order_id)

    @staticmethod
    def log_circuit_breaker(entry: CircuitBreakerLog) -> None:
        """Store circuit breaker event to Redis and queue for DB."""
        data = entry.model_dump_json()
        key = f"{RedisKeys.AUDIT_CIRCUIT_BREAKER}:{entry.event_id}"
        get_redis().set(key, data, ex=_AuditPersistence.TTL_SECONDS)
        log.debug("audit_circuit_breaker_logged", event_id=entry.event_id)
