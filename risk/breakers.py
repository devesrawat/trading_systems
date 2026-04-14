"""
Circuit breaker — halts all trading when drawdown limits are breached.

Rules (non-negotiable):
  - Every check() call returns (False, reason) when halted
  - Halts caused by drawdown NEVER auto-reset — manual_reset() only
  - reset_daily() and reset_weekly() update capital baselines but do NOT clear halts
  - State persisted in Redis — survives process restarts
"""

from __future__ import annotations

import json

import structlog
from sqlalchemy import text

from data.redis_keys import RedisKeys
from data.store import get_engine, get_redis
from monitoring.alerts import TelegramAlerter

log = structlog.get_logger(__name__)

_REDIS_KEY = RedisKeys.CIRCUIT_STATE


class CircuitBreaker:
    def __init__(
        self,
        daily_limit: float = 0.03,
        weekly_limit: float = 0.07,
        max_consecutive_losses: int = 5,
    ) -> None:
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.max_consecutive_losses = max_consecutive_losses

        # Internal state — overwritten by Redis if persisted state exists
        self._halted: bool = False
        self._halt_reason: str | None = None
        self._daily_start_capital: float = 0.0
        self._peak_capital: float = 0.0
        self._weekly_start_capital: float = 0.0
        self._consecutive_losses: int = 0
        # Operator soft-pause (separate from risk-triggered halt)
        self._operator_paused: bool = False
        self._operator_pause_reason: str | None = None

        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, current_capital: float) -> tuple[bool, str | None]:
        """
        Return (allowed, reason).

        Evaluates all breaker conditions. If any breach is detected,
        calls halt() and returns (False, reason).
        Always returns (False, reason) when already halted.
        """
        if self._halted:
            return False, self._halt_reason

        # Daily drawdown
        if self._daily_start_capital > 0:
            daily_dd = (self._daily_start_capital - current_capital) / self._daily_start_capital
            if daily_dd > self.daily_limit:
                reason = f"daily drawdown {daily_dd:.2%} exceeded limit {self.daily_limit:.2%}"
                self.halt(reason)
                return False, reason

        # Weekly drawdown
        if self._weekly_start_capital > 0:
            weekly_dd = (self._weekly_start_capital - current_capital) / self._weekly_start_capital
            if weekly_dd > self.weekly_limit:
                reason = f"weekly drawdown {weekly_dd:.2%} exceeded limit {self.weekly_limit:.2%}"
                self.halt(reason)
                return False, reason

        # Consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            reason = (
                f"consecutive losses {self._consecutive_losses} "
                f"reached limit {self.max_consecutive_losses}"
            )
            self.halt(reason)
            return False, reason

        return True, None

    def halt(self, reason: str) -> None:
        """
        Engage the circuit breaker. Only the first call sets the reason.
        Sends a Telegram alert and persists state.
        """
        if self._halted:
            return  # already halted — don't overwrite reason

        self._halted = True
        self._halt_reason = reason
        log.error("circuit_breaker_triggered", reason=reason)

        try:
            TelegramAlerter().alert_circuit_breaker(
                reason=reason,
                dd_pct=0.0,
                capital=self._daily_start_capital,
            )
        except Exception as exc:
            log.warning("telegram_alert_failed", error=str(exc))

        self._persist_state()
        self._write_circuit_event("halt", reason)

    def is_halted(self) -> bool:
        return self._halted or self._operator_paused

    def halt_reason(self) -> str | None:
        """Return the reason the breaker was triggered, or None if not halted."""
        if self._operator_paused:
            return self._operator_pause_reason
        return self._halt_reason if self._halted else None

    def force_halt(self, reason: str) -> None:
        """Alias for halt() — explicitly called by operator/Telegram."""
        self.halt(reason)

    def operator_pause(self, reason: str = "manual_operator_pause") -> None:
        """
        Soft-pause by an operator (e.g. via Telegram /pause).
        Does NOT clear drawdown / risk state. Use manual_reset() for a
        full risk reset (admin-only, never from Telegram).
        """
        self._operator_paused = True
        self._operator_pause_reason = reason
        self._persist_state()
        self._write_circuit_event("operator_pause", reason=reason)
        log.warning("circuit_operator_paused", reason=reason)

    def operator_resume(self) -> None:
        """
        Lift an operator-pause. Only clears operator pause state —
        genuine risk halts are NOT cleared here.
        """
        if not self._operator_paused:
            return
        self._operator_paused = False
        self._operator_pause_reason = None
        self._persist_state()
        self._write_circuit_event("operator_resume")
        log.warning("circuit_operator_resumed")

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def reset_daily(self, current_capital: float) -> None:
        """
        Called at 9:15 IST each morning.
        Updates daily baseline. Does NOT clear a drawdown-caused halt.
        """
        self._daily_start_capital = current_capital
        self._peak_capital = max(self._peak_capital, current_capital)
        self._consecutive_losses = 0
        self._persist_state()
        self._write_circuit_event("daily_reset", capital=current_capital)
        log.info("circuit_daily_reset", capital=current_capital)

    def reset_weekly(self, current_capital: float) -> None:
        """
        Called Monday morning.
        Updates weekly baseline. Does NOT clear a halt.
        """
        self._weekly_start_capital = current_capital
        self._persist_state()
        self._write_circuit_event("weekly_reset", capital=current_capital)
        log.info("circuit_weekly_reset", capital=current_capital)

    def record_loss(self) -> None:
        """Increment consecutive loss counter."""
        self._consecutive_losses += 1
        self._persist_state()

    def record_win(self) -> None:
        """Reset consecutive loss counter on a win."""
        self._consecutive_losses = 0
        self._persist_state()

    def manual_reset(self, current_capital: float) -> None:
        """
        CLI-only manual reset. Clears halt and restores all state.
        Must never be called automatically.
        """
        self._halted = False
        self._halt_reason = None
        self._daily_start_capital = current_capital
        self._weekly_start_capital = current_capital
        self._peak_capital = current_capital
        self._consecutive_losses = 0
        self._persist_state()
        self._write_circuit_event("manual_reset", capital=current_capital)
        log.warning("circuit_breaker_manually_reset", capital=current_capital)

    # ------------------------------------------------------------------
    # DB audit
    # ------------------------------------------------------------------

    def _write_circuit_event(
        self,
        event_type: str,
        reason: str | None = None,
        capital: float | None = None,
    ) -> None:
        """Write a circuit breaker event to the circuit_events audit table."""
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO circuit_events
                            (event_type, reason, capital_at_event)
                        VALUES (:event_type, :reason, :capital)
                    """),
                    {
                        "event_type": event_type,
                        "reason": reason,
                        "capital": capital if capital is not None else self._daily_start_capital,
                    },
                )
                conn.commit()
        except Exception as exc:
            log.warning("circuit_event_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _persist_state(self) -> None:
        state = {
            "halted": self._halted,
            "reason": self._halt_reason,
            "daily_start_capital": self._daily_start_capital,
            "peak_capital": self._peak_capital,
            "weekly_start_capital": self._weekly_start_capital,
            "consecutive_losses": self._consecutive_losses,
            "operator_paused": self._operator_paused,
            "operator_pause_reason": self._operator_pause_reason,
        }
        get_redis().set(_REDIS_KEY, json.dumps(state))

    def _load_state(self) -> None:
        raw = get_redis().get(_REDIS_KEY)
        if raw is None:
            return
        try:
            state = json.loads(raw)
            self._halted = bool(state.get("halted", False))
            self._halt_reason = state.get("reason")
            self._daily_start_capital = float(state.get("daily_start_capital", 0.0))
            self._peak_capital = float(state.get("peak_capital", 0.0))
            self._weekly_start_capital = float(state.get("weekly_start_capital", 0.0))
            self._consecutive_losses = int(state.get("consecutive_losses", 0))
            self._operator_paused = bool(state.get("operator_paused", False))
            self._operator_pause_reason = state.get("operator_pause_reason")
            if self._halted:
                log.warning("circuit_breaker_loaded_halted", reason=self._halt_reason)
            if self._operator_paused:
                log.warning("circuit_operator_paused_on_load", reason=self._operator_pause_reason)
        except Exception as exc:
            log.error("circuit_state_load_failed", error=str(exc))
