"""
System health monitor.

Writes a heartbeat timestamp to Redis after each trading_loop() run.
Checks the heartbeat periodically during market hours (09:15–15:30 IST)
and sends a Telegram alert if the system hasn't written to Redis in > 15 min.

Usage:
    # In trading_loop() — always write even on early return
    HealthMonitor().write_heartbeat()

    # In scheduler — check every 5 min during market hours
    HealthMonitor().send_alert_if_stale()
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import structlog

log = structlog.get_logger(__name__)

_TZ_IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = (9, 15)    # (hour, minute) IST
_MARKET_CLOSE = (15, 30)
_STALE_SECONDS = 15 * 60  # 15 minutes


class HealthMonitor:
    """Heartbeat writer and stale-check alerter."""

    # ------------------------------------------------------------------
    # Write heartbeat (called from trading_loop finally block)
    # ------------------------------------------------------------------

    def write_heartbeat(self) -> None:
        """Write current UTC timestamp to Redis as system heartbeat."""
        try:
            from data.redis_keys import RedisKeys
            from data.store import get_redis
            ts = datetime.utcnow().isoformat()
            get_redis().set(RedisKeys.SYSTEM_HEARTBEAT, ts, ex=3600)
        except Exception as exc:
            log.warning("heartbeat_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Check heartbeat (called from scheduler, every 5 min during hours)
    # ------------------------------------------------------------------

    def send_alert_if_stale(self) -> bool:
        """
        If market hours and no heartbeat within 15 min, send Telegram alert.

        Returns True if alert was sent.
        """
        if not self._is_market_hours():
            return False

        last_ts = self._read_last_heartbeat()
        if last_ts is None:
            log.warning("heartbeat_missing_during_market_hours")
            self._alert("⚠️ <b>HEALTH: No heartbeat found</b>\nSystem may not be running.")
            return True

        age_seconds = (datetime.utcnow() - last_ts).total_seconds()
        if age_seconds > _STALE_SECONDS:
            age_min = int(age_seconds / 60)
            log.warning("heartbeat_stale", age_minutes=age_min)
            self._alert(
                f"⚠️ <b>HEALTH: Stale heartbeat</b>\n"
                f"Last seen {age_min} min ago during market hours.\n"
                "Check if the trading system is running."
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_market_hours(self) -> bool:
        now = datetime.now(tz=_TZ_IST)
        if now.weekday() >= 5:  # Saturday / Sunday
            return False
        now_t = (now.hour, now.minute)
        return _MARKET_OPEN <= now_t <= _MARKET_CLOSE

    def _read_last_heartbeat(self) -> datetime | None:
        try:
            from data.redis_keys import RedisKeys
            from data.store import get_redis
            raw = get_redis().get(RedisKeys.SYSTEM_HEARTBEAT)
            if raw:
                return datetime.fromisoformat(raw if isinstance(raw, str) else raw.decode())
        except Exception as exc:
            log.debug("heartbeat_read_failed", error=str(exc))
        return None

    def _alert(self, message: str) -> None:
        try:
            from monitoring.alerts import TelegramAlerter
            TelegramAlerter().send(message)
        except Exception:
            pass
