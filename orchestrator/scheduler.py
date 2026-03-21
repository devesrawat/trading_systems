"""
APScheduler job definitions for the NSE trading system.

Schedule (all times IST / Asia/Kolkata):
  08:45        daily    → pre_market_setup()
  09:15–15:25  every 5m → trading_loop()
  15:35        daily    → post_market_summary()
  Monday 08:30 weekly   → reset_weekly_circuit_breaker()
  Sunday 02:00 weekly   → retrain_check()

All jobs wrapped in try/except — a failed job sends a Telegram alert
but never crashes the scheduler process.
"""
from __future__ import annotations

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

log = structlog.get_logger(__name__)

_TZ = "Asia/Kolkata"


class TradingScheduler:
    """Wraps APScheduler and registers all trading system jobs."""

    def __init__(self, system: "TradingSystem") -> None:  # type: ignore[name-defined]
        self._system = system
        self._scheduler = BackgroundScheduler(timezone=_TZ)

    def start(self) -> None:
        """Register all jobs and start the scheduler."""
        s = self._system

        # Pre-market setup — 08:45 IST every weekday
        self._scheduler.add_job(
            func=self._safe(s.pre_market_setup),
            trigger=CronTrigger(hour=8, minute=45, day_of_week="mon-fri", timezone=_TZ),
            id="pre_market",
            name="Pre-market setup",
            replace_existing=True,
        )

        # Trading loop — every 5 minutes, 09:15–15:25 IST weekdays
        self._scheduler.add_job(
            func=self._safe(s.trading_loop),
            trigger=CronTrigger(
                hour="9-15",
                minute="15,20,25,30,35,40,45,50,55,0,5,10",
                day_of_week="mon-fri",
                timezone=_TZ,
            ),
            id="trading_loop",
            name="Trading loop (5-min)",
            replace_existing=True,
        )

        # Post-market summary — 15:35 IST every weekday
        self._scheduler.add_job(
            func=self._safe(s.post_market_summary),
            trigger=CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone=_TZ),
            id="post_market",
            name="Post-market summary",
            replace_existing=True,
        )

        # Weekly circuit breaker reset — Monday 08:30 IST
        self._scheduler.add_job(
            func=self._safe(s.reset_weekly),
            trigger=CronTrigger(hour=8, minute=30, day_of_week="mon", timezone=_TZ),
            id="weekly_reset",
            name="Weekly circuit breaker reset",
            replace_existing=True,
        )

        # Retrain check — Sunday 02:00 IST
        self._scheduler.add_job(
            func=self._safe(s.retrain_check),
            trigger=CronTrigger(hour=2, minute=0, day_of_week="sun", timezone=_TZ),
            id="retrain_check",
            name="Sunday retrain check",
            replace_existing=True,
        )

        self._scheduler.start()
        log.info("scheduler_started", jobs=len(self._scheduler.get_jobs()))

    def stop(self) -> None:
        self._scheduler.shutdown(wait=True)
        log.info("scheduler_stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe(self, fn):
        """Wrap a job function so exceptions alert Telegram but never crash."""
        def wrapper(*args, **kwargs):
            try:
                fn(*args, **kwargs)
            except Exception as exc:
                log.error("scheduler_job_failed", job=fn.__name__, error=str(exc))
                try:
                    from monitoring.alerts import TelegramAlerter
                    TelegramAlerter().alert_system_error(
                        module=fn.__name__,
                        error_msg=str(exc),
                    )
                except Exception:
                    pass
        wrapper.__name__ = getattr(fn, "__name__", "unknown")
        return wrapper
