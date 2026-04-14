"""
APScheduler job definitions for the NSE + Crypto trading system.

Equity schedule (all times IST / Asia/Kolkata):
  08:45        daily (weekday)  → pre_market_setup()
  09:15–15:25  every 5 min      → trading_loop()
  15:35        daily (weekday)  → post_market_summary()
  Monday 08:30 weekly           → reset_weekly()
  Sunday 02:00 weekly           → retrain_check()

Crypto schedule (UTC, 24/7):
  00:01 daily UTC               → pre_market_setup()  (refresh universe + model)
  every 5 min 24/7              → trading_loop()
  00:30 daily UTC               → post_market_summary()
  Monday 00:05 UTC weekly       → reset_weekly()
  Sunday 01:00 UTC weekly       → retrain_check()

Both schedule (equity + crypto):
  Equity jobs AND crypto jobs run concurrently on their respective schedules.

All jobs are wrapped in try/except — a failed job sends a Telegram alert
but never crashes the scheduler process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

if TYPE_CHECKING:
    from orchestrator.main import TradingSystem

log = structlog.get_logger(__name__)

_TZ_IST = "Asia/Kolkata"
_TZ_UTC = "UTC"


class TradingScheduler:
    """Wraps APScheduler and registers trading jobs based on market_type."""

    def __init__(
        self,
        system: TradingSystem,  # type: ignore[name-defined]
        market_type: str = "equity",
    ) -> None:
        self._system = system
        self._market_type = market_type.lower()
        self._scheduler = BackgroundScheduler(timezone=_TZ_IST)

    def start(self) -> None:
        """Register all jobs for the active market_type and start the scheduler."""
        if self._market_type in ("equity", "both"):
            self._register_equity_jobs()
        if self._market_type in ("crypto", "both"):
            self._register_crypto_jobs()

        # Weekly and retrain run once regardless of mode
        self._register_maintenance_jobs()

        self._scheduler.start()
        log.info(
            "scheduler_started",
            market=self._market_type,
            jobs=len(self._scheduler.get_jobs()),
        )

    def stop(self) -> None:
        self._scheduler.shutdown(wait=True)
        log.info("scheduler_stopped")

    # ------------------------------------------------------------------
    # Equity jobs  (IST weekdays)
    # ------------------------------------------------------------------

    def _register_equity_jobs(self) -> None:
        s = self._system

        # Pre-market setup — 08:45 IST every weekday
        self._scheduler.add_job(
            func=self._safe(s.pre_market_setup),
            trigger=CronTrigger(hour=8, minute=45, day_of_week="mon-fri", timezone=_TZ_IST),
            id="equity_pre_market",
            name="Equity pre-market setup",
            replace_existing=True,
        )

        # Trading loop — every 5 min, 09:15–15:25 IST weekdays
        self._scheduler.add_job(
            func=self._safe(s.trading_loop),
            trigger=CronTrigger(
                hour="9-15",
                minute="15,20,25,30,35,40,45,50,55,0,5,10",
                day_of_week="mon-fri",
                timezone=_TZ_IST,
            ),
            id="equity_trading_loop",
            name="Equity trading loop (5-min)",
            replace_existing=True,
        )

        # Post-market summary — 15:35 IST every weekday
        self._scheduler.add_job(
            func=self._safe(s.post_market_summary),
            trigger=CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone=_TZ_IST),
            id="equity_post_market",
            name="Equity post-market summary",
            replace_existing=True,
        )

        # Health check — every 5 min during market hours (09:15–15:35 IST)
        self._scheduler.add_job(
            func=self._safe(self._health_check),
            trigger=CronTrigger(
                hour="9-15",
                minute="15,20,25,30,35,40,45,50,55,0,5,10",
                day_of_week="mon-fri",
                timezone=_TZ_IST,
            ),
            id="equity_health_check",
            name="Health monitor stale-heartbeat check (5-min)",
            replace_existing=True,
        )

        # FII/DII data fetch — 16:30 IST (post-market data released by NSE)
        self._scheduler.add_job(
            func=self._safe(self._fetch_fii_dii),
            trigger=CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone=_TZ_IST),
            id="equity_fii_dii_fetch",
            name="FII/DII flows fetch (post-market)",
            replace_existing=True,
        )

    # ------------------------------------------------------------------
    # Crypto jobs  (UTC, 24/7)
    # ------------------------------------------------------------------

    def _register_crypto_jobs(self) -> None:
        s = self._system

        # Daily universe + model refresh — 00:01 UTC
        self._scheduler.add_job(
            func=self._safe(s.pre_market_setup),
            trigger=CronTrigger(hour=0, minute=1, timezone=_TZ_UTC),
            id="crypto_pre_market",
            name="Crypto daily setup (UTC midnight)",
            replace_existing=True,
        )

        # Trading loop — every 5 min, 24/7
        self._scheduler.add_job(
            func=self._safe(s.trading_loop),
            trigger=CronTrigger(minute="*/5", timezone=_TZ_UTC),
            id="crypto_trading_loop",
            name="Crypto trading loop (5-min, 24/7)",
            replace_existing=True,
        )

        # Daily summary — 00:30 UTC
        self._scheduler.add_job(
            func=self._safe(s.post_market_summary),
            trigger=CronTrigger(hour=0, minute=30, timezone=_TZ_UTC),
            id="crypto_post_market",
            name="Crypto daily summary (UTC)",
            replace_existing=True,
        )

    # ------------------------------------------------------------------
    # Maintenance jobs  (run regardless of market_type)
    # ------------------------------------------------------------------

    def _register_maintenance_jobs(self) -> None:
        s = self._system

        tz = _TZ_UTC if self._market_type == "crypto" else _TZ_IST
        reset_hour = 0 if self._market_type == "crypto" else 8
        reset_minute = 5 if self._market_type == "crypto" else 30
        retrain_hour = 1 if self._market_type == "crypto" else 2

        # Weekly circuit breaker reset — Monday
        self._scheduler.add_job(
            func=self._safe(s.reset_weekly),
            trigger=CronTrigger(
                hour=reset_hour, minute=reset_minute, day_of_week="mon", timezone=tz
            ),
            id="weekly_reset",
            name="Weekly circuit breaker reset",
            replace_existing=True,
        )

        # Retrain check — Sunday
        self._scheduler.add_job(
            func=self._safe(s.retrain_check),
            trigger=CronTrigger(hour=retrain_hour, minute=0, day_of_week="sun", timezone=tz),
            id="retrain_check",
            name="Sunday retrain check",
            replace_existing=True,
        )

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

    def _health_check(self) -> None:
        """Alert if the trading loop heartbeat is stale."""
        from monitoring.health import HealthMonitor

        HealthMonitor().send_alert_if_stale()

    def _fetch_fii_dii(self) -> None:
        """Fetch post-market FII/DII flows from NSE and cache in Redis."""
        try:
            from data.ingest import NSEDataScraper
            from data.store import get_redis

            flows = NSEDataScraper().get_fii_dii_flows()
            if flows:
                import json

                get_redis().set("trading:fii_dii:latest", json.dumps(flows), ex=86400)
                log.info("fii_dii_fetched", net_fii=flows.get("fii_net"))
        except Exception as exc:
            log.warning("fii_dii_fetch_failed", error=str(exc))
