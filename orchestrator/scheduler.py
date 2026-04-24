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

        # Daily reporting — 16:00 IST (post-market)
        self._scheduler.add_job(
            func=self._safe(self._daily_reporting),
            trigger=CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone=_TZ_IST),
            id="daily_reporting",
            name="Daily reporting and summary",
            replace_existing=True,
        )

        # Weekly reporting — Friday 17:00 IST
        self._scheduler.add_job(
            func=self._safe(self._weekly_reporting),
            trigger=CronTrigger(hour=17, minute=0, day_of_week="fri", timezone=_TZ_IST),
            id="weekly_reporting",
            name="Weekly reporting and performance summary",
            replace_existing=True,
        )

        # Monthly reporting — Last day of month at 17:00 IST
        self._scheduler.add_job(
            func=self._safe(self._monthly_reporting),
            trigger=CronTrigger(hour=17, minute=0, day="l", timezone=_TZ_IST),
            id="monthly_reporting",
            name="Monthly reporting and review",
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

    def _daily_reporting(self) -> None:
        """Generate and send daily report."""
        try:
            from datetime import datetime

            from monitoring.reporters import DailyMetrics, DailyReport
            from monitoring.telegram_notifier import TelegramNotifier

            # Placeholder metrics — Phase 7 will compute actual values from DB
            metrics = DailyMetrics(
                date=datetime.utcnow(),
                scans_completed=0,
                signals_generated=0,
                signals_executed=0,
                trades_entered=0,
                trades_closed=0,
                total_pnl=0,
                total_pnl_pct=0,
                win_count=0,
                loss_count=0,
                win_rate=0,
                avg_win=0,
                avg_loss=0,
                profit_factor=0,
                max_intraday_dd=0,
                daily_dd=0,
            )
            report = DailyReport.generate(metrics)
            log.info("daily_report_generated")

            # Send via Telegram
            notifier = TelegramNotifier()
            notifier.add_to_batch(report)
            notifier.flush_batch(force=True)
        except Exception as exc:
            log.error("daily_reporting_failed", error=str(exc))

    def _weekly_reporting(self) -> None:
        """Generate and send weekly report."""
        try:
            from datetime import datetime, timedelta

            from monitoring.reporters import WeeklyMetrics, WeeklyReport
            from monitoring.telegram_notifier import TelegramNotifier

            # Placeholder metrics — Phase 7 will compute actual values from DB
            now = datetime.utcnow()
            metrics = WeeklyMetrics(
                week_start=now - timedelta(days=7),
                week_end=now,
                days_traded=5,
                total_pnl=0,
                total_pnl_pct=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                max_drawdown=0,
                strategy_performance={},
                best_performer="",
                worst_performer="",
                best_trade=0,
                worst_trade=0,
                average_trade=0,
            )
            report = WeeklyReport.generate(metrics)
            log.info("weekly_report_generated")

            # Send via Telegram
            notifier = TelegramNotifier()
            notifier.add_to_batch(report)
            notifier.flush_batch(force=True)
        except Exception as exc:
            log.error("weekly_reporting_failed", error=str(exc))

    def _monthly_reporting(self) -> None:
        """Generate and send monthly report."""
        try:
            from datetime import datetime

            from monitoring.reporters import MonthlyMetrics, MonthlyReport
            from monitoring.telegram_notifier import TelegramNotifier

            # Placeholder metrics — Phase 7 will compute actual values from DB
            now = datetime.utcnow()
            first_of_month = now.replace(day=1)
            metrics = MonthlyMetrics(
                month_start=first_of_month,
                month_end=now,
                days_traded=20,
                total_pnl=0,
                total_pnl_pct=0,
                win_rate=0,
                profit_factor=0,
                sharpe_ratio=0,
                max_drawdown=0,
                calmar_ratio=0,
                strategy_rankings=[],
                multibagger_count=0,
                multibagger_candidates=[],
            )
            report = MonthlyReport.generate(metrics)
            log.info("monthly_report_generated")

            # Send via Telegram
            notifier = TelegramNotifier()
            notifier.add_to_batch(report)
            notifier.flush_batch(force=True)
        except Exception as exc:
            log.error("monthly_reporting_failed", error=str(exc))
