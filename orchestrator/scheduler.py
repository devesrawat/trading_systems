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

from typing import TYPE_CHECKING, Any

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

        # QUICK WIN 3: Track job failures for consecutive failure alerts
        self._job_failures: dict[str, list[float]] = {}  # job_name -> [failure_timestamps]
        self._job_last_start: dict[str, float] = {}  # job_name -> start_time

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

        # Kite access-token refresh — 08:30 IST (15 min before pre-market setup)
        self._scheduler.add_job(
            func=self._safe(s._refresh_kite_token),
            trigger=CronTrigger(hour=8, minute=30, day_of_week="mon-fri", timezone=_TZ_IST),
            id="kite_token_refresh",
            name="Kite daily access-token refresh",
            replace_existing=True,
        )

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

        # A/B test reporting — Friday 17:00 IST (after weekly report)
        self._scheduler.add_job(
            func=self._safe(self._ab_test_reporting),
            trigger=CronTrigger(hour=17, minute=5, day_of_week="fri", timezone=_TZ_IST),
            id="ab_test_reporting",
            name="A/B test weekly comparison and promotion check",
            replace_existing=True,
        )

        # Monthly reporting — Last day of month at 17:00 IST
        self._scheduler.add_job(
            func=self._safe(self._monthly_reporting),
            trigger=CronTrigger(hour=17, minute=0, day="31", timezone=_TZ_IST),
            id="monthly_reporting",
            name="Monthly reporting and review",
            replace_existing=True,
        )

        # Phase 8: Monthly ensemble retraining (1st of month, 2 AM IST)
        self._scheduler.add_job(
            func=self._safe(self._monthly_ensemble_retrain),
            trigger=CronTrigger(hour=2, minute=0, day="1", timezone=_TZ_IST),
            id="monthly_ensemble_retrain",
            name="Phase 8 — Monthly walk-forward ensemble retraining",
            replace_existing=True,
        )

        # Phase 8: Weekly concept drift check (Monday 6 AM IST)
        self._scheduler.add_job(
            func=self._safe(self._weekly_concept_drift_check),
            trigger=CronTrigger(hour=6, minute=0, day_of_week="mon", timezone=_TZ_IST),
            id="weekly_concept_drift_check",
            name="Phase 8 — Weekly concept drift detection",
            replace_existing=True,
        )

        # Hermes Wealth Architect — weekly scan (Saturday 09:00 IST)
        self._scheduler.add_job(
            func=self._safe(self._wealth_architect_scan),
            trigger=CronTrigger(hour=9, minute=0, day_of_week="sat", timezone=_TZ_IST),
            id="wealth_architect_weekly_scan",
            name="Hermes Wealth Architect — weekly SIP candidate scan (Saturday 9 AM IST)",
            replace_existing=True,
        )

        # Phase 8: Quarterly hyperparameter optimization (1st of Q, 3 AM IST)
        # Q1: Jan 1, Q2: Apr 1, Q3: Jul 1, Q4: Oct 1
        for quarter_month, quarter_name in [(1, "Q1"), (4, "Q2"), (7, "Q3"), (10, "Q4")]:
            self._scheduler.add_job(
                func=self._safe(lambda q=quarter_name: self._quarterly_hpo(q)),
                trigger=CronTrigger(
                    hour=3, minute=0, day="1", month=quarter_month, timezone=_TZ_IST
                ),
                id=f"quarterly_hpo_{quarter_name}",
                name=f"Phase 8 — {quarter_name} hyperparameter optimization",
                replace_existing=True,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe(self, fn: Any) -> Any:
        """
        Wrap a job function with execution tracking and failure alerting.

        QUICK WIN 3: Enhanced with:
        - Start/end time tracking (visibility into job duration)
        - Consecutive failure detection (alert on 2+ consecutive failures)
        - Detailed error logging with duration
        - Telegram alerting with retry guidance
        """
        import time

        def wrapper(*args: Any, **kwargs: Any) -> None:
            job_name = getattr(fn, "__name__", "unknown")
            start_time = time.time()

            # Initialize failure tracking for this job if not exists
            if job_name not in self._job_failures:
                self._job_failures[job_name] = []

            self._job_last_start[job_name] = start_time

            try:
                fn(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log successful execution with duration
                log.info(
                    "scheduler_job_executed",
                    job=job_name,
                    duration_sec=round(elapsed, 2),
                )

                # Clear failure history on success
                self._job_failures[job_name] = []

            except Exception as exc:
                elapsed = time.time() - start_time

                # Record failure timestamp
                self._job_failures[job_name].append(start_time)

                # Clean up failures older than 1 hour
                one_hour_ago = start_time - 3600
                self._job_failures[job_name] = [
                    ts for ts in self._job_failures[job_name] if ts > one_hour_ago
                ]

                # Determine if this is consecutive failure
                consecutive_failures = len(self._job_failures[job_name])

                log.error(
                    "scheduler_job_failed",
                    job=job_name,
                    duration_sec=round(elapsed, 2),
                    error=str(exc),
                    consecutive_failures=consecutive_failures,
                )

                # Alert on Telegram if 2+ consecutive failures
                if consecutive_failures >= 2:
                    try:
                        from monitoring.alerts import TelegramAlerter

                        TelegramAlerter().alert_system_error(
                            module=job_name,
                            error_msg=f"🚨 Job failed {consecutive_failures} consecutive times: {exc!s} (after {elapsed:.1f}s)",
                        )
                    except Exception as tg_err:
                        log.error("telegram_alerter_failed", error=str(tg_err))
                else:
                    # Single failure: log but don't spam Telegram yet
                    log.warning(
                        "scheduler_job_single_failure",
                        job=job_name,
                        elapsed_sec=round(elapsed, 2),
                    )

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

    def _ab_test_reporting(self) -> None:
        """Generate and send A/B test weekly comparison report."""
        try:
            from monitoring.ab_test_reporter import ABTestReporter
            from monitoring.telegram_notifier import TelegramNotifier
            from orchestrator.ab_tester import ABTestOrchestrator

            # Generate A/B test report
            ab_tester = ABTestOrchestrator()
            reporter = ABTestReporter(ab_tester=ab_tester)
            report = reporter.generate_weekly_report(lookback_days=7)

            log.info("ab_test_report_generated")

            # Send via Telegram
            notifier = TelegramNotifier()
            notifier.send_message(report)

        except Exception as exc:
            log.error("ab_test_reporting_failed", error=str(exc))

    def _monthly_ensemble_retrain(self) -> None:
        """Phase 8: Monthly walk-forward ensemble retraining job."""
        try:
            from datetime import datetime, timedelta

            from monitoring.telegram_notifier import TelegramNotifier
            from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer

            log.info("monthly_ensemble_retrain_started")

            # Use 5-year lookback for training
            now = datetime.utcnow()
            train_end = now - timedelta(days=5)  # Purge 5 days
            train_start = train_end - timedelta(days=365 * 5)

            trainer = WalkForwardEnsembleTrainer(
                train_months=60,  # 5 years training
                val_months=12,  # 1 year validation
                test_months=6,  # 6 months test
                experiment_name="ensemble_monthly_retrain",
            )

            # Run walk-forward training
            report = trainer.run_walk_forward(
                symbols=["INFY", "TCS", "RELIANCE", "HDFCBANK", "ICICIBANK"],
                date_range=(train_start.strftime("%Y-%m-%d"), train_end.strftime("%Y-%m-%d")),
            )

            log.info(
                "monthly_ensemble_retrain_complete",
                auc=report.aggregate_auc,
                folds=report.total_folds,
                drift_detected=sum(1 for r in report.fold_results if r.drift_detected),
            )

            # Alert with results
            notifier = TelegramNotifier()
            notifier.send_message(
                f"✅ Ensemble Monthly Retrain Complete\n"
                f"Aggregate AUC: {report.aggregate_auc:.4f}\n"
                f"Folds: {report.total_folds}\n"
                f"Drift Events: {sum(1 for r in report.fold_results if r.drift_detected)}"
            )

        except Exception as exc:
            log.error("monthly_ensemble_retrain_failed", error=str(exc))

    def _weekly_concept_drift_check(self) -> None:
        """Phase 8: Weekly concept drift detection (Monday 6 AM IST)."""
        try:
            from monitoring.telegram_notifier import TelegramNotifier
            from signals.training.concept_drift import ConceptDriftDetector

            log.info("weekly_concept_drift_check_started")

            detector = ConceptDriftDetector(threshold=0.5)

            # Fetch latest data
            import pandas as pd

            from data.store import get_engine
            from signals.features import FEATURE_COLUMNS

            engine = get_engine()
            df = pd.read_sql(
                "SELECT * FROM ohlcv WHERE time >= NOW() - INTERVAL '30 days' ORDER BY time",
                engine,
                parse_dates=["time"],
                index_col="time",
            )

            if len(df) < 100:
                log.warning("concept_drift_check_skipped_insufficient_data")
                return

            # Check drift
            drift_result = detector.check(
                df[FEATURE_COLUMNS] if FEATURE_COLUMNS[0] in df.columns else df
            )

            drifted_features = [f for f, p in drift_result.items() if p < 0.05]

            log.info(
                "concept_drift_check_complete",
                drifted_features_count=len(drifted_features),
                drifted_features=drifted_features,
            )

            if drifted_features:
                log.warning("concept_drift_detected", features=drifted_features)

                notifier = TelegramNotifier()
                notifier.send_message(
                    f"⚠️ Concept Drift Detected\n"
                    f"Features: {', '.join(drifted_features[:5])}\n"
                    f"Emergency retraining may be needed."
                )

                # Optionally trigger emergency retrain
                self._system.retrain_check()

        except Exception as exc:
            log.error("weekly_concept_drift_check_failed", error=str(exc))

    def _wealth_architect_scan(self) -> None:
        """Hermes Wealth Architect weekly scan — delegates to TradingSystem."""
        self._system.run_wealth_scan()

    def _quarterly_hpo(self, quarter: str) -> None:
        """Phase 8: Quarterly hyperparameter optimization."""
        try:
            from datetime import datetime, timedelta

            from monitoring.telegram_notifier import TelegramNotifier
            from signals.training.hyperparameter_optimizer import BayesianHyperparameterOptimizer

            log.info("quarterly_hpo_started", quarter=quarter)

            # Run Bayesian optimization
            optimizer = BayesianHyperparameterOptimizer(n_trials=20, warm_start=True)

            # Use last 2 years of data for HPO
            import pandas as pd

            from data.store import get_engine
            from signals.features import FEATURE_COLUMNS

            engine = get_engine()
            now = datetime.utcnow()
            start_date = now - timedelta(days=730)

            df = pd.read_sql(
                "SELECT * FROM ohlcv WHERE time >= :start_date ORDER BY time",
                engine,
                params={"start_date": start_date},
                parse_dates=["time"],
                index_col="time",
            )

            if len(df) < 1000:
                log.warning("quarterly_hpo_skipped_insufficient_data")
                return

            _, history = optimizer.optimize(
                X=df[FEATURE_COLUMNS],
                y=df.get("label", pd.Series([0] * len(df))),
            )

            log.info(
                "quarterly_hpo_complete",
                quarter=quarter,
                best_score=history[-1]["score"],
                n_trials=len(history),
            )

            notifier = TelegramNotifier()
            notifier.send_message(
                f"📊 {quarter} Hyperparameter Optimization Complete\n"
                f"Best Score: {history[-1]['score']:.4f}\n"
                f"Trials: {len(history)}\n"
                f"New hyperparameters available in MLflow."
            )

        except Exception as exc:
            log.error("quarterly_hpo_failed", error=str(exc), quarter=quarter)
