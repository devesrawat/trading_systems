"""Unit tests for monitoring/ — TDD RED phase. Mocks Telegram API and MLflow."""

from unittest.mock import AsyncMock, MagicMock, patch

from monitoring.alerts import TelegramAlerter
from monitoring.mlflow_tracker import ModelDriftMonitor

# ---------------------------------------------------------------------------
# TelegramAlerter
# ---------------------------------------------------------------------------


class TestTelegramAlerter:
    def test_send_calls_telegram_api(self):
        with patch("monitoring.alerts.Bot") as mock_bot_cls:
            mock_bot = AsyncMock()
            mock_bot_cls.return_value = mock_bot
            alerter = TelegramAlerter(bot_token="test_token", chat_id="12345")
            alerter.send("Test message")
        mock_bot.send_message.assert_called_once()

    def test_send_includes_message_text(self):
        with patch("monitoring.alerts.Bot") as mock_bot_cls:
            mock_bot = AsyncMock()
            mock_bot_cls.return_value = mock_bot
            alerter = TelegramAlerter(bot_token="test_token", chat_id="12345")
            alerter.send("Hello from trading system")
            call_kwargs = mock_bot.send_message.call_args[1]
            assert "Hello from trading system" in call_kwargs.get("text", "")

    def test_send_failure_does_not_raise(self):
        with patch("monitoring.alerts.Bot") as mock_bot_cls:
            mock_bot = AsyncMock()
            mock_bot.send_message.side_effect = Exception("Network error")
            mock_bot_cls.return_value = mock_bot
            alerter = TelegramAlerter(bot_token="test_token", chat_id="12345")
            alerter.send("Test")  # must not raise

    def test_send_retries_on_failure(self):
        with patch("monitoring.alerts.Bot") as mock_bot_cls, patch("monitoring.alerts.time.sleep"):
            mock_bot = AsyncMock()
            mock_bot.send_message.side_effect = [Exception("fail"), None]
            mock_bot_cls.return_value = mock_bot
            alerter = TelegramAlerter(bot_token="test_token", chat_id="12345", max_retries=2)
            alerter.send("Test")
        assert mock_bot.send_message.call_count >= 1


class TestAlertFormats:
    def _alerter(self) -> tuple[TelegramAlerter, AsyncMock]:
        with patch("monitoring.alerts.Bot") as mock_bot_cls:
            mock_bot = AsyncMock()
            mock_bot_cls.return_value = mock_bot
            alerter = TelegramAlerter(bot_token="test", chat_id="123")
        return alerter, mock_bot

    def test_circuit_breaker_alert_contains_reason(self):
        alerter, mock_bot = self._alerter()
        with patch.object(alerter, "send") as mock_send:
            alerter.alert_circuit_breaker(
                reason="daily drawdown exceeded 3%",
                dd_pct=-0.0321,
                capital=483_200.0,
            )
        msg = mock_send.call_args[0][0]
        assert "CIRCUIT" in msg.upper() or "circuit" in msg.lower()
        assert "3%" in msg or "drawdown" in msg.lower()

    def test_circuit_breaker_alert_contains_capital(self):
        alerter, mock_bot = self._alerter()
        with patch.object(alerter, "send") as mock_send:
            alerter.alert_circuit_breaker("test reason", -0.03, 483_200.0)
        msg = mock_send.call_args[0][0]
        assert "483" in msg or "₹" in msg

    def test_daily_summary_alert_contains_trades(self):
        alerter, _ = self._alerter()
        with patch.object(alerter, "send") as mock_send:
            alerter.alert_daily_summary(trades=12, pnl_pct=0.021, sharpe=1.8, win_rate=0.583)
        msg = mock_send.call_args[0][0]
        assert "12" in msg or "trade" in msg.lower()

    def test_model_drift_alert_contains_win_rates(self):
        alerter, _ = self._alerter()
        with patch.object(alerter, "send") as mock_send:
            alerter.alert_model_drift(current_win_rate=0.47, baseline_win_rate=0.58)
        msg = mock_send.call_args[0][0]
        assert "47" in msg or "drift" in msg.lower() or "win" in msg.lower()

    def test_system_error_alert_contains_module_name(self):
        alerter, _ = self._alerter()
        with patch.object(alerter, "send") as mock_send:
            alerter.alert_system_error(module="data.ingest", error_msg="Connection refused")
        msg = mock_send.call_args[0][0]
        assert "data.ingest" in msg or "error" in msg.lower()

    def test_no_token_skips_silently(self):
        """If bot_token is None, all alerts are silently no-ops."""
        alerter = TelegramAlerter(bot_token=None, chat_id=None)
        alerter.send("test")  # must not raise
        alerter.alert_circuit_breaker("test", 0.0, 100_000.0)


# ---------------------------------------------------------------------------
# ModelDriftMonitor
# ---------------------------------------------------------------------------


class TestModelDriftMonitor:
    def test_compare_returns_float(self):
        with patch("monitoring.mlflow_tracker.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client
            mock_client.search_runs.return_value = []
            monitor = ModelDriftMonitor()
            score = monitor.compare_live_vs_backtest(window_trades=20)
        assert isinstance(score, float)

    def test_no_runs_returns_zero(self):
        with patch("monitoring.mlflow_tracker.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client
            mock_client.search_runs.return_value = []
            monitor = ModelDriftMonitor()
            score = monitor.compare_live_vs_backtest(window_trades=20)
        assert score == 0.0

    def test_drift_detected_on_poor_win_rate(self):
        with patch("monitoring.mlflow_tracker.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client

            # Simulate 20 runs all with pnl_pct < 0 (all losses)
            runs = []
            for _i in range(20):
                run = MagicMock()
                run.data.metrics = {"pnl_pct": -0.01, "signal_prob": 0.70}
                run.data.tags = {"outcome": "loss"}
                runs.append(run)
            mock_client.search_runs.return_value = runs

            monitor = ModelDriftMonitor()
            score = monitor.compare_live_vs_backtest(window_trades=20)
        assert score > 0.0

    def test_high_win_rate_low_drift_score(self):
        with patch("monitoring.mlflow_tracker.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client

            runs = []
            for _i in range(20):
                run = MagicMock()
                run.data.metrics = {"pnl_pct": 0.03, "signal_prob": 0.72}
                run.data.tags = {"outcome": "win"}
                runs.append(run)
            mock_client.search_runs.return_value = runs

            monitor = ModelDriftMonitor()
            score = monitor.compare_live_vs_backtest(window_trades=20)
        assert score < 0.5


# ---------------------------------------------------------------------------
# Phase 6: Reporting Tests
# ---------------------------------------------------------------------------


class TestReports:
    """Test report formatting and content."""

    def test_daily_report_format(self):
        """Daily report should have correct format with all fields."""
        from datetime import datetime

        from monitoring.reporters import DailyMetrics, DailyReport

        metrics = DailyMetrics(
            date=datetime(2024, 1, 15),
            scans_completed=10,
            signals_generated=5,
            signals_executed=3,
            trades_entered=2,
            trades_closed=1,
            total_pnl=1500.0,
            total_pnl_pct=0.03,
            win_count=2,
            loss_count=0,
            win_rate=1.0,
            avg_win=750.0,
            avg_loss=0,
            profit_factor=float("inf"),
            max_intraday_dd=0.01,
            daily_dd=0.03,
        )

        report = DailyReport.generate(metrics)

        assert "DAILY REPORT" in report
        assert "2024-01-15" in report
        assert "10" in report  # scans
        assert "5" in report  # signals
        assert "3" in report  # executed
        assert "1500" in report  # pnl
        assert "3.00%" in report  # pnl_pct
        assert "100.0%" in report  # win_rate

    def test_weekly_report_format(self):
        """Weekly report should include strategy performance."""
        from datetime import datetime

        from monitoring.reporters import WeeklyMetrics, WeeklyReport

        metrics = WeeklyMetrics(
            week_start=datetime(2024, 1, 8),
            week_end=datetime(2024, 1, 14),
            days_traded=5,
            total_pnl=5000.0,
            total_pnl_pct=0.1,
            win_rate=0.75,
            profit_factor=3.0,
            sharpe_ratio=1.5,
            max_drawdown=0.02,
            strategy_performance={
                "vcp": {"pnl": 3000, "win_rate": 0.8, "trades": 10},
                "rs_breakout": {"pnl": 2000, "win_rate": 0.67, "trades": 6},
            },
            best_performer="vcp",
            worst_performer="rs_breakout",
            best_trade=1000.0,
            worst_trade=-500.0,
            average_trade=250.0,
        )

        report = WeeklyReport.generate(metrics)

        assert "WEEKLY REPORT" in report
        assert "2024-01-08" in report
        assert "vcp" in report
        assert "rs_breakout" in report
        assert "3000" in report

    def test_monthly_report_with_multibagger(self):
        """Monthly report should include multibagger candidates."""
        from datetime import datetime

        from monitoring.reporters import MonthlyMetrics, MonthlyReport

        metrics = MonthlyMetrics(
            month_start=datetime(2024, 1, 1),
            month_end=datetime(2024, 1, 31),
            days_traded=20,
            total_pnl=15000.0,
            total_pnl_pct=0.3,
            win_rate=0.65,
            profit_factor=2.5,
            sharpe_ratio=1.8,
            max_drawdown=0.05,
            calmar_ratio=6.0,
            strategy_rankings=[("vcp", 10000), ("rs_breakout", 5000)],
            multibagger_count=2,
            multibagger_candidates=["INFY", "TCS"],
        )

        report = MonthlyReport.generate(metrics)

        assert "MONTHLY REPORT" in report
        assert "Multibaggers: 2" in report
        assert "INFY" in report

    def test_portfolio_status_report(self):
        """Portfolio status should show holdings and exposure."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from monitoring.reporters import PortfolioSnapshot, PortfolioStatusReport

        ps = PortfolioSnapshot(
            timestamp=datetime(2024, 1, 15, 16, 0, 0, tzinfo=ZoneInfo("UTC")),
            total_value=520000,
            cash=20000,
            invested=500000,
            unrealized_pnl=20000,
            unrealized_pnl_pct=0.04,
            holdings={
                "INFY": {"qty": 10, "avg_price": 4000, "current_price": 4100, "pnl_pct": 0.025},
                "TCS": {"qty": 5, "avg_price": 5000, "current_price": 5100, "pnl_pct": 0.02},
            },
            sector_exposure={"IT": 0.8, "Finance": 0.2},
            correlation_matrix={"INFY": {"TCS": 0.85}},
            liquidity_score=0.95,
        )

        report = PortfolioStatusReport.generate(ps)

        assert "PORTFOLIO STATUS" in report
        assert "520" in report  # Has commas: ₹520,000
        assert "INFY" in report
        assert "TCS" in report

    def test_system_health_report(self):
        """System health should show status and metrics."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from monitoring.reporters import SystemHealthReport, SystemHealthSnapshot

        sh = SystemHealthSnapshot(
            timestamp=datetime(2024, 1, 15, 16, 0, 0, tzinfo=ZoneInfo("UTC")),
            uptime_hours=24.5,
            api_latency_ms=150.0,
            cache_hit_rate=0.85,
            error_rate=0.001,
            broker_connection_status="connected",
            db_connection_status="connected",
        )

        report = SystemHealthReport.generate(sh)

        assert "SYSTEM HEALTH" in report
        assert "✅" in report  # connected emoji
        assert "24.5" in report
        assert "150" in report


# ---------------------------------------------------------------------------
# Phase 6: Telegram Notifier Tests
# ---------------------------------------------------------------------------


class TestRateLimiter:
    """Test rate limiter logic."""

    def test_should_send_first_call(self):
        """First call should always be allowed."""
        from monitoring.telegram_notifier import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        assert limiter.should_send("test_key") is True

    def test_should_not_send_within_window(self):
        """Calls within window should be rejected."""
        from monitoring.telegram_notifier import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        limiter.should_send("test_key")  # First call
        assert limiter.should_send("test_key") is False

    def test_multiple_keys_independent(self):
        """Different keys should have independent rate limits."""
        from monitoring.telegram_notifier import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        assert limiter.should_send("key1") is True
        assert limiter.should_send("key2") is True
        assert limiter.should_send("key1") is False  # key1 still limited
        assert limiter.should_send("key2") is False  # key2 still limited

    def test_reset_single_key(self):
        """Reset should clear limit for a key."""
        from monitoring.telegram_notifier import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        limiter.should_send("test_key")
        limiter.reset("test_key")
        assert limiter.should_send("test_key") is True

    def test_reset_all_keys(self):
        """Reset with no key should clear all limits."""
        from monitoring.telegram_notifier import RateLimiter

        limiter = RateLimiter(window_seconds=60)
        limiter.should_send("key1")
        limiter.should_send("key2")
        limiter.reset()  # Reset all
        assert limiter.should_send("key1") is True
        assert limiter.should_send("key2") is True


class TestTelegramNotifier:
    """Test notifier formatting and rate limiting."""

    def test_signal_alert_format(self):
        """Signal alert should have correct format."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from monitoring.telegram_notifier import TelegramNotifier
        from signals.contracts import Direction, EntrySpec, RiskSpec, Signal, SignalType

        signal = Signal(
            signal_id="test_signal",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=ZoneInfo("UTC")),
            symbol="INFY",
            exchange="NSE",
            strategy_name="vcp",
            confidence=0.85,
            score=0.9,
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            entry=EntrySpec(
                entry_price=1800.0,
                stop_price=1750.0,
                target_price=1900.0,
            ),
            risk=RiskSpec(size_hint_pct=0.01),
        )

        notifier = TelegramNotifier()
        notifier.alert_signal(signal)  # Should not raise

    def test_trade_entry_alert(self):
        """Trade entry alert should format correctly."""
        from monitoring.telegram_notifier import TelegramNotifier

        notifier = TelegramNotifier()
        notifier.alert_trade_entry(
            symbol="INFY",
            direction="long",
            quantity=10,
            entry_price=1800.0,
            stop_price=1750.0,
            target_price=1900.0,
            risk_reward=2.0,
        )

    def test_batch_messages(self):
        """Batch functionality should queue and flush."""
        from monitoring.telegram_notifier import TelegramNotifier

        notifier = TelegramNotifier(batch_window_sec=1)

        notifier.add_to_batch("Message 1")
        notifier.add_to_batch("Message 2")

        # Messages should be queued
        assert len(notifier._pending_batch) == 2

        # Force flush should clear queue
        notifier.flush_batch(force=True)
        assert len(notifier._pending_batch) == 0


# ---------------------------------------------------------------------------
# Phase 6: Audit Tests
# ---------------------------------------------------------------------------


class TestAuditLogger:
    """Test audit logging functionality."""

    def test_log_signal_entry(self):
        """Signal logging should succeed without raising."""
        from datetime import datetime

        from audit.persistence import AuditLogger
        from audit.schema import SignalLogEntry

        entry = SignalLogEntry(
            signal_id="test_signal",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            strategy_name="vcp",
            direction="long",
            confidence=0.85,
            score=0.9,
            signal_type="scanner_hit",
            mode="paper",
        )

        AuditLogger.log_signal(entry)  # Should not raise

    def test_log_trade_entry(self):
        """Trade logging should succeed without raising."""
        from datetime import datetime

        from audit.persistence import AuditLogger
        from audit.schema import TradeLogEntry

        entry = TradeLogEntry(
            trade_id="test_trade",
            timestamp=datetime.utcnow(),
            symbol="INFY",
            direction="long",
            order_type="entry",
            quantity=10,
            price=1800.0,
            status="executed",
        )

        AuditLogger.log_trade(entry)  # Should not raise

    def test_log_risk_decision(self):
        """Risk decision logging should succeed without raising."""
        from datetime import datetime

        from audit.persistence import AuditLogger
        from audit.schema import RiskDecisionLog

        entry = RiskDecisionLog(
            decision_id="test_decision",
            timestamp=datetime.utcnow(),
            decision_type="position_rejected",
            reason="daily_dd_limit_exceeded",
            symbol="INFY",
        )

        AuditLogger.log_risk_decision(entry)  # Should not raise


class TestAuditQuery:
    """Test audit query functionality."""

    def test_get_signals_by_strategy_empty(self):
        """Query for non-existent strategy should return empty list."""
        from audit.query import AuditQuery

        results = AuditQuery.get_signals_by_strategy("nonexistent_strategy")
        assert isinstance(results, list)

    def test_get_trades_by_date_empty(self):
        """Query for trades in empty range should return empty list."""
        from datetime import datetime, timedelta

        from audit.query import AuditQuery

        start = datetime.utcnow() - timedelta(days=7)
        end = datetime.utcnow()
        results = AuditQuery.get_trades_by_date(start, end)
        assert isinstance(results, list)

    def test_get_signal_statistics_empty(self):
        """Statistics query on empty set should return empty dict."""
        from audit.query import AuditQuery

        stats = AuditQuery.get_signal_statistics()
        assert isinstance(stats, dict)
