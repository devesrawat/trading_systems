"""Unit tests for monitoring/ — TDD RED phase. Mocks Telegram API and MLflow."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
            alerter.send("Test")   # must not raise

    def test_send_retries_on_failure(self):
        with patch("monitoring.alerts.Bot") as mock_bot_cls, \
             patch("monitoring.alerts.time.sleep"):
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
        alerter.send("test")              # must not raise
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
            for i in range(20):
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
            for i in range(20):
                run = MagicMock()
                run.data.metrics = {"pnl_pct": 0.03, "signal_prob": 0.72}
                run.data.tags = {"outcome": "win"}
                runs.append(run)
            mock_client.search_runs.return_value = runs

            monitor = ModelDriftMonitor()
            score = monitor.compare_live_vs_backtest(window_trades=20)
        assert score < 0.5
