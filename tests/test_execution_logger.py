"""Unit tests for execution/logger.py — TDD RED phase. Mocks MLflow."""

from datetime import date
from unittest.mock import MagicMock, patch

from execution.logger import TradeLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_logger() -> TradeLogger:
    with patch("execution.logger.mlflow"):
        return TradeLogger(experiment_name="test_experiment")


_SAMPLE_FEATURES = {
    "rsi_14": 55.2,
    "macd": 0.3,
    "atr_pct": 0.012,
    "vol_regime": 0,
    "bb_position": 0.65,
}


# ---------------------------------------------------------------------------
# log_signal
# ---------------------------------------------------------------------------


class TestLogSignal:
    def test_returns_run_id_string(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_abc123"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run

            logger = TradeLogger()
            run_id = logger.log_signal(
                symbol="RELIANCE",
                features_dict=_SAMPLE_FEATURES,
                signal_prob=0.72,
                action_taken="BUY",
            )
        assert isinstance(run_id, str)

    def test_logs_symbol_as_param(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_xyz"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run

            logger = TradeLogger()
            logger.log_signal("TCS", _SAMPLE_FEATURES, 0.68, "BUY")

            logged_params = {}
            for c in mock_mlflow.log_params.call_args_list:
                logged_params.update(c[0][0])
            assert "symbol" in logged_params or any(
                "symbol" in str(c) for c in mock_mlflow.mock_calls
            )

    def test_logs_signal_prob_as_metric(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_xyz"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run

            logger = TradeLogger()
            logger.log_signal("TCS", _SAMPLE_FEATURES, 0.68, "BUY")

            all_metrics = {}
            for c in mock_mlflow.log_metrics.call_args_list:
                all_metrics.update(c[0][0])
            assert "signal_prob" in all_metrics or any(
                "signal_prob" in str(c) for c in mock_mlflow.mock_calls
            )

    def test_logs_all_feature_values(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_xyz"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run

            logger = TradeLogger()
            logger.log_signal("INFY", _SAMPLE_FEATURES, 0.70, "BUY")

            all_logged = " ".join(str(c) for c in mock_mlflow.mock_calls)
            for key in _SAMPLE_FEATURES:
                assert key in all_logged

    def test_no_action_logs_skip(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_skip"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
            mock_mlflow.active_run.return_value = mock_run

            logger = TradeLogger()
            run_id = logger.log_signal("WIPRO", _SAMPLE_FEATURES, 0.45, action_taken="SKIP")

        assert run_id is not None


# ---------------------------------------------------------------------------
# log_outcome
# ---------------------------------------------------------------------------


class TestLogOutcome:
    def test_updates_run_with_exit_metrics(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            logger = TradeLogger()
            logger.log_outcome(
                run_id="run_abc123",
                exit_price=2650.0,
                exit_date=date(2024, 2, 10),
                pnl_pct=0.06,
            )
            mock_mlflow.log_metrics.assert_called()
            metrics_logged = mock_mlflow.log_metrics.call_args[0][0]
            assert "pnl_pct" in metrics_logged

    def test_outcome_logs_exit_price(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            logger = TradeLogger()
            logger.log_outcome("run_abc", 3000.0, date(2024, 2, 5), 0.03)
            all_calls = " ".join(str(c) for c in mock_mlflow.mock_calls)
            assert "exit_price" in all_calls or "3000" in all_calls


# ---------------------------------------------------------------------------
# log_circuit_breaker_event
# ---------------------------------------------------------------------------


class TestLogCircuitBreaker:
    def test_logs_circuit_event(self):
        with patch("execution.logger.mlflow") as mock_mlflow:
            mock_run = MagicMock()
            mock_run.info.run_id = "run_cb"
            mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
            mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

            logger = TradeLogger()
            logger.log_circuit_breaker_event(
                reason="daily drawdown exceeded 3%",
                capital_at_halt=485_000.0,
            )
            assert mock_mlflow.start_run.called or mock_mlflow.log_params.called


# ---------------------------------------------------------------------------
# daily_summary
# ---------------------------------------------------------------------------


class TestDailySummary:
    def test_returns_dict(self):
        with patch("execution.logger.mlflow"), patch("execution.logger.get_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

            logger = TradeLogger()
            result = logger.daily_summary()

        assert isinstance(result, dict)
        assert "trades_today" in result
        assert "win_rate" in result
        assert "pnl" in result

    def test_no_trades_returns_zero_metrics(self):
        with patch("execution.logger.mlflow"), patch("execution.logger.get_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchall.return_value = []
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

            logger = TradeLogger()
            result = logger.daily_summary()

        assert result["trades_today"] == 0
        assert result["win_rate"] == 0.0
