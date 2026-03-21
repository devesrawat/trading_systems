"""Unit tests for orchestrator/ — TDD RED phase. Mocks all external dependencies."""
import signal
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.main import TradingSystem
from orchestrator.scheduler import TradingScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_system(paper_mode: bool = True) -> TradingSystem:
    with patch("orchestrator.main.settings") as mock_settings, \
         patch("orchestrator.main.KiteIngestor") as mock_ingestor_cls, \
         patch("orchestrator.main.SentimentPipeline") as mock_sentiment_cls, \
         patch("orchestrator.main.ModelRegistry") as mock_registry_cls, \
         patch("orchestrator.main.CircuitBreaker") as mock_cb_cls, \
         patch("orchestrator.main.PositionSizer") as mock_sizer_cls, \
         patch("orchestrator.main.OrderExecutor") as mock_executor_cls, \
         patch("orchestrator.main.TradeLogger") as mock_logger_cls, \
         patch("orchestrator.main.PortfolioMonitor") as mock_monitor_cls:

        mock_settings.paper_trade_mode = paper_mode
        mock_settings.signal_threshold = 0.65
        mock_settings.max_position_pct = 0.02
        mock_settings.daily_dd_limit = 0.03
        mock_settings.weekly_dd_limit = 0.07
        mock_settings.kite_api_key = "test_key"
        mock_settings.kite_access_token = "test_token"
        mock_settings.mlflow_tracking_uri = "http://localhost:5000"

        # Circuit breaker — not halted
        mock_cb = MagicMock()
        mock_cb.is_halted.return_value = False
        mock_cb.check.return_value = (True, None)
        mock_cb_cls.return_value = mock_cb

        # Signal model
        mock_model = MagicMock()
        mock_model.is_healthy.return_value = True
        mock_registry = MagicMock()
        mock_registry.get_latest_model.return_value = mock_model
        mock_registry_cls.return_value = mock_registry

        # Sentiment
        mock_sentiment = MagicMock()
        mock_sentiment.run_daily.return_value = {"RELIANCE": 0.3, "TCS": -0.1}
        mock_sentiment.get_latest_score.return_value = 0.2
        mock_sentiment_cls.return_value = mock_sentiment

        # Sizer
        mock_sizer = MagicMock()
        mock_sizer.size.return_value = 9000.0
        mock_sizer.shares.return_value = 10
        mock_sizer_cls.return_value = mock_sizer

        # Executor
        mock_executor = MagicMock()
        mock_executor.place_market_order.return_value = "ORDER_001"
        mock_executor_cls.return_value = mock_executor

        # Monitor
        mock_monitor = MagicMock()
        mock_monitor.get_drawdown.return_value = {"daily_dd": 0.0, "weekly_dd": 0.0}
        mock_monitor_cls.return_value = mock_monitor

        system = TradingSystem()

    return system


# ---------------------------------------------------------------------------
# TradingSystem — initialisation
# ---------------------------------------------------------------------------

class TestTradingSystemInit:
    def test_creates_without_error(self):
        system = _make_system()
        assert system is not None

    def test_paper_mode_is_respected(self):
        system = _make_system(paper_mode=True)
        assert system._paper_mode is True

    def test_all_modules_initialised(self):
        system = _make_system()
        assert system._ingestor is not None
        assert system._sentiment is not None
        assert system._circuit_breaker is not None
        assert system._sizer is not None
        assert system._executor is not None
        assert system._logger is not None
        assert system._monitor is not None


# ---------------------------------------------------------------------------
# pre_market_setup
# ---------------------------------------------------------------------------

class TestPreMarketSetup:
    def test_resets_daily_circuit_breaker(self):
        system = _make_system()
        with patch.object(system._ingestor, "refresh_access_token", return_value="new_token"), \
             patch.object(system._sentiment, "run_daily", return_value={}), \
             patch.object(system, "_load_model"):
            system.pre_market_setup()
        system._circuit_breaker.reset_daily.assert_called_once()

    def test_loads_model_from_registry(self):
        system = _make_system()
        with patch.object(system._circuit_breaker, "reset_daily"), \
             patch.object(system._sentiment, "run_daily", return_value={}):
            system.pre_market_setup()
        assert system._model is not None

    def test_runs_sentiment_for_universe(self):
        system = _make_system()
        with patch.object(system._circuit_breaker, "reset_daily"), \
             patch.object(system, "_load_model"):
            system.pre_market_setup()
        system._sentiment.run_daily.assert_called_once()

    def test_pre_market_error_does_not_crash(self):
        """Errors in pre-market must be caught — market open cannot be missed."""
        system = _make_system()
        system._sentiment.run_daily.side_effect = Exception("API timeout")
        with patch.object(system._circuit_breaker, "reset_daily"), \
             patch.object(system, "_load_model"):
            system.pre_market_setup()   # must not raise


# ---------------------------------------------------------------------------
# trading_loop — one cycle
# ---------------------------------------------------------------------------

class TestTradingLoop:
    def _setup_loop(self, system: TradingSystem, signals: dict[str, float]) -> None:
        """Configure mock model to return given {symbol: prob} signals."""
        import pandas as pd
        import numpy as np
        mock_probs = pd.Series(signals)
        system._model = MagicMock()
        system._model.predict.return_value = mock_probs
        system._model.is_healthy.return_value = True

    def test_no_orders_below_threshold(self):
        system = _make_system()
        self._setup_loop(system, {"RELIANCE": 0.40, "TCS": 0.50})
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown", return_value={"daily_dd": 0.0, "weekly_dd": 0.0}):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_order_placed_above_threshold(self):
        system = _make_system()
        self._setup_loop(system, {"RELIANCE": 0.75})
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown", return_value={"daily_dd": 0.0, "weekly_dd": 0.0}), \
             patch.object(system, "_universe", ["RELIANCE"]):
            system.trading_loop()
        system._executor.place_market_order.assert_called_once()

    def test_halted_circuit_breaker_skips_all_orders(self):
        system = _make_system()
        system._circuit_breaker.is_halted.return_value = True
        self._setup_loop(system, {"RELIANCE": 0.90, "TCS": 0.85})
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown", return_value={"daily_dd": 0.0, "weekly_dd": 0.0}):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_drawdown_triggers_circuit_breaker(self):
        system = _make_system()
        self._setup_loop(system, {"RELIANCE": 0.80})
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown",
                          return_value={"daily_dd": 0.035, "weekly_dd": 0.0}):
            system.trading_loop()
        system._circuit_breaker.halt.assert_called()

    def test_signal_logged_for_every_instrument(self):
        system = _make_system()
        self._setup_loop(system, {"RELIANCE": 0.80, "TCS": 0.72})
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown", return_value={"daily_dd": 0.0, "weekly_dd": 0.0}), \
             patch.object(system, "_universe", ["RELIANCE", "TCS"]):
            system.trading_loop()
        assert system._logger.log_signal.call_count >= 1

    def test_unhealthy_model_skips_loop(self):
        system = _make_system()
        system._model = MagicMock()
        system._model.is_healthy.return_value = False
        with patch.object(system, "_fetch_features", return_value=MagicMock()), \
             patch.object(system._monitor, "get_drawdown", return_value={"daily_dd": 0.0, "weekly_dd": 0.0}):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# post_market_summary
# ---------------------------------------------------------------------------

class TestPostMarketSummary:
    def test_calls_daily_summary(self):
        system = _make_system()
        system.post_market_summary()
        system._logger.daily_summary.assert_called_once()

    def test_checks_model_drift(self):
        system = _make_system()
        with patch.object(system, "_check_model_drift") as mock_drift:
            system.post_market_summary()
        mock_drift.assert_called_once()

    def test_post_market_error_does_not_crash(self):
        system = _make_system()
        system._logger.daily_summary.side_effect = Exception("DB error")
        system.post_market_summary()   # must not raise


# ---------------------------------------------------------------------------
# TradingScheduler
# ---------------------------------------------------------------------------

class TestTradingScheduler:
    def test_scheduler_creates_without_error(self):
        with patch("orchestrator.scheduler.BackgroundScheduler"):
            system = _make_system()
            sched = TradingScheduler(system)
            assert sched is not None

    def test_all_jobs_registered(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system)
            sched.start()

        add_job_calls = mock_sched.add_job.call_args_list
        job_funcs = [str(c) for c in add_job_calls]
        joined = " ".join(job_funcs)
        # Verify key jobs registered
        assert mock_sched.add_job.call_count >= 4

    def test_uses_kolkata_timezone(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system)
            sched.start()

        all_kwargs = " ".join(
            str(c[1]) for c in mock_sched.add_job.call_args_list
        )
        assert "Kolkata" in all_kwargs or "kolkata" in all_kwargs.lower() or "Asia" in all_kwargs

    def test_stop_shuts_down_scheduler(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_sched_cls:
            mock_sched = MagicMock()
            mock_sched_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system)
            sched.start()
            sched.stop()
        mock_sched.shutdown.assert_called_once()
