"""Unit tests for orchestrator/ — mocks all external dependencies."""
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from orchestrator.main import TradingSystem
from orchestrator.scheduler import TradingScheduler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_broker(paper: bool = True) -> MagicMock:
    b = MagicMock()
    b.is_paper = paper
    b.get_balance.return_value = 500_000.0
    b.refresh_auth.return_value = None
    b.cancel_order.return_value = True
    return b


def _mock_settings(**overrides) -> MagicMock:
    s = MagicMock()
    s.paper_trade_mode = True
    s.market_type = "equity"
    s.initial_capital = 500_000.0
    s.signal_threshold = 0.65
    s.crypto_signal_threshold = 0.65
    s.max_position_pct = 0.02
    s.crypto_max_position_pct = 0.01
    s.daily_dd_limit = 0.03
    s.weekly_dd_limit = 0.07
    s.mlflow_tracking_uri = "http://localhost:5000"
    s.coingecko_api_key = None
    s.crypto_min_volume_usd = 5_000_000
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def _make_system(market_type: str = "equity", **settings_overrides) -> TradingSystem:
    """Create a TradingSystem with all external dependencies mocked."""
    mock_settings = _mock_settings(market_type=market_type, **settings_overrides)

    mock_broker = _mock_broker()
    mock_cb = MagicMock()
    mock_cb.is_halted.return_value = False
    mock_cb.check.return_value = (True, None)
    mock_model = MagicMock()
    mock_model.is_healthy.return_value = True
    mock_registry = MagicMock()
    mock_registry.get_latest_model.return_value = mock_model
    mock_sentiment = MagicMock()
    mock_sentiment.run_daily.return_value = {}
    mock_sentiment.get_latest_score.return_value = 0.2
    mock_sizer = MagicMock()
    mock_sizer.size.return_value = 9_000.0
    mock_sizer.shares.return_value = 10
    mock_executor = MagicMock()
    mock_executor.place_market_order.return_value = "ORDER_001"
    mock_monitor = MagicMock()
    mock_monitor.get_drawdown.return_value = {"daily_dd": 0.0, "weekly_dd": 0.0}

    mock_equity_provider = MagicMock()
    mock_crypto_provider = MagicMock()

    with patch("orchestrator.main.settings", mock_settings), \
         patch("orchestrator.main.get_provider", return_value=mock_equity_provider), \
         patch("orchestrator.main.get_crypto_provider", return_value=mock_crypto_provider), \
         patch("orchestrator.main.get_broker_adapter", return_value=mock_broker), \
         patch("orchestrator.main.CircuitBreaker", return_value=mock_cb), \
         patch("orchestrator.main.PositionSizer", return_value=mock_sizer), \
         patch("orchestrator.main.OrderExecutor", return_value=mock_executor), \
         patch("orchestrator.main.TradeLogger", return_value=MagicMock()), \
         patch("orchestrator.main.PortfolioMonitor", return_value=mock_monitor), \
         patch("orchestrator.main.ModelRegistry", return_value=mock_registry), \
         patch("llm.pipeline.SentimentPipeline", return_value=mock_sentiment):
        system = TradingSystem(market_type=market_type)

    # Inject the pre-built mocks so tests can assert on them
    system._circuit_breaker = mock_cb
    system._sizer = mock_sizer
    system._executor = mock_executor
    system._monitor = mock_monitor
    system._registry = mock_registry
    system._broker = mock_broker
    if market_type != "crypto":
        system._sentiment = mock_sentiment
    return system


# ---------------------------------------------------------------------------
# TradingSystem — initialisation
# ---------------------------------------------------------------------------

class TestTradingSystemInit:
    def test_creates_without_error(self):
        system = _make_system()
        assert system is not None

    def test_paper_mode_via_broker(self):
        system = _make_system()
        assert system._broker.is_paper is True

    def test_equity_mode_has_equity_provider(self):
        system = _make_system(market_type="equity")
        assert system._equity_provider is not None
        assert system._crypto_provider is None

    def test_crypto_mode_has_crypto_provider(self):
        system = _make_system(market_type="crypto")
        assert system._crypto_provider is not None
        assert system._equity_provider is None

    def test_both_mode_has_both_providers(self):
        system = _make_system(market_type="both")
        assert system._equity_provider is not None
        assert system._crypto_provider is not None

    def test_equity_mode_has_sentiment(self):
        system = _make_system(market_type="equity")
        assert system._sentiment is not None

    def test_all_core_modules_initialised(self):
        system = _make_system()
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
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_load_model"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()
        system._circuit_breaker.reset_daily.assert_called_once()

    def test_loads_model_from_registry(self):
        system = _make_system()
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()
        # _load_model calls registry.get_latest_model
        system._registry.get_latest_model.assert_called()

    def test_runs_sentiment_for_equity_universe(self):
        system = _make_system(market_type="equity")
        system._equity_universe = ["RELIANCE", "TCS"]
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_load_model"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()
        system._sentiment.run_daily.assert_called_once()

    def test_pre_market_sentiment_error_does_not_crash(self):
        system = _make_system(market_type="equity")
        system._equity_universe = ["RELIANCE"]
        system._sentiment.run_daily.side_effect = Exception("API timeout")
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_load_model"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()   # must not raise

    def test_broker_auth_refresh_called(self):
        system = _make_system()
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_load_model"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()
        system._broker.refresh_auth.assert_called_once()

    def test_auth_failure_does_not_crash(self):
        system = _make_system()
        system._broker.refresh_auth.side_effect = Exception("Token expired")
        with patch.object(system, "_load_equity_universe"), \
             patch.object(system, "_load_model"), \
             patch.object(system, "_send_alert"):
            system.pre_market_setup()   # must not raise


# ---------------------------------------------------------------------------
# trading_loop
# ---------------------------------------------------------------------------

class TestTradingLoop:
    def _make_feature_df(self) -> pd.DataFrame:
        """Minimal feature DataFrame the loop can operate on."""
        from signals.features import FEATURE_COLUMNS
        data = {col: [0.0] * 5 for col in FEATURE_COLUMNS}
        data["ema_50"] = [1500.0] * 5
        data["realized_vol_20"] = [0.2] * 5
        data["close"] = [1500.0] * 5  # execution price pass-through
        return pd.DataFrame(data)

    def test_halted_circuit_breaker_skips_all_orders(self):
        system = _make_system()
        system._circuit_breaker.is_halted.return_value = True
        system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_drawdown_breach_halts_circuit_breaker(self):
        system = _make_system()
        system._circuit_breaker.is_halted.return_value = False
        system._monitor.get_drawdown.return_value = {"daily_dd": 0.04, "weekly_dd": 0.0}
        system.trading_loop()
        system._circuit_breaker.halt.assert_called_once()

    def test_unhealthy_model_skips_equity_loop(self):
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = False
        system._equity_universe = ["RELIANCE"]
        with patch.object(system, "_fetch_equity_features", return_value={}):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_no_model_skips_equity_loop(self):
        system = _make_system(market_type="equity")
        system._model = None
        system._equity_universe = ["RELIANCE"]
        system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_signal_below_threshold_no_order(self):
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._model.predict.return_value = pd.Series([0.40])
        system._equity_universe = ["RELIANCE"]

        with patch.object(system, "_fetch_equity_features",
                          return_value={"RELIANCE": self._make_feature_df()}), \
             patch.object(system, "_logger"):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()

    def test_signal_above_threshold_places_order(self):
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._model.predict.return_value = pd.Series([0.80])
        system._equity_universe = ["RELIANCE"]

        with patch.object(system, "_fetch_equity_features",
                          return_value={"RELIANCE": self._make_feature_df()}), \
             patch.object(system._logger, "log_signal"):
            system.trading_loop()
        system._executor.place_market_order.assert_called_once()

    def test_open_position_skipped(self):
        """Symbol already in _open_positions must not trigger another order."""
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._model.predict.return_value = pd.Series([0.90])
        system._equity_universe = ["RELIANCE"]
        system._open_positions.add("RELIANCE")

        with patch.object(system, "_fetch_equity_features",
                          return_value={"RELIANCE": self._make_feature_df()}):
            system.trading_loop()
        system._executor.place_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# post_market_summary
# ---------------------------------------------------------------------------

class TestPostMarketSummary:
    def test_calls_daily_summary(self):
        system = _make_system()
        with patch.object(system, "_check_model_drift"):
            system.post_market_summary()
        system._logger.daily_summary.assert_called_once()

    def test_checks_model_drift(self):
        system = _make_system()
        with patch.object(system, "_check_model_drift") as mock_drift:
            system.post_market_summary()
        mock_drift.assert_called_once()

    def test_clears_open_positions(self):
        system = _make_system()
        system._open_positions.add("RELIANCE")
        with patch.object(system, "_check_model_drift"):
            system.post_market_summary()
        assert len(system._open_positions) == 0

    def test_post_market_error_does_not_crash(self):
        system = _make_system()
        system._logger.daily_summary.side_effect = Exception("DB error")
        system.post_market_summary()   # must not raise


# ---------------------------------------------------------------------------
# TradingScheduler (via orchestrator.scheduler)
# ---------------------------------------------------------------------------

class TestTradingScheduler:
    def test_scheduler_creates_without_error(self):
        with patch("orchestrator.scheduler.BackgroundScheduler"):
            system = _make_system()
            sched = TradingScheduler(system)
            assert sched is not None

    def test_equity_registers_at_least_four_jobs(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system, market_type="equity")
            sched.start()
        assert mock_sched.add_job.call_count >= 4

    def test_equity_scheduler_uses_kolkata_timezone(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system, market_type="equity")
            sched.start()

        all_kwargs = " ".join(str(c) for c in mock_sched.add_job.call_args_list)
        assert "Kolkata" in all_kwargs or "Asia" in all_kwargs

    def test_crypto_registers_crypto_jobs(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system(market_type="crypto")
            sched = TradingScheduler(system, market_type="crypto")
            sched.start()

        ids = [c[1].get("id") for c in mock_sched.add_job.call_args_list]
        assert "crypto_trading_loop" in ids

    def test_stop_shuts_down_scheduler(self):
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system)
            sched.start()
            sched.stop()
        mock_sched.shutdown.assert_called_once()

    def test_equity_registers_health_check_job(self):
        """Equity mode must register a health check job for stale heartbeat detection."""
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system, market_type="equity")
            sched.start()
        ids = [c[1].get("id") for c in mock_sched.add_job.call_args_list]
        assert "equity_health_check" in ids

    def test_equity_registers_fii_dii_job(self):
        """Equity mode must register a FII/DII fetch job at 16:30 IST."""
        with patch("orchestrator.scheduler.BackgroundScheduler") as mock_cls:
            mock_sched = MagicMock()
            mock_cls.return_value = mock_sched
            system = _make_system()
            sched = TradingScheduler(system, market_type="equity")
            sched.start()
        ids = [c[1].get("id") for c in mock_sched.add_job.call_args_list]
        assert "equity_fii_dii_fetch" in ids


# ---------------------------------------------------------------------------
# Kill switch and both-mode regime fix
# ---------------------------------------------------------------------------

class TestKillSwitchAndRegimeFix:
    """Tests for Redis kill switch and both-mode regime suppression fix."""

    def _make_feature_df(self) -> pd.DataFrame:
        from signals.features import FEATURE_COLUMNS
        data = {col: [0.0] * 5 for col in FEATURE_COLUMNS}
        data["close"] = [1500.0] * 5
        data["realized_vol_20"] = [0.2] * 5
        return pd.DataFrame(data)

    def test_kill_switch_active_skips_trading(self):
        """When Redis kill switch key is '1', trading_loop must return without executing."""
        system = _make_system()
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"1"

        with patch("orchestrator.main.get_redis", return_value=mock_redis), \
             patch("orchestrator.main.HealthMonitor"):
            system.trading_loop()

        system._executor.place_market_order.assert_not_called()

    def test_kill_switch_inactive_allows_trading(self):
        """When kill switch key is not '1', trading should proceed normally."""
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._model.predict.return_value = pd.Series([0.85])
        system._equity_universe = ["RELIANCE"]

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # not set

        with patch("orchestrator.main.get_redis", return_value=mock_redis), \
             patch("orchestrator.main.HealthMonitor"), \
             patch.object(system, "_fetch_equity_features",
                          return_value={"RELIANCE": self._make_feature_df()}), \
             patch.object(system._logger, "log_signal"):
            system.trading_loop()

        system._executor.place_market_order.assert_called_once()

    def test_heartbeat_written_even_when_circuit_halted(self):
        """Heartbeat must be written in finally so it fires even on early returns."""
        system = _make_system()
        system._circuit_breaker.is_halted.return_value = True

        mock_health = MagicMock()
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # kill switch off

        with patch("orchestrator.main.HealthMonitor", return_value=mock_health), \
             patch("orchestrator.main.get_redis", return_value=mock_redis):
            system.trading_loop()

        mock_health.write_heartbeat.assert_called_once()

    def test_both_mode_choppy_regime_runs_crypto_not_equity(self):
        """In 'both' mode, a choppy regime must suppress equity but still run crypto."""
        from signals.regime import RegimeState
        system = _make_system(market_type="both")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._crypto_model = None  # no crypto model — crypto cycle skips model check
        system._crypto_universe = ["BTC"]

        # Simulate CHOPPY regime
        mock_regime = MagicMock()
        mock_regime.state = RegimeState.CHOPPY
        system._current_regime = mock_regime

        mock_regime_detector = MagicMock()
        mock_regime_detector.should_suppress_new_entries.return_value = True
        system._regime_detector = mock_regime_detector

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # kill switch off

        run_equity_called = []
        run_crypto_called = []
        original_equity = system._run_equity_cycle
        original_crypto = system._run_crypto_cycle

        def track_equity():
            run_equity_called.append(True)
        def track_crypto():
            run_crypto_called.append(True)

        with patch("orchestrator.main.get_redis", return_value=mock_redis), \
             patch("orchestrator.main.HealthMonitor"), \
             patch.object(system, "_update_regime"), \
             patch.object(system, "_run_equity_cycle", side_effect=track_equity), \
             patch.object(system, "_run_crypto_cycle", side_effect=track_crypto):
            system.trading_loop()

        assert len(run_equity_called) == 0, "Equity cycle should be suppressed in choppy regime"
        assert len(run_crypto_called) == 1, "Crypto cycle must still run in both-mode choppy regime"

    def test_crypto_signal_skipped_when_no_model(self):
        """_execute_crypto_signal must return immediately if no crypto model is available."""
        system = _make_system(market_type="crypto")
        system._crypto_model = None  # no model loaded

        system._execute_crypto_signal("BTC")  # must not raise or place any order

        system._executor.place_market_order.assert_not_called()

    def test_buy_alert_sent_on_equity_buy_signal(self):
        """When an equity BUY is executed, a Telegram alert with SHAP must be sent."""
        system = _make_system(market_type="equity")
        system._model = MagicMock()
        system._model.is_healthy.return_value = True
        system._model.predict.return_value = pd.Series([0.85])
        system._model.explain.return_value = {"rsi_14": 0.15, "ema_ratio": -0.08, "vol": 0.05}
        system._equity_universe = ["RELIANCE"]

        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with patch("orchestrator.main.get_redis", return_value=mock_redis), \
             patch("orchestrator.main.HealthMonitor"), \
             patch.object(system, "_fetch_equity_features",
                          return_value={"RELIANCE": self._make_feature_df()}), \
             patch.object(system._logger, "log_signal"), \
             patch.object(system, "_send_alert") as mock_alert:
            system.trading_loop()

        # Should have been called at least once with a BUY message
        calls = [str(c) for c in mock_alert.call_args_list]
        assert any("BUY" in c or "SHAP" in c for c in calls), \
            f"Expected a BUY alert with SHAP, got: {calls}"
