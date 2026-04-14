"""
Tests for remaining roadmap items:
 - BinanceBrokerAdapter (paper mode, get_quote, factory routing)
 - PositionSizer correlation_penalty parameter
 - SignalRouter (A/B determinism, threshold boundary, Redis write, summary)
 - retrain_check / _auto_retrain_and_promote (high AUC promotes, low AUC skips)
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import pytest

# ===========================================================================
# BinanceBrokerAdapter
# ===========================================================================


class TestBinanceBrokerAdapter:
    def _make(self, capital: float = 1_000_000.0):
        from execution.broker import BinanceBrokerAdapter

        return BinanceBrokerAdapter(initial_capital=capital)

    def test_is_paper_true(self):
        assert self._make().is_paper is True

    def test_get_balance(self):
        assert self._make(500_000).get_balance() == 500_000.0

    def test_place_order_raises(self):
        adapter = self._make()
        with pytest.raises(NotImplementedError):
            adapter.place_order("BTCUSDT", "BUY", 1, "test")

    def test_cancel_order_paper_noop(self):
        assert self._make().cancel_order("fake_id") is True

    def test_get_order_status_paper(self):
        status = self._make().get_order_status("o123")
        assert status["status"] == "PAPER_COMPLETE"

    def test_get_quote_success(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"symbol": "BTCUSDT", "price": "67000.50"}
        mock_resp.raise_for_status.return_value = None

        adapter = self._make()
        with patch("requests.get", return_value=mock_resp) as mock_get:
            result = adapter.get_quote(["BTCUSDT"])

        mock_get.assert_called_once()
        assert "BTCUSDT" in result
        assert result["BTCUSDT"]["last_price"] == pytest.approx(67000.50)

    def test_get_quote_network_error_returns_empty(self):
        adapter = self._make()
        with patch("requests.get", side_effect=Exception("timeout")):
            result = adapter.get_quote(["ETHUSDT"])
        assert result == {}

    def test_get_quote_multiple_symbols(self):
        def _mock_get(url, params, timeout):
            sym = params["symbol"]
            prices = {"BTCUSDT": "67000.00", "ETHUSDT": "3500.00"}
            resp = MagicMock()
            resp.json.return_value = {"symbol": sym, "price": prices[sym]}
            resp.raise_for_status.return_value = None
            return resp

        adapter = self._make()
        with patch("requests.get", side_effect=_mock_get):
            result = adapter.get_quote(["BTCUSDT", "ETHUSDT"])

        assert len(result) == 2
        assert result["BTCUSDT"]["last_price"] == pytest.approx(67000.0)
        assert result["ETHUSDT"]["last_price"] == pytest.approx(3500.0)


class TestGetBrokerAdapterFactory:
    def test_binance_returns_binance_adapter(self):
        from execution.broker import BinanceBrokerAdapter, get_broker_adapter

        mock_settings = MagicMock()
        mock_settings.paper_trade_mode = False
        mock_settings.data_provider = "binance"
        mock_settings.initial_capital = 1_000_000.0

        with patch("config.settings.settings", mock_settings):
            adapter = get_broker_adapter()

        assert isinstance(adapter, BinanceBrokerAdapter)

    def test_zerodha_returns_kite_adapter(self):
        from execution.broker import KiteBrokerAdapter, get_broker_adapter

        mock_settings = MagicMock()
        mock_settings.paper_trade_mode = False
        mock_settings.data_provider = "kite"
        mock_settings.initial_capital = 1_000_000.0
        mock_settings.kite_api_key = "test_key"
        mock_settings.kite_access_token = "test_token"

        mock_kite_instance = MagicMock()
        mock_kite_cls = MagicMock(return_value=mock_kite_instance)
        with (
            patch("config.settings.settings", mock_settings),
            patch("kiteconnect.KiteConnect", mock_kite_cls, create=True),
        ):
            adapter = get_broker_adapter()

        assert isinstance(adapter, KiteBrokerAdapter)


# ===========================================================================
# PositionSizer correlation_penalty
# ===========================================================================


class TestPositionSizerCorrelationPenalty:
    def _sizer(self):
        from risk.sizer import PositionSizer

        return PositionSizer(total_capital=1_000_000.0, max_position_pct=0.02)

    def test_no_penalty_baseline(self):
        sizer = self._sizer()
        base = sizer.size(
            signal_probability=0.70, asset_volatility=0.15, current_capital=1_000_000.0
        )
        no_penalty = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=0.0,
        )
        assert base == pytest.approx(no_penalty)

    def test_30pct_penalty_reduces_size_by_30pct(self):
        sizer = self._sizer()
        base = sizer.size(
            signal_probability=0.70, asset_volatility=0.15, current_capital=1_000_000.0
        )
        penalised = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=0.30,
        )
        assert penalised == pytest.approx(base * 0.70, rel=1e-4)

    def test_full_penalty_produces_near_zero(self):
        sizer = self._sizer()
        result = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=1.0,
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_penalty_clamped_below_zero(self):
        sizer = self._sizer()
        base = sizer.size(
            signal_probability=0.70, asset_volatility=0.15, current_capital=1_000_000.0
        )
        negative = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=-0.5,
        )
        assert negative == pytest.approx(base)

    def test_penalty_clamped_above_one(self):
        sizer = self._sizer()
        result = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=1.5,
        )
        assert result == pytest.approx(0.0, abs=0.01)

    def test_50pct_penalty_halves_size(self):
        sizer = self._sizer()
        base = sizer.size(
            signal_probability=0.70, asset_volatility=0.15, current_capital=1_000_000.0
        )
        half = sizer.size(
            signal_probability=0.70,
            asset_volatility=0.15,
            current_capital=1_000_000.0,
            correlation_penalty=0.50,
        )
        assert half == pytest.approx(base * 0.50, rel=1e-4)


# ===========================================================================
# SignalRouter (A/B test routing)
# ===========================================================================


class TestSignalRouter:
    def _router(self, pct=0.20):
        from orchestrator.ab_router import SignalRouter

        return SignalRouter(challenger_pct=pct)

    def test_invalid_pct_raises(self):
        from orchestrator.ab_router import SignalRouter

        with pytest.raises(ValueError):
            SignalRouter(challenger_pct=1.5)
        with pytest.raises(ValueError):
            SignalRouter(challenger_pct=-0.1)

    def test_disabled_always_champion(self):
        router = self._router(pct=0.0)
        for sym in ["RELIANCE", "TCS", "INFY", "BTC", "ETH"]:
            assert router.route(sym, "2026-01-01") == "champion"

    def test_full_pct_always_challenger(self):
        router = self._router(pct=1.0)
        with patch("data.store.get_redis"):
            # With pct=1.0 all buckets < 100 → challenger
            for sym in ["RELIANCE", "TCS", "BTCUSDT"]:
                slot = router.route(sym, "2026-01-01")
                assert slot == "challenger"

    def test_deterministic_across_calls(self):
        router = self._router(pct=0.50)
        with patch("orchestrator.ab_router.SignalRouter._persist_slot"):
            slot1 = router.route("RELIANCE", "2026-04-14")
            slot2 = router.route("RELIANCE", "2026-04-14")
        assert slot1 == slot2

    def test_deterministic_uses_md5(self):
        """The bucket calculation must match raw MD5 logic."""
        from orchestrator.ab_router import SignalRouter

        symbol, date_str = "RELIANCE", "2026-04-14"
        raw = f"{symbol}:{date_str}".encode()
        digest = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        bucket = int(digest[:8], 16) % 100

        router = SignalRouter(challenger_pct=0.50)
        with patch("orchestrator.ab_router.SignalRouter._persist_slot"):
            slot = router.route(symbol, date_str)

        expected = "challenger" if bucket < 50 else "champion"
        assert slot == expected

    def test_threshold_boundary(self):
        """At pct=0, bucket 0 → champion; at pct=1, bucket 99 → challenger."""
        from orchestrator.ab_router import SignalRouter

        # Find a symbol+date combo that produces bucket 0
        for day in range(1, 400):
            d = f"2026-{day // 30 + 1:02d}-{day % 28 + 1:02d}"
            sym = "AAPL"
            raw = f"{sym}:{d}".encode()
            bucket = int(hashlib.md5(raw, usedforsecurity=False).hexdigest()[:8], 16) % 100
            if bucket == 0:
                router_1pct = SignalRouter(challenger_pct=0.01)
                with patch("orchestrator.ab_router.SignalRouter._persist_slot"):
                    slot = router_1pct.route(sym, d)
                assert slot == "challenger"
                break

    def test_redis_persist_called(self):
        router = self._router(pct=0.50)
        mock_redis = MagicMock()
        with patch("data.store.get_redis", return_value=mock_redis):
            router.route("RELIANCE", "2026-04-14")
        mock_redis.setex.assert_called_once()

    def test_redis_persist_failure_silent(self):
        router = self._router(pct=0.50)
        with patch("data.store.get_redis", side_effect=Exception("connection refused")):
            # Must not raise
            try:
                router.route("RELIANCE", "2026-04-14")
            except Exception as exc:
                pytest.fail(f"route() raised unexpectedly: {exc}")

    def test_get_slot_returns_persisted(self):
        router = self._router(pct=0.50)
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"champion"
        with patch("data.store.get_redis", return_value=mock_redis):
            slot = router.get_slot("RELIANCE", "2026-04-14")
        assert slot == "champion"

    def test_get_slot_returns_none_on_miss(self):
        router = self._router(pct=0.50)
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        with patch("data.store.get_redis", return_value=mock_redis):
            assert router.get_slot("RELIANCE", "2026-04-14") is None

    def test_record_outcome_writes_redis(self):
        router = self._router(pct=0.50)
        mock_redis = MagicMock()
        with patch("data.store.get_redis", return_value=mock_redis):
            router.record_outcome("RELIANCE", "2026-04-14", "champion", 0.023)
        mock_redis.setex.assert_called_once()
        key = mock_redis.setex.call_args[0][0]
        assert "RELIANCE" in key

    def test_different_symbols_can_get_different_slots(self):
        """Different symbols should not all map to the same slot (statistical check)."""
        router = self._router(pct=0.50)
        slots = set()
        with patch("orchestrator.ab_router.SignalRouter._persist_slot"):
            for i in range(20):
                sym = f"SYM{i:03d}"
                slots.add(router.route(sym, "2026-04-14"))
        # With 50% split and 20 symbols, expect both slots to appear
        assert len(slots) == 2


# ===========================================================================
# retrain_check / _auto_retrain_and_promote
# ===========================================================================


class TestAutoRetrain:
    def _orch(self):
        """Build a minimal stub TradingSystem with no real deps."""
        from orchestrator.main import TradingSystem

        orch = object.__new__(TradingSystem)
        orch._open_positions = set()
        orch._circuit_breaker = MagicMock()
        orch._registry = MagicMock()
        orch._model = None
        orch._challenger_model = None
        orch._ab_router = MagicMock()
        orch._market_type = "equity"
        orch._sizer = MagicMock()
        orch._broker = MagicMock()
        orch._executor = MagicMock()
        orch._logger = MagicMock()
        orch._sentiment = None
        orch._current_regime = None
        return orch

    def test_retrain_check_no_drift_no_retrain(self):
        orch = self._orch()
        orch._send_alert = MagicMock()

        mock_monitor = MagicMock()
        mock_monitor.compare_live_vs_backtest.return_value = 0.15  # below 0.3 threshold

        with (
            patch("monitoring.mlflow_tracker.ModelDriftMonitor", return_value=mock_monitor),
            patch.object(orch, "_auto_retrain_and_promote") as mock_retrain,
        ):
            orch.retrain_check()

        mock_retrain.assert_not_called()

    def test_retrain_check_drift_triggers_retrain(self):
        orch = self._orch()
        orch._send_alert = MagicMock()

        mock_monitor = MagicMock()
        mock_monitor.compare_live_vs_backtest.return_value = 0.45  # above 0.3 threshold

        with (
            patch("monitoring.mlflow_tracker.ModelDriftMonitor", return_value=mock_monitor),
            patch.object(orch, "_auto_retrain_and_promote") as mock_retrain,
        ):
            orch.retrain_check()

        mock_retrain.assert_called_once()

    def test_auto_retrain_low_auc_no_promote(self):
        orch = self._orch()
        orch._send_alert = MagicMock()

        import pandas as pd

        mock_df = pd.DataFrame(
            {c: [0.5] * 600 for c in ["open", "high", "low", "close", "volume", "label"]},
        )

        mock_trainer = MagicMock()
        mock_trainer.run.return_value = {
            "mean_auc": 0.52,  # below 0.60 threshold
            "std_auc": 0.03,
            "n_folds": 5,
            "folds": [],
        }
        mock_trainer.save_drift_reference.return_value = None
        mock_trainer._fold_results = []

        with (
            patch("data.store.get_engine"),
            patch("pandas.read_sql", return_value=mock_df),
            patch("signals.train.WalkForwardTrainer", return_value=mock_trainer),
            patch("signals.features.FEATURE_COLUMNS", ["close"]),
        ):
            orch._auto_retrain_and_promote()

        orch._send_alert.assert_called_once()
        alert_msg = orch._send_alert.call_args[0][0]
        assert "0.52" in alert_msg or "below" in alert_msg.lower() or "AUC" in alert_msg

    def test_auto_retrain_high_auc_promotes(self):
        orch = self._orch()
        orch._send_alert = MagicMock()

        import pandas as pd

        mock_df = pd.DataFrame(
            {c: [0.5] * 600 for c in ["open", "high", "low", "close", "volume", "label"]},
        )

        mock_trainer = MagicMock()
        mock_trainer.run.return_value = {
            "mean_auc": 0.68,  # above 0.60 threshold
            "std_auc": 0.02,
            "n_folds": 5,
            "folds": [{"auc": 0.70, "run_id": "abc123"}],
        }
        mock_trainer.save_drift_reference.return_value = None
        mock_trainer._fold_results = []

        mock_registry = MagicMock()
        mock_registry.register_model.return_value = "7"
        orch._registry = mock_registry

        mock_mlflow_client = MagicMock()

        with (
            patch("data.store.get_engine"),
            patch("pandas.read_sql", return_value=mock_df),
            patch("signals.train.WalkForwardTrainer", return_value=mock_trainer),
            patch("signals.features.FEATURE_COLUMNS", ["close"]),
            patch("mlflow.MlflowClient", return_value=mock_mlflow_client),
        ):
            orch._auto_retrain_and_promote()

        orch._send_alert.assert_called_once()
        alert_msg = orch._send_alert.call_args[0][0]
        assert "✅" in alert_msg or "promot" in alert_msg.lower() or "v7" in alert_msg

    def test_auto_retrain_exception_does_not_propagate(self):
        """Any exception inside _auto_retrain_and_promote must be caught."""
        orch = self._orch()
        orch._send_alert = MagicMock()

        with patch("data.store.get_engine", side_effect=RuntimeError("db down")):
            # Must not raise
            orch._auto_retrain_and_promote()

        orch._send_alert.assert_called_once()
        assert "❌" in orch._send_alert.call_args[0][0]
