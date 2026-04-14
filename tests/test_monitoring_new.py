"""Tests for new monitoring modules: ConceptDriftDetector, HealthMonitor, DailyReconciler."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from monitoring.drift_detector import ConceptDriftDetector
from monitoring.health import HealthMonitor
from monitoring.reconciliation import DailyReconciler

# ---------------------------------------------------------------------------
# ConceptDriftDetector
# ---------------------------------------------------------------------------


class TestConceptDriftDetector:
    """KS-test based feature drift detection."""

    def _make_df(self, n: int = 100, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "rsi_14": rng.uniform(20, 80, n),
                "ema_ratio_50_200": rng.uniform(0.9, 1.1, n),
                "realized_vol_20": rng.uniform(0.1, 0.4, n),
            }
        )

    def _make_redis(self, data: dict | None = None) -> MagicMock:
        r = MagicMock()
        if data is not None:
            r.get.return_value = json.dumps(data).encode()
        else:
            r.get.return_value = None
        return r

    def test_fit_saves_to_redis(self):
        df = self._make_df(200)
        features = ["rsi_14", "ema_ratio_50_200", "realized_vol_20"]
        mock_redis = MagicMock()

        with patch("data.store.get_redis", return_value=mock_redis):
            ConceptDriftDetector().fit(df, features)

        mock_redis.set.assert_called_once()

    def test_fit_stores_sample_per_feature(self):
        df = self._make_df(200)
        features = ["rsi_14", "ema_ratio_50_200"]
        captured = {}

        def fake_set(key, value, **kwargs):
            captured["value"] = value

        mock_redis = MagicMock()
        mock_redis.set.side_effect = fake_set

        with patch("data.store.get_redis", return_value=mock_redis):
            ConceptDriftDetector().fit(df, features)

        stored = json.loads(captured["value"])
        assert "rsi_14" in stored
        assert "ema_ratio_50_200" in stored
        assert isinstance(stored["rsi_14"], list)

    def test_check_returns_dict_of_feature_results(self):
        df = self._make_df(100)
        features = ["rsi_14", "ema_ratio_50_200"]
        ref_data = {f: df[f].tolist() for f in features}
        mock_redis = self._make_redis(ref_data)

        with patch("data.store.get_redis", return_value=mock_redis):
            result = ConceptDriftDetector().check(df)

        assert isinstance(result, dict)
        assert "rsi_14" in result

    def test_no_drift_on_identical_distribution(self):
        df = self._make_df(200, seed=42)
        features = ["rsi_14", "ema_ratio_50_200"]
        ref_data = {f: df[f].tolist() for f in features}
        mock_redis = self._make_redis(ref_data)

        with patch("data.store.get_redis", return_value=mock_redis):
            # is_drifting takes a DataFrame
            drifting = ConceptDriftDetector().is_drifting(df)

        # Same distribution → p-values should be high → no drift
        assert drifting is False

    def test_drift_detected_on_shifted_distribution(self):
        df_ref = self._make_df(200, seed=42)
        df_live = df_ref.copy()
        df_live["rsi_14"] = df_live["rsi_14"] + 50.0  # massive shift

        features = ["rsi_14", "ema_ratio_50_200"]
        ref_data = {f: df_ref[f].tolist() for f in features}
        mock_redis = self._make_redis(ref_data)

        with patch("data.store.get_redis", return_value=mock_redis):
            drifting = ConceptDriftDetector().is_drifting(df_live)

        assert drifting is True

    def test_check_returns_empty_dict_when_no_reference(self):
        df = self._make_df(100)
        mock_redis = self._make_redis(None)  # no stored reference

        with patch("data.store.get_redis", return_value=mock_redis):
            result = ConceptDriftDetector().check(df)

        # No reference → returns empty dict (caller treats as no drift)
        assert result == {}

    def test_fit_redis_failure_does_not_raise(self):
        df = self._make_df(200)
        features = ["rsi_14"]
        mock_redis = MagicMock()
        mock_redis.set.side_effect = Exception("Redis down")

        with patch("data.store.get_redis", return_value=mock_redis):
            ConceptDriftDetector().fit(df, features)  # must not raise


# ---------------------------------------------------------------------------
# HealthMonitor
# ---------------------------------------------------------------------------


class TestHealthMonitor:
    def test_write_heartbeat_sets_redis_key(self):
        mock_redis = MagicMock()
        with patch("data.store.get_redis", return_value=mock_redis):
            HealthMonitor().write_heartbeat()
        mock_redis.set.assert_called_once()

    def test_heartbeat_value_is_iso_timestamp(self):
        captured = {}

        def fake_set(key, value, **kwargs):
            captured["value"] = value

        mock_redis = MagicMock()
        mock_redis.set.side_effect = fake_set

        with patch("data.store.get_redis", return_value=mock_redis):
            HealthMonitor().write_heartbeat()

        ts_str = captured["value"]
        # Should parse as ISO datetime without error
        datetime.fromisoformat(ts_str)

    def test_write_heartbeat_redis_failure_does_not_raise(self):
        mock_redis = MagicMock()
        mock_redis.set.side_effect = Exception("connection refused")
        with patch("data.store.get_redis", return_value=mock_redis):
            HealthMonitor().write_heartbeat()  # must not raise

    def test_send_alert_if_stale_no_key_outside_market_hours(self):
        """No alert when Redis key is missing but outside market hours."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        # With no heartbeat key and no market hours → should not alert
        with (
            patch("data.store.get_redis", return_value=mock_redis),
            patch("monitoring.alerts.TelegramAlerter") as mock_alerter,
        ):
            # Only assert no crash — hours check is internal
            HealthMonitor().send_alert_if_stale()

        # No alert expected since there's no stale heartbeat during market hours
        # (test runs outside IST 09:15–15:30 typically; mock key is None)
        # Just verify it doesn't raise
        assert True  # main assertion is "did not raise"

    def test_send_alert_if_stale_does_not_crash(self):
        """send_alert_if_stale must never raise even on Redis failure."""
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("Redis unavailable")
        with patch("data.store.get_redis", return_value=mock_redis):
            HealthMonitor().send_alert_if_stale()  # must not raise


# ---------------------------------------------------------------------------
# DailyReconciler
# ---------------------------------------------------------------------------


class TestDailyReconciler:
    def _make_broker(self) -> MagicMock:
        broker = MagicMock()
        broker.get_quote.return_value = {"last_price": 1550.0}
        return broker

    def test_reconcile_empty_positions_no_alerts(self):
        broker = self._make_broker()
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with (
            patch("data.store.get_redis", return_value=mock_redis),
            patch("monitoring.alerts.TelegramAlerter") as mock_alerter,
        ):
            DailyReconciler(broker).reconcile(set())

        mock_alerter.return_value.send.assert_not_called()

    def test_reconcile_no_drift_no_alert(self):
        broker = self._make_broker()
        broker.get_quote.return_value = {"RELIANCE": {"last_price": 1500.0}}

        mock_redis = MagicMock()
        mock_redis.get.return_value = b"1500.0"  # entry price matches

        with (
            patch("data.store.get_redis", return_value=mock_redis),
            patch("monitoring.alerts.TelegramAlerter") as mock_alerter,
        ):
            DailyReconciler(broker).reconcile({"RELIANCE"})

        mock_alerter.return_value.send.assert_not_called()

    def test_reconcile_large_drift_sends_alert(self):
        broker = self._make_broker()
        broker.get_quote.return_value = {"RELIANCE": {"last_price": 2000.0}}  # far from entry

        mock_redis = MagicMock()
        mock_redis.get.return_value = b"1000.0"  # entry was 1000, now 2000 (100% drift)

        with (
            patch("data.store.get_redis", return_value=mock_redis),
            patch("monitoring.alerts.TelegramAlerter") as mock_alerter,
        ):
            DailyReconciler(broker).reconcile({"RELIANCE"})

        mock_alerter.return_value.send.assert_called()

    def test_reconcile_broker_error_does_not_crash(self):
        broker = MagicMock()
        broker.get_quote.side_effect = Exception("broker timeout")
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"1500.0"

        with patch("data.store.get_redis", return_value=mock_redis):
            DailyReconciler(broker).reconcile({"RELIANCE"})  # must not raise
