"""Unit tests for risk/monitor.py — TDD RED phase. Mocks Redis."""

from unittest.mock import MagicMock, patch

import pytest

from risk.monitor import PortfolioMonitor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_monitor() -> tuple[PortfolioMonitor, MagicMock]:
    with patch("risk.monitor.get_redis") as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.hgetall.return_value = {}
        mock_get_redis.return_value = mock_redis
        monitor = PortfolioMonitor(initial_capital=500_000.0)
    return monitor, mock_redis


# ---------------------------------------------------------------------------
# update_position
# ---------------------------------------------------------------------------


class TestUpdatePosition:
    def test_adds_new_position(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("RELIANCE", qty=10, avg_price=2500.0, current_price=2550.0)
        assert "RELIANCE" in monitor._positions

    def test_removes_position_on_zero_qty(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("RELIANCE", qty=10, avg_price=2500.0, current_price=2550.0)
            monitor.update_position("RELIANCE", qty=0, avg_price=2500.0, current_price=2550.0)
        assert "RELIANCE" not in monitor._positions

    def test_updates_existing_position(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("TCS", qty=5, avg_price=3500.0, current_price=3600.0)
            monitor.update_position("TCS", qty=10, avg_price=3550.0, current_price=3600.0)
        assert monitor._positions["TCS"]["qty"] == 10


# ---------------------------------------------------------------------------
# get_pnl
# ---------------------------------------------------------------------------


class TestGetPnl:
    def test_unrealized_pnl_correct(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("RELIANCE", qty=10, avg_price=2500.0, current_price=2600.0)
        pnl = monitor.get_pnl()
        assert pnl["unrealized"] == pytest.approx(1000.0)  # 10 * (2600 - 2500)

    def test_zero_pnl_on_no_positions(self):
        monitor, _ = _make_monitor()
        pnl = monitor.get_pnl()
        assert pnl["unrealized"] == 0.0
        assert pnl["realized"] == 0.0
        assert pnl["total"] == 0.0

    def test_pct_return_calculated(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("INFY", qty=10, avg_price=1000.0, current_price=1100.0)
        pnl = monitor.get_pnl()
        assert pnl["pct_return"] == pytest.approx(0.002)  # 1000 / 500_000

    def test_record_realized_pnl(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.record_realized(symbol="HDFC", pnl=5000.0)
        pnl = monitor.get_pnl()
        assert pnl["realized"] == 5000.0

    def test_total_is_realized_plus_unrealized(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.record_realized("HDFC", 3000.0)
            monitor.update_position("RELIANCE", qty=5, avg_price=2000.0, current_price=2200.0)
        pnl = monitor.get_pnl()
        assert pnl["total"] == pytest.approx(3000.0 + 1000.0)


# ---------------------------------------------------------------------------
# get_drawdown
# ---------------------------------------------------------------------------


class TestGetDrawdown:
    def test_no_loss_zero_drawdown(self):
        monitor, _ = _make_monitor()
        dd = monitor.get_drawdown()
        assert dd["daily_dd"] == 0.0
        assert dd["current_dd"] == 0.0

    def test_daily_drawdown_computed(self):
        monitor, _ = _make_monitor()
        monitor._daily_start_capital = 500_000.0
        monitor._current_capital = 485_000.0
        dd = monitor.get_drawdown()
        assert dd["daily_dd"] == pytest.approx(0.03)

    def test_max_drawdown_tracks_worst(self):
        monitor, _ = _make_monitor()
        monitor._peak_capital = 510_000.0
        monitor._current_capital = 480_000.0
        dd = monitor.get_drawdown()
        # max_dd = (510k - 480k) / 510k ≈ 5.88%
        assert dd["max_dd"] == pytest.approx(30_000 / 510_000)

    def test_weekly_drawdown_computed(self):
        monitor, _ = _make_monitor()
        monitor._weekly_start_capital = 500_000.0
        monitor._current_capital = 465_000.0
        dd = monitor.get_drawdown()
        assert dd["weekly_dd"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# get_exposure
# ---------------------------------------------------------------------------


class TestGetExposure:
    def test_no_positions_zero_exposure(self):
        monitor, _ = _make_monitor()
        exp = monitor.get_exposure()
        assert exp["gross_exposure"] == 0.0
        assert exp["net_exposure"] == 0.0

    def test_gross_exposure_sum_of_position_values(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("RELIANCE", qty=10, avg_price=2500.0, current_price=2500.0)
            monitor.update_position("TCS", qty=5, avg_price=3500.0, current_price=3500.0)
        exp = monitor.get_exposure()
        assert exp["gross_exposure"] == pytest.approx(25_000.0 + 17_500.0)

    def test_largest_position_pct_correct(self):
        monitor, _ = _make_monitor()
        with patch.object(monitor, "_persist"):
            monitor.update_position("RELIANCE", qty=10, avg_price=2500.0, current_price=2500.0)
        exp = monitor.get_exposure()
        expected_pct = 25_000.0 / 500_000.0
        assert exp["largest_position_pct"] == pytest.approx(expected_pct)


# ---------------------------------------------------------------------------
# Prometheus metrics export
# ---------------------------------------------------------------------------


class TestPrometheusExport:
    def test_export_returns_string(self):
        monitor, _ = _make_monitor()
        result = monitor.export_prometheus_metrics()
        assert isinstance(result, str)

    def test_export_contains_pnl_metric(self):
        monitor, _ = _make_monitor()
        result = monitor.export_prometheus_metrics()
        assert "pnl" in result.lower() or "unrealized" in result.lower()

    def test_export_contains_drawdown_metric(self):
        monitor, _ = _make_monitor()
        result = monitor.export_prometheus_metrics()
        assert "drawdown" in result.lower() or "dd" in result.lower()
