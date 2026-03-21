"""Unit tests for execution/orders.py — TDD RED phase. Mocks Kite API."""
from unittest.mock import MagicMock, patch, call

import pytest

from execution.orders import OrderExecutor, SlippageTier, write_live_trade


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_executor(paper_mode: bool = True) -> tuple[OrderExecutor, MagicMock]:
    mock_kite = MagicMock()
    mock_kite.place_order.return_value = "ORDER123"
    mock_kite.cancel_order.return_value = True
    mock_kite.order_history.return_value = [
        {"order_id": "ORDER123", "status": "COMPLETE", "filled_quantity": 10}
    ]
    mock_kite.VARIETY_REGULAR = "regular"
    mock_kite.EXCHANGE_NSE = "NSE"
    mock_kite.ORDER_TYPE_MARKET = "MARKET"
    mock_kite.ORDER_TYPE_LIMIT = "LIMIT"

    mock_cb = MagicMock()
    mock_cb.is_halted.return_value = False
    mock_cb.check.return_value = (True, None)

    executor = OrderExecutor(kite=mock_kite, paper_mode=paper_mode, circuit_breaker=mock_cb)
    return executor, mock_kite


# ---------------------------------------------------------------------------
# Paper mode — default safety
# ---------------------------------------------------------------------------

class TestPaperMode:
    def test_paper_mode_on_by_default(self):
        mock_kite = MagicMock()
        mock_cb = MagicMock()
        mock_cb.check.return_value = (True, None)
        executor = OrderExecutor(kite=mock_kite, circuit_breaker=mock_cb)
        assert executor.paper_mode is True

    def test_paper_mode_never_calls_kite_place_order(self):
        executor, mock_kite = _make_executor(paper_mode=True)
        with patch("execution.orders._write_paper_trade"):
            executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        mock_kite.place_order.assert_not_called()

    def test_live_mode_calls_kite_place_order(self):
        executor, mock_kite = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade"):
            executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        mock_kite.place_order.assert_called_once()

    def test_paper_mode_writes_to_paper_trades_table(self):
        executor, _ = _make_executor(paper_mode=True)
        with patch("execution.orders._write_paper_trade") as mock_write:
            executor.place_market_order("TCS", "BUY", quantity=5, tag="signal_test")
        mock_write.assert_called_once()


# ---------------------------------------------------------------------------
# place_market_order
# ---------------------------------------------------------------------------

class TestPlaceMarketOrder:
    def test_returns_order_id(self):
        executor, _ = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade"):
            order_id = executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        assert order_id is not None

    def test_blocked_when_circuit_breaker_halted(self):
        mock_kite = MagicMock()
        mock_cb = MagicMock()
        mock_cb.is_halted.return_value = True
        mock_cb.check.return_value = (False, "daily drawdown exceeded")
        executor = OrderExecutor(kite=mock_kite, paper_mode=False, circuit_breaker=mock_cb)
        with pytest.raises(RuntimeError, match="circuit breaker"):
            executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")

    def test_zero_quantity_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(ValueError, match="quantity"):
            executor.place_market_order("RELIANCE", "BUY", quantity=0, tag="test")

    def test_negative_quantity_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(ValueError, match="quantity"):
            executor.place_market_order("RELIANCE", "BUY", quantity=-5, tag="test")

    def test_invalid_transaction_type_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(ValueError, match="transaction_type"):
            executor.place_market_order("RELIANCE", "HOLD", quantity=10, tag="test")

    def test_buy_uses_mis_product_by_default(self):
        executor, mock_kite = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade"):
            executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        call_kwargs = mock_kite.place_order.call_args[1]
        assert call_kwargs.get("product") in ("MIS", "CNC")

    def test_paper_order_id_is_string(self):
        executor, _ = _make_executor(paper_mode=True)
        with patch("execution.orders._write_paper_trade"):
            order_id = executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        assert isinstance(order_id, str)


# ---------------------------------------------------------------------------
# place_limit_order
# ---------------------------------------------------------------------------

class TestPlaceLimitOrder:
    def test_returns_order_id(self):
        executor, _ = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade"):
            order_id = executor.place_limit_order("TCS", "BUY", quantity=5, price=3500.0, tag="test")
        assert order_id is not None

    def test_large_order_prefers_limit(self):
        """Orders > ₹50,000 should use limit orders to reduce slippage."""
        executor, mock_kite = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade"):
            executor.place_limit_order("RELIANCE", "BUY", quantity=100, price=600.0, tag="test")
        mock_kite.place_order.assert_called_once()
        call_kwargs = mock_kite.place_order.call_args[1]
        assert call_kwargs.get("order_type") == "LIMIT"

    def test_zero_price_raises(self):
        executor, _ = _make_executor()
        with pytest.raises(ValueError, match="price"):
            executor.place_limit_order("TCS", "BUY", quantity=5, price=0.0, tag="test")


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    def test_cancel_returns_true_on_success(self):
        executor, mock_kite = _make_executor(paper_mode=False)
        mock_kite.cancel_order.return_value = {"order_id": "ORDER123"}
        result = executor.cancel_order("ORDER123")
        assert result is True

    def test_cancel_returns_false_on_error(self):
        executor, mock_kite = _make_executor(paper_mode=False)
        mock_kite.cancel_order.side_effect = Exception("Order not found")
        result = executor.cancel_order("BAD_ORDER")
        assert result is False

    def test_paper_mode_cancel_always_true(self):
        executor, mock_kite = _make_executor(paper_mode=True)
        result = executor.cancel_order("PAPER_ORDER_1")
        assert result is True
        mock_kite.cancel_order.assert_not_called()


# ---------------------------------------------------------------------------
# get_order_status
# ---------------------------------------------------------------------------

class TestGetOrderStatus:
    def test_returns_dict(self):
        executor, _ = _make_executor(paper_mode=False)
        status = executor.get_order_status("ORDER123")
        assert isinstance(status, dict)

    def test_status_contains_order_id(self):
        executor, _ = _make_executor(paper_mode=False)
        status = executor.get_order_status("ORDER123")
        assert "order_id" in status or "status" in status


# ---------------------------------------------------------------------------
# slippage_estimate
# ---------------------------------------------------------------------------

class TestSlippageEstimate:
    def test_large_cap_lower_slippage(self):
        executor, _ = _make_executor()
        slip = executor.slippage_estimate("RELIANCE", quantity=10, side="BUY")
        assert slip <= 0.001   # ≤ 0.1%

    def test_slippage_is_fraction_not_rupees(self):
        executor, _ = _make_executor()
        slip = executor.slippage_estimate("RELIANCE", quantity=10, side="BUY")
        assert 0.0 < slip < 0.01    # should be between 0% and 1%

    def test_slippage_tiers(self):
        executor, _ = _make_executor()
        large = executor.slippage_estimate("RELIANCE", quantity=10, side="BUY",
                                           tier=SlippageTier.LARGE_CAP)
        mid = executor.slippage_estimate("RELIANCE", quantity=10, side="BUY",
                                         tier=SlippageTier.MID_CAP)
        small = executor.slippage_estimate("RELIANCE", quantity=10, side="BUY",
                                           tier=SlippageTier.SMALL_CAP)
        assert large < mid < small


# ---------------------------------------------------------------------------
# write_live_trade (SEBI audit trail)
# ---------------------------------------------------------------------------

class TestWriteLiveTrade:
    def _mock_engine(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine, mock_conn

    def test_executes_insert(self):
        engine, conn = self._mock_engine()
        with patch("execution.orders.get_engine", return_value=engine):
            write_live_trade(
                symbol="RELIANCE", side="BUY", quantity=10, price=2500.0,
                order_id="ORD001", tag="signal",
            )
        conn.execute.assert_called_once()

    def test_commits_after_insert(self):
        engine, conn = self._mock_engine()
        with patch("execution.orders.get_engine", return_value=engine):
            write_live_trade(
                symbol="TCS", side="SELL", quantity=5, price=3600.0,
                order_id="ORD002", tag="exit",
            )
        conn.commit.assert_called_once()

    def test_passes_order_id_in_params(self):
        engine, conn = self._mock_engine()
        with patch("execution.orders.get_engine", return_value=engine):
            write_live_trade(
                symbol="INFY", side="BUY", quantity=8, price=1400.0,
                order_id="ORD999", tag="test",
            )
        _, kwargs = conn.execute.call_args
        params = conn.execute.call_args[0][1]
        assert params["order_id"] == "ORD999"

    def test_live_market_order_calls_write_live_trade(self):
        executor, _ = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade") as mock_write:
            executor.place_market_order("RELIANCE", "BUY", quantity=10, tag="test")
        mock_write.assert_called_once()

    def test_live_limit_order_calls_write_live_trade(self):
        executor, _ = _make_executor(paper_mode=False)
        with patch("execution.orders.write_live_trade") as mock_write:
            executor.place_limit_order("TCS", "BUY", quantity=5, price=3500.0, tag="test")
        mock_write.assert_called_once()
