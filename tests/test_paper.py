"""Unit tests for execution/paper.py — TDD RED phase."""

from __future__ import annotations

import pytest

from execution.paper import InsufficientCapitalError, PaperPosition, PaperTrader


class TestPaperTraderInit:
    def test_default_capital(self):
        pt = PaperTrader(capital=500_000.0)
        assert pt.capital == 500_000.0

    def test_initial_positions_empty(self):
        pt = PaperTrader(capital=100_000.0)
        assert pt.get_positions() == {}

    def test_initial_pnl_zero(self):
        pt = PaperTrader(capital=100_000.0)
        pnl = pt.get_pnl()
        assert pnl["realized"] == 0.0
        assert pnl["unrealized"] == 0.0
        assert pnl["total"] == 0.0

    def test_initial_trade_count_zero(self):
        pt = PaperTrader(capital=100_000.0)
        assert pt.trade_count == 0


class TestPaperTraderBuy:
    def test_buy_creates_position(self):
        pt = PaperTrader(capital=100_000.0)
        pt.buy("RELIANCE", qty=10, price=2_800.0)
        positions = pt.get_positions()
        assert "RELIANCE" in positions

    def test_buy_stores_correct_qty(self):
        pt = PaperTrader(capital=100_000.0)
        pt.buy("INFY", qty=20, price=1_500.0)
        assert pt.get_positions()["INFY"].qty == 20

    def test_buy_stores_avg_price(self):
        pt = PaperTrader(capital=100_000.0)
        pt.buy("TCS", qty=5, price=3_600.0)
        assert pt.get_positions()["TCS"].avg_price == pytest.approx(3_600.0)

    def test_buy_deducts_capital(self):
        pt = PaperTrader(capital=100_000.0)
        pt.buy("WIPRO", qty=10, price=400.0)
        assert pt.available_capital == pytest.approx(100_000.0 - 10 * 400.0)

    def test_buy_raises_on_insufficient_capital(self):
        pt = PaperTrader(capital=1_000.0)
        with pytest.raises(InsufficientCapitalError):
            pt.buy("RELIANCE", qty=100, price=2_800.0)

    def test_buy_increments_trade_count(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("NIFTY", qty=1, price=22_000.0)
        assert pt.trade_count == 1

    def test_buy_average_cost_on_multiple_buys(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("HDFC", qty=10, price=1_600.0)
        pt.buy("HDFC", qty=10, price=1_800.0)
        avg = pt.get_positions()["HDFC"].avg_price
        assert avg == pytest.approx(1_700.0)


class TestPaperTraderSell:
    def test_sell_reduces_position(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("SBIN", qty=20, price=600.0)
        pt.sell("SBIN", qty=5, price=620.0, current_price=620.0)
        assert pt.get_positions()["SBIN"].qty == 15

    def test_sell_all_removes_position(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("ONGC", qty=10, price=200.0)
        pt.sell("ONGC", qty=10, price=210.0, current_price=210.0)
        assert "ONGC" not in pt.get_positions()

    def test_sell_books_realized_pnl(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("COALINDIA", qty=100, price=400.0)
        pt.sell("COALINDIA", qty=100, price=420.0, current_price=420.0)
        pnl = pt.get_pnl()
        assert pnl["realized"] == pytest.approx(2_000.0)  # (420-400)*100

    def test_sell_loss_books_negative_pnl(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("ITC", qty=50, price=300.0)
        pt.sell("ITC", qty=50, price=280.0, current_price=280.0)
        pnl = pt.get_pnl()
        assert pnl["realized"] == pytest.approx(-1_000.0)  # (280-300)*50

    def test_sell_restores_capital(self):
        pt = PaperTrader(capital=100_000.0)
        pt.buy("BPCL", qty=10, price=400.0)
        pt.sell("BPCL", qty=10, price=420.0, current_price=420.0)
        # capital = 100_000 - 4000 (buy) + 4200 (sell) = 100_200
        assert pt.available_capital == pytest.approx(100_200.0)

    def test_sell_nonexistent_raises(self):
        pt = PaperTrader(capital=100_000.0)
        with pytest.raises(KeyError):
            pt.sell("GHOST", qty=5, price=100.0, current_price=100.0)

    def test_sell_more_than_held_raises(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("TITAN", qty=10, price=3_000.0)
        with pytest.raises(ValueError):
            pt.sell("TITAN", qty=20, price=3_100.0, current_price=3_100.0)


class TestPaperTraderUnrealizedPnL:
    def test_unrealized_pnl_on_open_position(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("MARUTI", qty=5, price=10_000.0)
        pnl = pt.get_pnl(current_prices={"MARUTI": 10_500.0})
        assert pnl["unrealized"] == pytest.approx(2_500.0)  # (10500-10000)*5

    def test_unrealized_pnl_negative(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("BAJAJ-AUTO", qty=10, price=8_000.0)
        pnl = pt.get_pnl(current_prices={"BAJAJ-AUTO": 7_500.0})
        assert pnl["unrealized"] == pytest.approx(-5_000.0)

    def test_total_pnl_is_realized_plus_unrealized(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("HCLTECH", qty=20, price=1_200.0)
        pt.sell("HCLTECH", qty=10, price=1_250.0, current_price=1_250.0)
        pnl = pt.get_pnl(current_prices={"HCLTECH": 1_300.0})
        expected_realized = (1_250 - 1_200) * 10  # 500
        expected_unrealized = (1_300 - 1_200) * 10  # 1000
        assert pnl["realized"] == pytest.approx(expected_realized)
        assert pnl["unrealized"] == pytest.approx(expected_unrealized)
        assert pnl["total"] == pytest.approx(expected_realized + expected_unrealized)


class TestPaperTraderHistory:
    def test_trade_history_recorded(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("ADANIPORTS", qty=10, price=700.0)
        pt.sell("ADANIPORTS", qty=10, price=750.0, current_price=750.0)
        history = pt.get_trade_history()
        assert len(history) == 2

    def test_trade_history_has_correct_fields(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("ASIANPAINT", qty=5, price=3_000.0)
        trade = pt.get_trade_history()[0]
        for field in ("symbol", "side", "qty", "price", "timestamp"):
            assert field in trade

    def test_win_rate_all_wins(self):
        pt = PaperTrader(capital=500_000.0)
        for _ in range(5):
            pt.buy("DRREDDY", qty=1, price=5_000.0)
            pt.sell("DRREDDY", qty=1, price=5_100.0, current_price=5_100.0)
        assert pt.win_rate() == pytest.approx(1.0)

    def test_win_rate_mixed(self):
        pt = PaperTrader(capital=500_000.0)
        pt.buy("CIPLA", qty=1, price=1_000.0)
        pt.sell("CIPLA", qty=1, price=1_100.0, current_price=1_100.0)  # win
        pt.buy("CIPLA", qty=1, price=1_000.0)
        pt.sell("CIPLA", qty=1, price=900.0, current_price=900.0)  # loss
        assert pt.win_rate() == pytest.approx(0.5)

    def test_win_rate_no_closed_trades_returns_zero(self):
        pt = PaperTrader(capital=500_000.0)
        assert pt.win_rate() == 0.0


class TestPaperPosition:
    def test_position_repr(self):
        pos = PaperPosition(symbol="NIFTY50", qty=2, avg_price=22_000.0)
        r = repr(pos)
        assert "NIFTY50" in r
        assert "22000" in r
