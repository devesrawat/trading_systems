"""
Paper trading simulator — tracks positions and P&L without touching the Kite API.

Used during the 200+ paper trade validation phase before live deployment.

Public API
----------
PaperTrader(capital)
  .buy(symbol, qty, price)
  .sell(symbol, qty, price, current_price)
  .get_positions()   → dict[str, PaperPosition]
  .get_pnl(current_prices) → dict  {realized, unrealized, total, pct_return}
  .get_trade_history() → list[dict]
  .win_rate()        → float [0–1]
  .trade_count       → int

Exceptions
----------
InsufficientCapitalError — raised when a buy exceeds available_capital
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from zoneinfo import ZoneInfo

log = structlog.get_logger(__name__)

_IST = ZoneInfo("Asia/Kolkata")


class InsufficientCapitalError(Exception):
    """Raised when a paper buy order exceeds available capital."""


@dataclass
class PaperPosition:
    """Represents one open paper position."""

    symbol: str
    qty: int
    avg_price: float

    def __repr__(self) -> str:
        return (
            f"PaperPosition(symbol={self.symbol!r}, qty={self.qty}, "
            f"avg_price={self.avg_price:.2f})"
        )

    def market_value(self, current_price: float) -> float:
        return self.qty * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return (current_price - self.avg_price) * self.qty


class PaperTrader:
    """
    In-memory paper trading simulator.

    Tracks open positions, realized P&L, and a full trade history.
    No external calls — suitable for backtesting and dry-run validation.
    """

    def __init__(self, capital: float) -> None:
        self._initial_capital = capital
        self.capital = capital            # total allocated capital
        self.available_capital = capital  # uninvested cash
        self._positions: dict[str, PaperPosition] = {}
        self._realized_pnl: float = 0.0
        self._trade_history: list[dict[str, Any]] = []
        self._closed_trades: list[float] = []  # pnl per closed round-trip

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trade_count(self) -> int:
        return len(self._trade_history)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def buy(self, symbol: str, qty: int, price: float) -> None:
        """
        Execute a paper buy order.

        Raises
        ------
        InsufficientCapitalError : if cost > available_capital
        """
        cost = qty * price
        if cost > self.available_capital:
            raise InsufficientCapitalError(
                f"Cannot buy {qty}x{symbol}@{price:.2f} "
                f"(cost ₹{cost:,.0f} > available ₹{self.available_capital:,.0f})"
            )

        if symbol in self._positions:
            pos = self._positions[symbol]
            total_qty = pos.qty + qty
            pos.avg_price = (pos.avg_price * pos.qty + price * qty) / total_qty
            pos.qty = total_qty
        else:
            self._positions[symbol] = PaperPosition(symbol=symbol, qty=qty, avg_price=price)

        self.available_capital -= cost
        self._record_trade("BUY", symbol, qty, price)
        log.info("paper_buy", symbol=symbol, qty=qty, price=price,
                 available_capital=self.available_capital)

    def sell(
        self,
        symbol: str,
        qty: int,
        price: float,
        current_price: float,
    ) -> None:
        """
        Execute a paper sell order.

        Raises
        ------
        KeyError   : symbol not in open positions
        ValueError : qty > held qty
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for '{symbol}'")

        pos = self._positions[symbol]
        if qty > pos.qty:
            raise ValueError(
                f"Cannot sell {qty} shares of {symbol} — only {pos.qty} held"
            )

        trade_pnl = (price - pos.avg_price) * qty
        self._realized_pnl += trade_pnl
        self._closed_trades.append(trade_pnl)

        self.available_capital += qty * price

        if qty == pos.qty:
            del self._positions[symbol]
        else:
            pos.qty -= qty

        self._record_trade("SELL", symbol, qty, price)
        log.info(
            "paper_sell",
            symbol=symbol,
            qty=qty,
            price=price,
            trade_pnl=round(trade_pnl, 2),
            realized_pnl=round(self._realized_pnl, 2),
        )

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, PaperPosition]:
        """Return a shallow copy of open positions."""
        return dict(self._positions)

    def get_pnl(
        self,
        current_prices: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute P&L snapshot.

        Parameters
        ----------
        current_prices : {symbol: price} for mark-to-market. Pass None
                         (or omit) to get realized-only figures.

        Returns
        -------
        dict with keys: realized, unrealized, total, pct_return
        """
        unrealized = 0.0
        if current_prices:
            for symbol, pos in self._positions.items():
                if symbol in current_prices:
                    unrealized += pos.unrealized_pnl(current_prices[symbol])

        total = self._realized_pnl + unrealized
        pct_return = total / self._initial_capital if self._initial_capital else 0.0

        return {
            "realized": self._realized_pnl,
            "unrealized": unrealized,
            "total": total,
            "pct_return": pct_return,
        }

    def get_trade_history(self) -> list[dict[str, Any]]:
        """Return a copy of the full trade log."""
        return list(self._trade_history)

    def win_rate(self) -> float:
        """
        Fraction of closed round-trips with positive P&L.
        Returns 0.0 if no closed trades exist.
        """
        if not self._closed_trades:
            return 0.0
        wins = sum(1 for pnl in self._closed_trades if pnl > 0)
        return wins / len(self._closed_trades)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _record_trade(self, side: str, symbol: str, qty: int, price: float) -> None:
        self._trade_history.append(
            {
                "side": side,
                "symbol": symbol,
                "qty": qty,
                "price": price,
                "timestamp": datetime.now(tz=_IST).isoformat(),
            }
        )
