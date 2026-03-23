"""
Kite order placement, cancellation, and paper trading simulator.

ALWAYS defaults to paper_mode=True. Live trading requires explicit opt-in
after 200+ paper trades and go-live checklist completion.
"""
from __future__ import annotations

import uuid
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import text

from data.store import get_engine
from execution.broker import BrokerAdapter

if TYPE_CHECKING:
    from risk.breakers import CircuitBreaker

log = structlog.get_logger(__name__)

_VALID_SIDES = frozenset({"BUY", "SELL"})
_LARGE_ORDER_THRESHOLD_INR = 50_000


class SlippageTier(str, Enum):
    LARGE_CAP = "large_cap"
    MID_CAP = "mid_cap"
    SMALL_CAP = "small_cap"


_SLIPPAGE_MAP = {
    SlippageTier.LARGE_CAP: 0.0005,   # 0.05%
    SlippageTier.MID_CAP: 0.0012,     # 0.12%
    SlippageTier.SMALL_CAP: 0.0020,   # 0.20%
}

_PAPER_ORDER_PREFIX = "PAPER"


class OrderExecutor:
    """
    Places, cancels, and tracks orders via the configured :class:`BrokerAdapter`.

    Works with Kite, Upstox, Binance, or any future broker — the adapter
    abstracts all broker-specific API calls.  In paper mode the adapter's
    :attr:`~BrokerAdapter.is_paper` flag is True and no real orders are sent.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        circuit_breaker: "CircuitBreaker",
    ) -> None:
        self._broker = broker
        self._cb = circuit_breaker
        if not broker.is_paper:
            log.warning("live_trading_mode_active", adapter=type(broker).__name__)

    @property
    def paper_mode(self) -> bool:
        """True when the underlying broker adapter is in paper mode."""
        return self._broker.is_paper

    # ------------------------------------------------------------------
    # place_market_order
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        tag: str,
        intraday: bool = True,
    ) -> str:
        """
        Place a market order. Returns order_id string.

        Validates circuit breaker, quantity, and transaction type before placing.
        """
        self._validate_order(transaction_type, quantity)
        self._check_circuit_breaker()

        log.info(
            "placing_market_order",
            symbol=symbol,
            side=transaction_type,
            qty=quantity,
            paper=self.paper_mode,
        )

        if self._broker.is_paper:
            order_id = f"{_PAPER_ORDER_PREFIX}_{uuid.uuid4().hex[:8].upper()}"
            _write_paper_trade(symbol, transaction_type, quantity, price=0.0, tag=tag)
            return order_id

        order_id = self._broker.place_order(
            symbol=symbol,
            side=transaction_type,
            quantity=quantity,
            tag=tag,
            order_type="MARKET",
            intraday=intraday,
        )
        write_live_trade(symbol, transaction_type, quantity, price=0.0, order_id=order_id, tag=tag)
        log.info("market_order_placed", order_id=order_id, symbol=symbol)
        return order_id

    # ------------------------------------------------------------------
    # place_limit_order
    # ------------------------------------------------------------------

    def place_limit_order(
        self,
        symbol: str,
        transaction_type: str,
        quantity: int,
        price: float,
        tag: str,
        intraday: bool = True,
    ) -> str:
        """
        Place a limit order. Preferred for sizes > ₹50,000 to reduce slippage.
        """
        self._validate_order(transaction_type, quantity)
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self._check_circuit_breaker()

        log.info(
            "placing_limit_order",
            symbol=symbol,
            side=transaction_type,
            qty=quantity,
            price=price,
            paper=self.paper_mode,
        )

        if self._broker.is_paper:
            order_id = f"{_PAPER_ORDER_PREFIX}_{uuid.uuid4().hex[:8].upper()}"
            _write_paper_trade(symbol, transaction_type, quantity, price=price, tag=tag)
            return order_id

        order_id = self._broker.place_order(
            symbol=symbol,
            side=transaction_type,
            quantity=quantity,
            tag=tag,
            price=price,
            order_type="LIMIT",
            intraday=intraday,
        )
        write_live_trade(symbol, transaction_type, quantity, price=price, order_id=order_id, tag=tag)
        log.info("limit_order_placed", order_id=order_id, symbol=symbol, price=price)
        return order_id

    # ------------------------------------------------------------------
    # cancel_order
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success, False on error."""
        return self._broker.cancel_order(order_id)

    # ------------------------------------------------------------------
    # get_order_status
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Return the current status dict for *order_id*."""
        return self._broker.get_order_status(order_id)

    # ------------------------------------------------------------------
    # slippage_estimate
    # ------------------------------------------------------------------

    def slippage_estimate(
        self,
        symbol: str,
        quantity: int,
        side: str,
        tier: SlippageTier = SlippageTier.LARGE_CAP,
    ) -> float:
        """
        Estimate slippage as a fraction of trade value.

        Returns a float between 0 and 1 (e.g. 0.0005 = 0.05%).
        """
        return _SLIPPAGE_MAP[tier]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_order(self, transaction_type: str, quantity: int) -> None:
        if transaction_type not in _VALID_SIDES:
            raise ValueError(
                f"Invalid transaction_type '{transaction_type}'. Must be BUY or SELL."
            )
        if quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {quantity}")

    def _check_circuit_breaker(self) -> None:
        if self._cb.is_halted():
            raise RuntimeError(
                "Order blocked — circuit breaker is active. "
                f"Reason: {self._cb._halt_reason}"
            )


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

_INSERT_PAPER_TRADE = text("""
    INSERT INTO paper_trades
        (symbol, side, quantity, price, signal_prob, position_size_inr, tag)
    VALUES
        (:symbol, :side, :quantity, :price, :signal_prob, :position_size_inr, :tag)
""")

_INSERT_LIVE_TRADE = text("""
    INSERT INTO live_trades
        (symbol, side, quantity, price, order_id, signal_prob,
         position_size_inr, tag, strategy_version)
    VALUES
        (:symbol, :side, :quantity, :price, :order_id, :signal_prob,
         :position_size_inr, :tag, :strategy_version)
""")


def _write_paper_trade(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    tag: str,
    signal_prob: float = 0.0,
    position_size_inr: float = 0.0,
) -> None:
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(_INSERT_PAPER_TRADE, {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "signal_prob": signal_prob,
            "position_size_inr": position_size_inr,
            "tag": tag,
        })
        conn.commit()
    log.debug("paper_trade_written", symbol=symbol, side=side, qty=quantity)


def write_live_trade(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    order_id: str,
    tag: str,
    signal_prob: float = 0.0,
    position_size_inr: float = 0.0,
    strategy_version: str = "1.0",
) -> None:
    """Write a confirmed live (non-paper) order to the live_trades audit table."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(_INSERT_LIVE_TRADE, {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "order_id": order_id,
            "signal_prob": signal_prob,
            "position_size_inr": position_size_inr,
            "tag": tag,
            "strategy_version": strategy_version,
        })
        conn.commit()
    log.info("live_trade_written", order_id=order_id, symbol=symbol, side=side, qty=quantity)
