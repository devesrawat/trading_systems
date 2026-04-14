"""
Broker adapter abstraction layer.

Decouples the order execution engine from any specific broker API so the
rest of the system can trade via Kite, Upstox, Binance, or any future
broker without touching the orchestrator or OrderExecutor.

Hierarchy::

    BrokerAdapter (ABC)
    ├── KiteBrokerAdapter     — Zerodha Kite Connect live orders
    ├── UpstoxBrokerAdapter   — Upstox v2 live orders
    └── PaperBrokerAdapter    — no-op; for dev, paper trading, and crypto

Factory::

    from execution.broker import get_broker_adapter
    broker = get_broker_adapter()   # reads DATA_PROVIDER + PAPER_TRADE_MODE from settings

Adding a new broker::

    class MyBrokerAdapter(BrokerAdapter):
        @property
        def is_paper(self) -> bool: return False
        def place_order(self, ...) -> str: ...
        def cancel_order(self, ...) -> bool: ...
        def get_order_status(self, ...) -> dict: ...
        def get_balance(self) -> float: ...
        def refresh_auth(self) -> None: ...  # optional

    # Register in get_broker_adapter() factory below.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_BASE_UPSTOX = "https://api.upstox.com/v2"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BrokerAdapter(ABC):
    """Common interface every broker adapter must implement.

    In paper mode (:attr:`is_paper` = ``True``) the adapter must return a
    synthetic order-id without touching any external API.  Only
    :class:`PaperBrokerAdapter` sets ``is_paper = True``; all real-broker
    adapters set it to ``False``.
    """

    @property
    @abstractmethod
    def is_paper(self) -> bool:
        """Return True if this adapter never touches a real broker."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,                  # "BUY" | "SELL"
        quantity: int,
        tag: str,
        price: float = 0.0,         # 0.0 → market order
        order_type: str = "MARKET", # "MARKET" | "LIMIT"
        intraday: bool = True,
    ) -> str:
        """Submit an order and return the broker's order-id string."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.  Returns True on success."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """Return the latest status dict for *order_id*."""

    @abstractmethod
    def get_balance(self) -> float:
        """Return the available cash balance in the account (INR or USD)."""

    def refresh_auth(self) -> None:
        """Refresh daily auth tokens if required (no-op for most adapters)."""


# ---------------------------------------------------------------------------
# Kite
# ---------------------------------------------------------------------------

class KiteBrokerAdapter(BrokerAdapter):
    """Zerodha Kite Connect live-order adapter.

    Requires a fully authenticated :class:`kiteconnect.KiteConnect` instance.
    Daily token refresh is handled by :meth:`refresh_auth` which reads the
    cached token from Redis.
    """

    def __init__(self, kite: Any) -> None:
        # kite: kiteconnect.KiteConnect — typed as Any to avoid hard import
        self._kite = kite

    @property
    def is_paper(self) -> bool:
        return False

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        tag: str,
        price: float = 0.0,
        order_type: str = "MARKET",
        intraday: bool = True,
    ) -> str:
        product = "MIS" if intraday else "CNC"
        kite_order_type = (
            self._kite.ORDER_TYPE_MARKET if order_type == "MARKET"
            else self._kite.ORDER_TYPE_LIMIT
        )
        kwargs: dict[str, Any] = dict(
            variety=self._kite.VARIETY_REGULAR,
            exchange=self._kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=side,
            quantity=quantity,
            order_type=kite_order_type,
            product=product,
            tag=tag,
        )
        if order_type == "LIMIT" and price > 0:
            kwargs["price"] = price

        order_id = str(self._kite.place_order(**kwargs))
        log.info("kite_order_placed", order_id=order_id, symbol=symbol, side=side)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._kite.cancel_order(
                variety=self._kite.VARIETY_REGULAR,
                order_id=order_id,
            )
            return True
        except Exception as exc:
            log.error("kite_cancel_failed", order_id=order_id, error=str(exc))
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        try:
            history = self._kite.order_history(order_id=order_id)
            return history[-1] if history else {"order_id": order_id, "status": "UNKNOWN"}
        except Exception as exc:
            log.error("kite_status_failed", order_id=order_id, error=str(exc))
            return {"order_id": order_id, "status": "ERROR", "error": str(exc)}

    def get_balance(self) -> float:
        try:
            margins = self._kite.margins()
            return float(
                margins.get("equity", {})
                       .get("available", {})
                       .get("live_balance", 0.0)
            )
        except Exception as exc:
            log.warning("kite_balance_failed", error=str(exc))
            return 0.0

    def refresh_auth(self) -> None:
        """Read the daily access token from Redis and set it on KiteConnect."""
        try:
            from data.store import get_redis
            from data.redis_keys import RedisKeys
            token = get_redis().get(RedisKeys.KITE_ACCESS_TOKEN)
            if token:
                self._kite.set_access_token(token)
                log.info("kite_token_refreshed_from_redis")
        except Exception as exc:
            log.warning("kite_token_refresh_failed", error=str(exc))


# ---------------------------------------------------------------------------
# Upstox
# ---------------------------------------------------------------------------

class UpstoxBrokerAdapter(BrokerAdapter):
    """Upstox v2 live-order adapter.

    Args:
        access_token:    Daily OAuth access token (set once per session).
        symbol_to_key:   ``{symbol: "NSE_EQ|ISIN"}`` mapping — same one used
                         by :class:`data.providers.upstox.UpstoxProvider`.
                         Pass an empty dict initially and populate via
                         :meth:`register_instruments`.
    """

    def __init__(
        self,
        access_token: str,
        symbol_to_key: dict[str, str] | None = None,
    ) -> None:
        self._token = access_token
        self._symbol_to_key: dict[str, str] = symbol_to_key or {}

    @property
    def is_paper(self) -> bool:
        return False

    def register_instruments(self, mapping: dict[str, str]) -> None:
        """Sync instrument key map from the data provider."""
        self._symbol_to_key.update(mapping)

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        tag: str,
        price: float = 0.0,
        order_type: str = "MARKET",
        intraday: bool = True,
    ) -> str:
        import requests as _req

        instrument_key = self._symbol_to_key.get(symbol)
        if not instrument_key:
            raise ValueError(
                f"No Upstox instrument key for '{symbol}'. "
                "Call register_instruments() first."
            )
        payload: dict[str, Any] = {
            "quantity":        quantity,
            "product":         "I" if intraday else "D",  # Intraday | Delivery
            "validity":        "DAY",
            "price":           price if order_type == "LIMIT" else 0,
            "instrument_token": instrument_key,
            "order_type":      order_type,
            "transaction_type": side,
            "tag":             tag,
        }
        resp = _req.post(
            f"{_BASE_UPSTOX}/order/place",
            json=payload,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept":        "application/json",
                "Content-Type":  "application/json",
            },
            timeout=10,
        )
        resp.raise_for_status()
        order_id: str = resp.json()["data"]["order_id"]
        log.info("upstox_order_placed", order_id=order_id, symbol=symbol, side=side)
        return order_id

    def cancel_order(self, order_id: str) -> bool:
        import requests as _req
        try:
            resp = _req.delete(
                f"{_BASE_UPSTOX}/order/cancel",
                params={"order_id": order_id},
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.error("upstox_cancel_failed", order_id=order_id, error=str(exc))
            return False

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        import requests as _req
        try:
            resp = _req.get(
                f"{_BASE_UPSTOX}/order/details",
                params={"order_id": order_id},
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json().get("data", {"order_id": order_id, "status": "UNKNOWN"})
        except Exception as exc:
            log.error("upstox_status_failed", order_id=order_id, error=str(exc))
            return {"order_id": order_id, "status": "ERROR", "error": str(exc)}

    def get_balance(self) -> float:
        import requests as _req
        try:
            resp = _req.get(
                f"{_BASE_UPSTOX}/user/get-funds-and-margin",
                params={"segment": "SEC"},
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            resp.raise_for_status()
            return float(
                resp.json()
                    .get("data", {})
                    .get("equity", {})
                    .get("available_margin", 0.0)
            )
        except Exception as exc:
            log.warning("upstox_balance_failed", error=str(exc))
            return 0.0

    def refresh_auth(self) -> None:
        """Upstox tokens are set once at startup; no intraday refresh needed."""


# ---------------------------------------------------------------------------
# Binance (paper mode only — live crypto execution deferred)
# ---------------------------------------------------------------------------

class BinanceBrokerAdapter(BrokerAdapter):
    """
    Binance adapter — paper mode only.

    Live order execution is intentionally not implemented; all calls that
    would modify state raise :exc:`NotImplementedError`. Read-only methods
    (``get_quote``, ``get_balance``) hit the public Binance REST API.

    This skeleton fulfils the ``BrokerAdapter`` contract so that
    ``get_broker_adapter()`` can return a typed broker for crypto workflows
    without triggering live trades.

    Usage
    -----
    When ``data_provider = binance`` and ``paper_trade_mode = false``,
    the factory returns this adapter. Live trading is still blocked by the
    ``is_paper = True`` guard in :class:`~execution.orders.OrderExecutor`.
    """

    _BINANCE_BASE = "https://api.binance.com"

    def __init__(self, initial_capital: float = 500_000.0) -> None:
        self._balance = initial_capital

    @property
    def is_paper(self) -> bool:
        return True  # Binance live execution is deferred

    # ------------------------------------------------------------------
    # Write methods — blocked until live execution is implemented
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        tag: str,
        price: float = 0.0,
        order_type: str = "MARKET",
        intraday: bool = True,
    ) -> str:
        raise NotImplementedError(
            "BinanceBrokerAdapter.place_order: live crypto execution is not yet "
            "implemented. Set PAPER_TRADE_MODE=true for paper trading."
        )

    def cancel_order(self, order_id: str) -> bool:
        log.info("binance_paper_order_cancelled", order_id=order_id)
        return True

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "PAPER_COMPLETE"}

    # ------------------------------------------------------------------
    # Read methods — real Binance public REST
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Return the paper capital balance (live balance requires API key)."""
        return self._balance

    def get_quote(self, symbols: list[str]) -> dict[str, dict[str, float]]:
        """
        Fetch last price for *symbols* from Binance public ticker API.

        Requests are issued in parallel (ThreadPoolExecutor) to cut latency
        from O(N×RTT) to O(RTT).  Returns ``{symbol: {"last_price": float}}``.
        Symbols must use Binance format (e.g. "BTCUSDT").
        Failed lookups are silently omitted.
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor

        def _fetch_one(symbol: str) -> tuple[str, dict[str, float] | None]:
            try:
                resp = requests.get(
                    f"{self._BINANCE_BASE}/api/v3/ticker/price",
                    params={"symbol": symbol.upper()},
                    timeout=5,
                )
                resp.raise_for_status()
                return symbol, {"last_price": float(resp.json()["price"])}
            except Exception as exc:
                log.warning("binance_quote_failed", symbol=symbol, error=str(exc))
                return symbol, None

        result: dict[str, dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as pool:
            for sym, data in pool.map(_fetch_one, symbols):
                if data is not None:
                    result[sym] = data
        return result


# ---------------------------------------------------------------------------
# Paper (dev / crypto / no-broker)
# ---------------------------------------------------------------------------

class PaperBrokerAdapter(BrokerAdapter):
    """No-op adapter for paper trading, local dev, and crypto (no live orders).

    - :meth:`get_balance` returns *initial_capital* from settings.
    - :meth:`place_order` raises ``AssertionError`` — callers must check
      :attr:`is_paper` and write paper trades themselves; the adapter is never
      asked to place orders.
    """

    def __init__(self, initial_capital: float = 500_000.0) -> None:
        self._balance = initial_capital

    @property
    def is_paper(self) -> bool:
        return True

    def place_order(self, symbol, side, quantity, tag, price=0.0,
                    order_type="MARKET", intraday=True) -> str:
        raise AssertionError(
            "PaperBrokerAdapter.place_order should never be called. "
            "The OrderExecutor must handle paper trades before reaching the broker."
        )

    def cancel_order(self, order_id: str) -> bool:
        log.info("paper_order_cancelled", order_id=order_id)
        return True

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return {"order_id": order_id, "status": "PAPER_COMPLETE"}

    def get_balance(self) -> float:
        return self._balance

    def refresh_auth(self) -> None:
        pass   # no auth for paper mode


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_broker_adapter() -> BrokerAdapter:
    """Return the appropriate :class:`BrokerAdapter` based on settings.

    Decision tree::

        paper_trade_mode = true  →  PaperBrokerAdapter  (always, all providers)
        data_provider = kite     →  KiteBrokerAdapter
        data_provider = upstox   →  UpstoxBrokerAdapter
        data_provider = binance  →  PaperBrokerAdapter  (live crypto orders: future)
        (anything else)          →  PaperBrokerAdapter  (safe default)

    Adding a new broker: implement :class:`BrokerAdapter`, then add an
    ``if provider == "mybroker":`` branch here.
    """
    from config.settings import settings

    if settings.paper_trade_mode:
        log.info("broker_adapter_paper", capital=settings.initial_capital)
        return PaperBrokerAdapter(initial_capital=settings.initial_capital)

    provider = settings.data_provider.lower()

    if provider == "kite":
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=settings.kite_api_key)
        if settings.kite_access_token:
            kite.set_access_token(settings.kite_access_token)
        log.info("broker_adapter_kite")
        return KiteBrokerAdapter(kite)

    if provider == "upstox":
        if not settings.upstox_access_token:
            raise RuntimeError(
                "UPSTOX_ACCESS_TOKEN is required for live Upstox trading. "
                "Run the OAuth flow first or set PAPER_TRADE_MODE=true for dev."
            )
        log.info("broker_adapter_upstox")
        return UpstoxBrokerAdapter(access_token=settings.upstox_access_token)

    # binance — paper mode skeleton (live execution deferred)
    if provider == "binance":
        log.info("broker_adapter_binance_paper", capital=settings.initial_capital)
        return BinanceBrokerAdapter(initial_capital=settings.initial_capital)

    raise ValueError(
        f"Unknown data_provider: {provider!r}. "
        "Valid values are 'kite', 'upstox', 'binance'. "
        "Set PAPER_TRADE_MODE=true for development."
    )
