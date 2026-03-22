"""
Market data provider factory.

Usage::

    from data.providers import get_provider, get_crypto_provider

    # Equity provider (NSE via Kite or Upstox)
    provider = get_provider()
    df = provider.fetch_historical("RELIANCE", from_date, to_date, "day")

    # Crypto provider (Binance public API — no API key required)
    crypto = get_crypto_provider()
    crypto.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
    df = crypto.fetch_historical("BTC", from_date, to_date, "day")

The equity provider is selected via the ``DATA_PROVIDER`` env var (kite | upstox).
The crypto provider is always Binance (the only free, no-auth option that covers
all major pairs with full WebSocket support).
"""
from __future__ import annotations

from .base import OHLCVProvider


def get_provider() -> OHLCVProvider:
    """Return the configured equity data provider (kite | upstox)."""
    from config.settings import settings

    name = settings.data_provider.lower()

    if name == "kite":
        from .kite import KiteProvider

        if not settings.kite_api_key:
            raise RuntimeError("KITE_API_KEY is required for the kite provider")
        return KiteProvider(
            api_key=settings.kite_api_key,
            access_token=settings.kite_access_token,
        )

    if name == "upstox":
        from .upstox import UpstoxProvider

        if not settings.upstox_api_key:
            raise RuntimeError("UPSTOX_API_KEY is required for the upstox provider")
        return UpstoxProvider(
            api_key=settings.upstox_api_key,
            access_token=settings.upstox_access_token,
        )

    raise ValueError(
        f"Unknown data provider '{name}'. Set DATA_PROVIDER to 'kite' or 'upstox'."
    )


def get_crypto_provider() -> OHLCVProvider:
    """Return a :class:`BinanceProvider` instance for crypto market data.

    No API key is required for market data.  Optionally set
    ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET`` in ``.env`` for future
    order-placement support.

    The caller is responsible for registering instruments::

        provider = get_crypto_provider()
        provider.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
    """
    from config.settings import settings
    from .binance import BinanceProvider

    return BinanceProvider(
        api_key=settings.binance_api_key,
        api_secret=settings.binance_api_secret,
    )


__all__ = ["OHLCVProvider", "get_provider", "get_crypto_provider"]
