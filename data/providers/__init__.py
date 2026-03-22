"""
Market data provider factory.

Usage::

    from data.providers import get_provider

    provider = get_provider()                     # uses DATA_PROVIDER env var
    df = provider.fetch_historical("RELIANCE", from_date, to_date, "day")
"""
from __future__ import annotations

from .base import OHLCVProvider


def get_provider() -> OHLCVProvider:
    """Return the configured data provider (kite | upstox)."""
    from config.settings import settings

    name = settings.data_provider.lower()

    if name == "kite":
        from .kite import KiteProvider

        if not settings.kite_api_key:
            raise RuntimeError(
                "KITE_API_KEY is required for the kite provider"
            )
        return KiteProvider(
            api_key=settings.kite_api_key,
            access_token=settings.kite_access_token,
        )

    if name == "upstox":
        from .upstox import UpstoxProvider

        if not settings.upstox_api_key:
            raise RuntimeError(
                "UPSTOX_API_KEY is required for the upstox provider"
            )
        return UpstoxProvider(
            api_key=settings.upstox_api_key,
            access_token=settings.upstox_access_token,
        )

    raise ValueError(
        f"Unknown data provider '{name}'. Set DATA_PROVIDER to 'kite' or 'upstox'."
    )


__all__ = ["OHLCVProvider", "get_provider"]
