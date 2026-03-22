"""
Crypto universe management via CoinGecko (free public API).

:class:`CryptoUniverse` fetches the top-N coins by market cap, applies a
daily-volume filter, and caches the result in Redis (1-hour TTL) to avoid
hammering the free-tier rate limit.

CoinGecko free-tier limits:
    ~30 calls / min without a key  (use User-Agent to stay polite)
    ~50 calls / min with a free key (https://www.coingecko.com/en/api)

Returned instrument dicts follow the same schema as the NSE universe so the
rest of the pipeline can treat them uniformly::

    {
        "symbol":          "BTC",         # short ticker
        "pair":            "BTCUSDT",     # Binance trading pair
        "name":            "Bitcoin",     # full name
        "market_cap_usd":  float,
        "volume_24h_usd":  float,
        "price_usd":       float,
        "rank":            int,           # CoinGecko market cap rank
        "asset_class":     "crypto",
    }
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests
import structlog

from data.redis_keys import RedisKeys
from data.store import get_redis

log = structlog.get_logger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_CACHE_TTL_SECONDS = 3600   # 1 hour — free tier allows ~30 req/min, no need to poll often
_REQUEST_TIMEOUT = 15

# Quote currency for Binance pairs (USDT is the most liquid on Binance)
_QUOTE_CURRENCY = "usdt"

# Well-known stablecoins to exclude from the tradeable universe
_STABLECOIN_IDS = frozenset({
    "tether", "usd-coin", "dai", "binance-usd", "trueusd",
    "first-digital-usd", "usdd", "frax", "pax-dollar",
})


class CryptoUniverse:
    """Fetch and filter the tradeable crypto universe from CoinGecko.

    Usage::

        universe = CryptoUniverse(api_key="optional-free-key")
        instruments = universe.get_tradeable(top_n=30, min_volume_usd=5_000_000)
        # → list of instrument dicts, deduplicated, stablecoins excluded

    The result is cached in Redis for 1 hour.  Call :meth:`refresh` to force
    a fresh fetch regardless of the cache.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({
            "Accept":     "application/json",
            "User-Agent": "nse-trading-system/1.0 (personal algo trader)",
        })
        if api_key:
            self._session.headers["x-cg-demo-api-key"] = api_key

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_tradeable(
        self,
        top_n: int = 50,
        min_volume_usd: float = 5_000_000,
    ) -> list[dict[str, Any]]:
        """Return up to *top_n* liquid coins ordered by market cap.

        Results are served from Redis cache when available.  Call
        :meth:`refresh` to bypass the cache.

        Args:
            top_n:           Maximum number of coins to return.
            min_volume_usd:  Minimum 24-hour trading volume in USD.
                             Filters out illiquid coins that are hard to enter/exit.

        Returns:
            List of instrument dicts — see module docstring for the schema.
        """
        cached = self._load_cache()
        if cached is not None:
            log.debug("crypto_universe_cache_hit", count=len(cached))
            return _apply_filters(cached, top_n, min_volume_usd)

        return self.refresh(top_n=top_n, min_volume_usd=min_volume_usd)

    def refresh(
        self,
        top_n: int = 50,
        min_volume_usd: float = 5_000_000,
    ) -> list[dict[str, Any]]:
        """Fetch a fresh universe from CoinGecko, cache it, and return it.

        Fetches the top 250 coins by market cap (to leave room for filtering)
        then applies stablecoin exclusion and volume filter.
        """
        raw = self._fetch_markets(per_page=250, page=1)
        instruments = [_to_instrument(item) for item in raw if _include(item)]
        self._save_cache(instruments)
        log.info("crypto_universe_refreshed", total_fetched=len(raw), after_filter=len(instruments))
        return _apply_filters(instruments, top_n, min_volume_usd)

    def get_binance_pairs(
        self,
        top_n: int = 50,
        min_volume_usd: float = 5_000_000,
    ) -> list[str]:
        """Return a list of Binance USDT pairs for the tradeable universe.

        Convenience wrapper over :meth:`get_tradeable` that returns the
        ``pair`` field only — ready to feed into :class:`BinanceProvider`::

            provider.register_instruments({p.replace("USDT", ""): p for p in pairs})
        """
        instruments = self.get_tradeable(top_n=top_n, min_volume_usd=min_volume_usd)
        return [i["pair"] for i in instruments]

    def get_instrument_map(
        self,
        top_n: int = 50,
        min_volume_usd: float = 5_000_000,
    ) -> dict[str, str]:
        """Return ``{ticker: binance_pair}`` dict for :meth:`BinanceProvider.register_instruments`."""
        instruments = self.get_tradeable(top_n=top_n, min_volume_usd=min_volume_usd)
        return {i["symbol"]: i["pair"] for i in instruments}

    # ------------------------------------------------------------------ #
    # CoinGecko REST                                                       #
    # ------------------------------------------------------------------ #

    def _fetch_markets(self, per_page: int = 250, page: int = 1) -> list[dict]:
        """Fetch the ``/coins/markets`` endpoint."""
        params: dict[str, Any] = {
            "vs_currency":            "usd",
            "order":                  "market_cap_desc",
            "per_page":               per_page,
            "page":                   page,
            "price_change_percentage": "24h",
            "sparkline":              "false",
        }
        try:
            resp = self._session.get(
                f"{_COINGECKO_BASE}/coins/markets",
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("coingecko_fetch_error", error=str(exc))
            return []

    # ------------------------------------------------------------------ #
    # Redis cache                                                          #
    # ------------------------------------------------------------------ #

    def _load_cache(self) -> list[dict] | None:
        try:
            r = get_redis()
            raw = r.get(RedisKeys.CRYPTO_UNIVERSE)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            log.warning("crypto_universe_cache_read_error", error=str(exc))
            return None

    def _save_cache(self, instruments: list[dict]) -> None:
        try:
            r = get_redis()
            r.setex(RedisKeys.CRYPTO_UNIVERSE, _CACHE_TTL_SECONDS, json.dumps(instruments))
        except Exception as exc:
            log.warning("crypto_universe_cache_write_error", error=str(exc))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _include(item: dict) -> bool:
    """Return False for stablecoins and items with missing market data."""
    if item.get("id", "") in _STABLECOIN_IDS:
        return False
    if item.get("market_cap") is None or item.get("total_volume") is None:
        return False
    return True


def _to_instrument(item: dict) -> dict[str, Any]:
    """Normalise a CoinGecko ``/coins/markets`` entry to the system schema."""
    symbol = (item.get("symbol") or "").upper()
    return {
        "symbol":         symbol,
        "pair":           f"{symbol}{_QUOTE_CURRENCY.upper()}",   # e.g. "BTCUSDT"
        "name":           item.get("name", ""),
        "coingecko_id":   item.get("id", ""),
        "market_cap_usd": float(item.get("market_cap") or 0),
        "volume_24h_usd": float(item.get("total_volume") or 0),
        "price_usd":      float(item.get("current_price") or 0),
        "rank":           int(item.get("market_cap_rank") or 9999),
        "asset_class":    "crypto",
    }


def _apply_filters(
    instruments: list[dict],
    top_n: int,
    min_volume_usd: float,
) -> list[dict]:
    """Apply volume filter and return at most *top_n* instruments."""
    liquid = [i for i in instruments if i.get("volume_24h_usd", 0) >= min_volume_usd]
    # Already sorted by rank (market cap desc) from CoinGecko
    return liquid[:top_n]
