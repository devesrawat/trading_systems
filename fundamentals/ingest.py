"""
Ingest and cache fundamentals data.

Fetches from providers, validates, and caches in Redis with TTL.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from data.store import get_redis
from fundamentals.providers import BaseFundamentalsProvider
from fundamentals.schema import QuarterlyFinancials, Shareholding, Valuations

log = structlog.get_logger(__name__)


class FundamentalsData:
    """Container for all fundamentals data for a symbol."""

    def __init__(
        self,
        symbol: str,
        financials: QuarterlyFinancials | None = None,
        valuations: Valuations | None = None,
        shareholding: Shareholding | None = None,
    ):
        self.symbol = symbol
        self.financials = financials
        self.valuations = valuations
        self.shareholding = shareholding
        self.timestamp = datetime.now(UTC)

    def is_complete(self) -> bool:
        """Check if all data is available."""
        return (
            self.financials is not None
            and self.valuations is not None
            and self.shareholding is not None
        )

    def completeness_ratio(self) -> float:
        """Fraction of available data (0-1)."""
        count = sum(
            [
                self.financials is not None,
                self.valuations is not None,
                self.shareholding is not None,
            ]
        )
        return count / 3.0


def fetch_fundamentals(symbol: str, providers: list[BaseFundamentalsProvider]) -> FundamentalsData:
    """
    Fetch fundamentals from multiple providers.

    Priority:
    1. Try each provider in order
    2. Use first successful result for each data type
    3. Return best-effort combined data

    Args:
        symbol: Stock symbol
        providers: List of provider instances to query

    Returns:
        FundamentalsData with available fields
    """
    data = FundamentalsData(symbol)

    for provider in providers:
        if not data.financials:
            try:
                data.financials = provider.fetch_financials(symbol)
                if data.financials:
                    log.debug(
                        "fetched_financials",
                        symbol=symbol,
                        provider=provider.__class__.__name__,
                    )
            except Exception as e:
                log.warning(
                    "fetch_financials_error",
                    symbol=symbol,
                    provider=provider.__class__.__name__,
                    error=str(e),
                )

        if not data.valuations:
            try:
                data.valuations = provider.fetch_valuations(symbol)
                if data.valuations:
                    log.debug(
                        "fetched_valuations",
                        symbol=symbol,
                        provider=provider.__class__.__name__,
                    )
            except Exception as e:
                log.warning(
                    "fetch_valuations_error",
                    symbol=symbol,
                    provider=provider.__class__.__name__,
                    error=str(e),
                )

        if not data.shareholding:
            try:
                data.shareholding = provider.fetch_shareholding(symbol)
                if data.shareholding:
                    log.debug(
                        "fetched_shareholding",
                        symbol=symbol,
                        provider=provider.__class__.__name__,
                    )
            except Exception as e:
                log.warning(
                    "fetch_shareholding_error",
                    symbol=symbol,
                    provider=provider.__class__.__name__,
                    error=str(e),
                )

        # Early exit if all data found
        if data.is_complete():
            break

    log.info(
        "fundamentals_fetched",
        symbol=symbol,
        completeness=data.completeness_ratio(),
        has_financials=data.financials is not None,
        has_valuations=data.valuations is not None,
        has_shareholding=data.shareholding is not None,
    )

    return data


def _redis_key_fundamentals(symbol: str, data_type: str) -> str:
    """Redis key for fundamentals cache."""
    return f"trading:fundamentals:{symbol.upper()}:{data_type}"


def get_cached(
    symbol: str,
    max_age_days: int = 7,
    providers: list[BaseFundamentalsProvider] | None = None,
) -> FundamentalsData:
    """
    Get fundamentals from cache or fetch if stale.

    Args:
        symbol: Stock symbol
        max_age_days: Cache TTL in days
        providers: Providers to fetch from if cache miss

    Returns:
        FundamentalsData from cache or fresh fetch
    """
    import json

    redis = get_redis()

    # Try to load from cache
    cached_data = {}
    for data_type in ["financials", "valuations", "shareholding"]:
        key = _redis_key_fundamentals(symbol, data_type)
        try:
            cached_json = redis.get(key)
            if cached_json:
                # Check age
                cached = json.loads(cached_json)
                if "timestamp" in cached:
                    cached_time = datetime.fromisoformat(cached["timestamp"])
                    age = (datetime.now(UTC) - cached_time).days
                    if age <= max_age_days:
                        cached_data[data_type] = cached
                        log.debug("cache_hit", symbol=symbol, data_type=data_type, age_days=age)
                    else:
                        log.debug("cache_expired", symbol=symbol, data_type=data_type, age_days=age)
        except Exception as e:
            log.warning("cache_read_error", symbol=symbol, data_type=data_type, error=str(e))

    # If we have all cached data, return it
    if len(cached_data) == 3:
        data = FundamentalsData(symbol)
        try:
            if "financials" in cached_data:
                data.financials = QuarterlyFinancials(**cached_data["financials"])
            if "valuations" in cached_data:
                data.valuations = Valuations(**cached_data["valuations"])
            if "shareholding" in cached_data:
                data.shareholding = Shareholding(**cached_data["shareholding"])
            return data
        except Exception as e:
            log.warning("cache_parse_error", symbol=symbol, error=str(e))

    # Cache miss or incomplete — fetch fresh if providers available
    if providers:
        return fetch_fundamentals(symbol, providers)

    # Return partial cached data or empty
    data = FundamentalsData(symbol)
    try:
        if "financials" in cached_data:
            data.financials = QuarterlyFinancials(**cached_data["financials"])
        if "valuations" in cached_data:
            data.valuations = Valuations(**cached_data["valuations"])
        if "shareholding" in cached_data:
            data.shareholding = Shareholding(**cached_data["shareholding"])
    except Exception as e:
        log.warning("cache_partial_parse_error", symbol=symbol, error=str(e))

    return data


def store_in_cache(symbol: str, fundamentals_data: FundamentalsData, ttl_days: int = 7) -> None:
    """
    Store fundamentals in Redis cache.

    Args:
        symbol: Stock symbol
        fundamentals_data: FundamentalsData object
        ttl_days: Time-to-live in days
    """
    import json

    redis = get_redis()
    ttl_seconds = ttl_days * 24 * 3600

    if fundamentals_data.financials:
        try:
            key = _redis_key_fundamentals(symbol, "financials")
            json_data = json.dumps(
                fundamentals_data.financials.model_dump(),
                default=str,
            )
            redis.setex(key, ttl_seconds, json_data)
            log.debug("cached_financials", symbol=symbol, ttl_days=ttl_days)
        except Exception as e:
            log.warning("cache_write_error", symbol=symbol, data_type="financials", error=str(e))

    if fundamentals_data.valuations:
        try:
            key = _redis_key_fundamentals(symbol, "valuations")
            json_data = json.dumps(
                fundamentals_data.valuations.model_dump(),
                default=str,
            )
            redis.setex(key, ttl_seconds, json_data)
            log.debug("cached_valuations", symbol=symbol, ttl_days=ttl_days)
        except Exception as e:
            log.warning("cache_write_error", symbol=symbol, data_type="valuations", error=str(e))

    if fundamentals_data.shareholding:
        try:
            key = _redis_key_fundamentals(symbol, "shareholding")
            json_data = json.dumps(
                fundamentals_data.shareholding.model_dump(),
                default=str,
            )
            redis.setex(key, ttl_seconds, json_data)
            log.debug("cached_shareholding", symbol=symbol, ttl_days=ttl_days)
        except Exception as e:
            log.warning("cache_write_error", symbol=symbol, data_type="shareholding", error=str(e))

    # Bridge for Wealth Architect Scanner
    if fundamentals_data.valuations:
        try:
            wa_key = f"FUND:{symbol.upper()}"
            wa_data = {
                "pe": fundamentals_data.valuations.pe_ratio,
                "roe": fundamentals_data.valuations.roe,
                "sector": fundamentals_data.valuations.sector or "Unknown",
                "sector_avg_pe": fundamentals_data.valuations.sector_avg_pe,
            }
            redis.setex(wa_key, ttl_seconds, json.dumps(wa_data))
            log.info("cached_for_wealth_architect", symbol=symbol)
        except Exception as e:
            log.warning("wa_bridge_error", symbol=symbol, error=str(e))


def store_in_db(symbol: str, fundamentals_data: FundamentalsData) -> None:
    """
    Store fundamentals in TimescaleDB.

    Placeholder for future DB implementation.
    Currently just logs that it was called.

    Args:
        symbol: Stock symbol
        fundamentals_data: FundamentalsData to store
    """
    log.info(
        "store_in_db_not_implemented",
        symbol=symbol,
        completeness=fundamentals_data.completeness_ratio(),
    )
