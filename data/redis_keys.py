"""
Centralized Redis key constants.

All modules that read from or write to Redis must use these helpers.
Centralizing prevents key collisions, inconsistent naming, and makes
schema changes a single-file edit.

Key namespace: ``trading:{domain}:{resource}[:{id}]``
"""
from __future__ import annotations


class RedisKeys:
    """Namespace for every Redis key used in the trading system."""

    # ------------------------------------------------------------------
    # Live tick cache  (TTL: 5 s — data/store.py)
    # ------------------------------------------------------------------

    @staticmethod
    def tick(token: int) -> str:
        """``trading:tick:{token}``"""
        return f"trading:tick:{token}"

    # ------------------------------------------------------------------
    # Risk state  (persistent — survives restarts)
    # ------------------------------------------------------------------

    CIRCUIT_STATE: str = "trading:risk:circuit:state"
    PORTFOLIO_STATE: str = "trading:risk:portfolio:state"

    # ------------------------------------------------------------------
    # Sentiment scores  (TTL: 1 h — llm/sentiment.py)
    # ------------------------------------------------------------------

    @staticmethod
    def sentiment(symbol: str, date: str) -> str:
        """``trading:sentiment:{symbol}:{date}``"""
        return f"trading:sentiment:{symbol}:{date}"

    # ------------------------------------------------------------------
    # Scanner / signal state  (TTL: varies — data/live_feed.py)
    # ------------------------------------------------------------------

    @staticmethod
    def vcp_pivot(symbol: str) -> str:
        """``trading:signal:vcp:pivot:{symbol}``"""
        return f"trading:signal:vcp:pivot:{symbol}"

    @staticmethod
    def bar_day(symbol: str) -> str:
        """``trading:bar:day:{symbol}``"""
        return f"trading:bar:day:{symbol}"

    # ------------------------------------------------------------------
    # Auth tokens  (TTL: 24 h — data/ingest.py)
    # ------------------------------------------------------------------

    KITE_ACCESS_TOKEN: str = "trading:auth:kite:access_token"
