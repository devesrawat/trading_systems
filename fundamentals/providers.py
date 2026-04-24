"""
Abstract base for fundamentals data providers.

Skeleton implementations for NSE, Screener, Trendlyne. Actual API calls
are future work — this layer defines the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import structlog

from fundamentals.schema import QuarterlyFinancials, Shareholding, Valuations

log = structlog.get_logger(__name__)


class BaseFundamentalsProvider(ABC):
    """Abstract base for fundamentals data sources."""

    @abstractmethod
    def fetch_financials(self, symbol: str) -> QuarterlyFinancials | None:
        """
        Fetch latest quarterly financials.

        Args:
            symbol: Stock symbol (e.g. 'INFY')

        Returns:
            QuarterlyFinancials or None if unavailable
        """

    @abstractmethod
    def fetch_valuations(self, symbol: str) -> Valuations | None:
        """
        Fetch latest valuation metrics.

        Args:
            symbol: Stock symbol

        Returns:
            Valuations or None if unavailable
        """

    @abstractmethod
    def fetch_shareholding(self, symbol: str) -> Shareholding | None:
        """
        Fetch shareholding pattern.

        Args:
            symbol: Stock symbol

        Returns:
            Shareholding or None if unavailable
        """


class NSEProvider(BaseFundamentalsProvider):
    """
    NSE filings provider (skeleton).

    Future: Fetch from NSE portal or integrated BSE API.
    - Quarterly filings: revenue, net income, EPS
    - Shareholding patterns: promoter, institutional, public
    """

    def fetch_financials(self, symbol: str) -> QuarterlyFinancials | None:
        """Fetch from NSE filings — not yet implemented."""
        log.debug("nse_financials_not_implemented", symbol=symbol)
        return None

    def fetch_valuations(self, symbol: str) -> Valuations | None:
        """NSE does not directly provide valuations — use aggregators."""
        return None

    def fetch_shareholding(self, symbol: str) -> Shareholding | None:
        """Fetch from NSE shareholding announcements — not yet implemented."""
        log.debug("nse_shareholding_not_implemented", symbol=symbol)
        return None


class ScreenerProvider(BaseFundamentalsProvider):
    """
    Screener.in API provider (skeleton).

    Future: Use Screener API to fetch:
    - Quarterly financials: revenue, margins, cash flow
    - Valuations: PE, PB, PEG, ROE, ROCE
    - Shareholding patterns
    - Price momentum indicators
    """

    def fetch_financials(self, symbol: str) -> QuarterlyFinancials | None:
        """Fetch from Screener.in — not yet implemented."""
        log.debug("screener_financials_not_implemented", symbol=symbol)
        return None

    def fetch_valuations(self, symbol: str) -> Valuations | None:
        """Fetch from Screener.in — not yet implemented."""
        log.debug("screener_valuations_not_implemented", symbol=symbol)
        return None

    def fetch_shareholding(self, symbol: str) -> Shareholding | None:
        """Fetch from Screener.in — not yet implemented."""
        log.debug("screener_shareholding_not_implemented", symbol=symbol)
        return None


class TrendlyneProvider(BaseFundamentalsProvider):
    """
    Trendlyne API provider (skeleton).

    Future: Use Trendlyne for:
    - High-quality quarterly financials
    - Normalized metrics
    - Score consistency checking
    """

    def fetch_financials(self, symbol: str) -> QuarterlyFinancials | None:
        """Fetch from Trendlyne — not yet implemented."""
        log.debug("trendlyne_financials_not_implemented", symbol=symbol)
        return None

    def fetch_valuations(self, symbol: str) -> Valuations | None:
        """Fetch from Trendlyne — not yet implemented."""
        log.debug("trendlyne_valuations_not_implemented", symbol=symbol)
        return None

    def fetch_shareholding(self, symbol: str) -> Shareholding | None:
        """Fetch from Trendlyne — not yet implemented."""
        log.debug("trendlyne_shareholding_not_implemented", symbol=symbol)
        return None
