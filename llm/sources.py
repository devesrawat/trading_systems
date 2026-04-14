"""
News source connectors.

FinnhubFetcher     — company + market news via Finnhub API
MoneycontrolRSS    — sector news via Moneycontrol RSS feeds
merge_and_rank()   — merge, deduplicate, filter by recency, sort descending
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import feedparser
import finnhub
import structlog

log = structlog.get_logger(__name__)

_RATE_LIMIT_SLEEP = 2.0  # seconds between Finnhub calls (max 30/min on free tier)


# ---------------------------------------------------------------------------
# Finnhub
# ---------------------------------------------------------------------------


class FinnhubFetcher:
    """Fetch company and market news from Finnhub."""

    def __init__(self, api_key: str) -> None:
        self._client = finnhub.Client(api_key=api_key)

    def fetch_news(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
    ) -> list[dict]:
        """
        Fetch company news for *symbol* between unix timestamps.

        Returns list of dicts: {headline, summary, datetime, source, url}
        """
        from_dt = datetime.fromtimestamp(from_ts, tz=UTC).strftime("%Y-%m-%d")
        to_dt = datetime.fromtimestamp(to_ts, tz=UTC).strftime("%Y-%m-%d")

        try:
            raw = self._client.company_news(symbol, _from=from_dt, to=to_dt)
        except Exception as exc:
            log.error("finnhub_fetch_error", symbol=symbol, error=str(exc))
            return []

        return [_normalise_finnhub(item) for item in (raw or [])]

    def fetch_market_news(self, category: str = "general") -> list[dict]:
        """Fetch broad market news for a category (general, forex, crypto, merger)."""
        try:
            raw = self._client.general_news(category, min_id=0)
        except Exception as exc:
            log.error("finnhub_market_news_error", category=category, error=str(exc))
            return []

        return [_normalise_finnhub(item) for item in (raw or [])]


def _normalise_finnhub(item: dict) -> dict:
    return {
        "headline": item.get("headline", ""),
        "summary": item.get("summary", ""),
        "datetime": float(item.get("datetime", 0)),
        "source": item.get("source", ""),
        "url": item.get("url", ""),
    }


# ---------------------------------------------------------------------------
# Moneycontrol RSS
# ---------------------------------------------------------------------------


class MoneycontrolRSS:
    """Parse Moneycontrol RSS feeds for market, economy, and company news."""

    FEEDS: dict[str, str] = {
        "markets": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "economy": "https://www.moneycontrol.com/rss/economy.xml",
        "companies": "https://www.moneycontrol.com/rss/business.xml",
    }

    def fetch(self, feed_name: str, max_items: int = 20) -> list[dict]:
        """
        Parse *feed_name* RSS feed and return up to *max_items* articles.

        Deduplicates by URL before returning.
        """
        url = self.FEEDS[feed_name]  # raises KeyError for unknown feed
        feed = feedparser.parse(url)
        seen_urls: set[str] = set()
        results: list[dict] = []

        for entry in feed.entries[: max_items * 2]:  # fetch extra to survive dedup
            item_url: str = getattr(entry, "link", "")
            if item_url in seen_urls:
                continue
            seen_urls.add(item_url)

            results.append(
                {
                    "headline": getattr(entry, "title", ""),
                    "summary": getattr(entry, "summary", ""),
                    "url": item_url,
                    "datetime": _parse_rss_date(getattr(entry, "published", "")),
                    "source": "moneycontrol",
                }
            )
            if len(results) >= max_items:
                break

        log.debug("rss_fetched", feed=feed_name, items=len(results))
        return results


def _parse_rss_date(published: str) -> float:
    """Parse an RSS date string to a unix timestamp. Returns now() on failure."""
    if not published:
        return time.time()
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(published).timestamp()
    except Exception:
        return time.time()


# ---------------------------------------------------------------------------
# merge_and_rank
# ---------------------------------------------------------------------------


def merge_and_rank(
    sources: list[list[dict]],
    hours_lookback: int = 24,
) -> list[dict]:
    """
    Merge news items from multiple sources, filter to last *hours_lookback* hours,
    deduplicate by URL, and sort by datetime descending (most recent first).
    """
    if not sources:
        return []

    cutoff = time.time() - hours_lookback * 3600
    seen_urls: set[str] = set()
    merged: list[dict] = []

    for source in sources:
        for item in source:
            url = item.get("url", "")
            if url in seen_urls:
                continue
            if item.get("datetime", 0) < cutoff:
                continue
            seen_urls.add(url)
            merged.append(item)

    merged.sort(key=lambda x: x.get("datetime", 0), reverse=True)
    return merged
