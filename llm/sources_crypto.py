"""
Crypto news source connectors.

All sources implement the :class:`NewsSource` ABC so the sentiment pipeline
can swap, combine, or extend them without touching call-site code.

Sources (all free-tier, no paid plan required):
    CryptoPanicFetcher  — curated crypto news with crowd vote scores
                          (free API key at https://cryptopanic.com/developers/api/)
    CoinDeskRSS         — institutional-grade editorial (RSS, no auth)
    CoinTelegraphRSS    — high-volume crypto news feed (RSS, no auth)
    DecryptRSS          — consumer-friendly analysis (RSS, no auth)
    RedditCryptoRSS     — crowd sentiment from r/CryptoCurrency & r/Bitcoin
                          (Reddit public RSS, no auth)

merge_and_rank_crypto() — merge, deduplicate, weight by recency and votes,
                          return sorted list ready for FinBERT scoring.

The normalised article dict returned by every source::

    {
        "headline": str,  # title / headline text
        "summary": str,  # body snippet (may be empty)
        "url": str,  # canonical URL (used for dedup)
        "datetime": float,  # unix timestamp (UTC)
        "source": str,  # human-readable source name
        "currencies": list[str],  # e.g. ["BTC", "ETH"] — empty if unknown
        "votes": int,  # crowd importance score (0 if unavailable)
    }
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any

import feedparser
import requests
import structlog

log = structlog.get_logger(__name__)

_REQUEST_TIMEOUT = 10  # seconds for all HTTP calls
_CRYPTOPANIC_BASE = "https://cryptopanic.com/api/free/v1/posts/"


# ---------------------------------------------------------------------------
# Abstract base — every source must implement this contract
# ---------------------------------------------------------------------------


class NewsSource(ABC):
    """Abstract news source.  Implement :meth:`fetch` to plug into the pipeline."""

    @abstractmethod
    def fetch(
        self,
        currencies: list[str] | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        max_items: int = 30,
    ) -> list[dict]:
        """Return up to *max_items* normalised article dicts.

        Args:
            currencies: Optional list of ticker symbols to filter by
                        (e.g. ``["BTC", "ETH"]``).  ``None`` means all news.
            from_ts:    Earliest article unix timestamp to include (inclusive).
            to_ts:      Latest article unix timestamp to include (inclusive).
            max_items:  Maximum number of articles to return.

        Returns:
            List of normalised dicts — see module docstring for the schema.
        """


# ---------------------------------------------------------------------------
# CryptoPanic
# ---------------------------------------------------------------------------


class CryptoPanicFetcher(NewsSource):
    """Fetch curated crypto news from CryptoPanic with crowd vote scores.

    Free API key available at https://cryptopanic.com/developers/api/.
    Free tier: 100 requests / hour.

    Vote scores are included in each article so ``merge_and_rank_crypto`` can
    weight important news higher than low-signal noise.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def fetch(
        self,
        currencies: list[str] | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        max_items: int = 30,
    ) -> list[dict]:
        params: dict[str, Any] = {
            "auth_token": self._api_key,
            "kind": "news",
            "public": "true",
        }
        if currencies:
            params["currencies"] = ",".join(c.upper() for c in currencies)

        try:
            resp = requests.get(
                _CRYPTOPANIC_BASE,
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data: dict = resp.json()
        except Exception as exc:
            log.error("cryptopanic_fetch_error", error=str(exc))
            return []

        results: list[dict] = []
        cutoff_lo = from_ts or 0
        cutoff_hi = to_ts or time.time() + 86400

        for item in data.get("results", []):
            ts = _parse_iso(item.get("published_at", ""))
            if ts < cutoff_lo or ts > cutoff_hi:
                continue

            votes_obj: dict = item.get("votes", {}) or {}
            vote_score = (
                votes_obj.get("important", 0)
                + votes_obj.get("liked", 0)
                - votes_obj.get("disliked", 0)
                - votes_obj.get("toxic", 0)
            )
            currencies_list = [c.get("code", "") for c in (item.get("currencies") or [])]
            results.append(
                {
                    "headline": item.get("title", ""),
                    "summary": "",  # CryptoPanic free tier omits body
                    "url": item.get("url", ""),
                    "datetime": ts,
                    "source": "cryptopanic",
                    "currencies": currencies_list,
                    "votes": max(vote_score, 0),
                }
            )
            if len(results) >= max_items:
                break

        log.debug("cryptopanic_fetched", count=len(results))
        return results


# ---------------------------------------------------------------------------
# RSS helpers
# ---------------------------------------------------------------------------


class _RSSSource(NewsSource):
    """Base class for RSS-backed news sources.

    Subclasses must define :attr:`_FEEDS` (name → URL) and :attr:`_SOURCE_NAME`.
    """

    _FEEDS: dict[str, str] = {}
    _SOURCE_NAME: str = "rss"

    def fetch(
        self,
        currencies: list[str] | None = None,
        from_ts: int | None = None,
        to_ts: int | None = None,
        max_items: int = 30,
    ) -> list[dict]:
        cutoff_lo = from_ts or 0
        cutoff_hi = to_ts or time.time() + 86400

        currency_set = {c.upper() for c in currencies} if currencies else None
        seen_urls: set[str] = set()
        results: list[dict] = []

        for feed_name, url in self._FEEDS.items():
            try:
                feed = feedparser.parse(url)
            except Exception as exc:
                log.error(
                    "rss_parse_error", source=self._SOURCE_NAME, feed=feed_name, error=str(exc)
                )
                continue

            for entry in feed.entries:
                if len(results) >= max_items:
                    break

                item_url: str = getattr(entry, "link", "")
                if item_url in seen_urls:
                    continue

                ts = _parse_rss_date(getattr(entry, "published", ""))
                if ts < cutoff_lo or ts > cutoff_hi:
                    continue

                headline: str = getattr(entry, "title", "")
                summary: str = getattr(entry, "summary", "")

                # Optional currency filter: skip article if headline+summary
                # mentions none of the requested tickers
                if currency_set and not _headline_mentions(headline + " " + summary, currency_set):
                    continue

                seen_urls.add(item_url)
                results.append(
                    {
                        "headline": headline,
                        "summary": summary,
                        "url": item_url,
                        "datetime": ts,
                        "source": self._SOURCE_NAME,
                        "currencies": [],  # RSS feeds don't tag currencies
                        "votes": 0,
                    }
                )

        log.debug("rss_fetched", source=self._SOURCE_NAME, count=len(results))
        return results


def _headline_mentions(text: str, currency_set: set[str]) -> bool:
    """Return True if *text* contains any ticker from *currency_set* as a word."""
    text_upper = text.upper()
    return any(f" {c} " in f" {text_upper} " or f"${c}" in text_upper for c in currency_set)


# ---------------------------------------------------------------------------
# Concrete RSS sources
# ---------------------------------------------------------------------------


class CoinDeskRSS(_RSSSource):
    """CoinDesk — institutional-grade editorial.  No API key required."""

    _FEEDS = {"main": "https://www.coindesk.com/arc/outboundfeeds/rss/"}
    _SOURCE_NAME = "coindesk"


class CoinTelegraphRSS(_RSSSource):
    """CoinTelegraph — high-volume crypto news.  No API key required."""

    _FEEDS = {"main": "https://cointelegraph.com/rss"}
    _SOURCE_NAME = "cointelegraph"


class DecryptRSS(_RSSSource):
    """Decrypt — consumer-friendly analysis.  No API key required."""

    _FEEDS = {"main": "https://decrypt.co/feed"}
    _SOURCE_NAME = "decrypt"


class RedditCryptoRSS(_RSSSource):
    """Reddit crowd sentiment — r/CryptoCurrency and r/Bitcoin hot posts.

    Uses Reddit's public Atom feed (no authentication, no API key).
    Captures narrative sentiment that often precedes price moves.
    """

    _FEEDS = {
        "cryptocurrency": "https://www.reddit.com/r/CryptoCurrency/hot/.rss?limit=25",
        "bitcoin": "https://www.reddit.com/r/Bitcoin/hot/.rss?limit=25",
    }
    _SOURCE_NAME = "reddit"


# ---------------------------------------------------------------------------
# merge_and_rank_crypto
# ---------------------------------------------------------------------------


def merge_and_rank_crypto(
    sources: list[list[dict]],
    hours_lookback: int = 6,
    vote_weight: float = 0.3,
) -> list[dict]:
    """Merge articles from multiple sources into a single ranked list.

    Ranking formula::

        score = recency_score + vote_weight * normalised_votes

    where ``recency_score`` decays linearly from 1.0 (now) to 0.0 (cutoff).

    Args:
        sources:       Output lists from each :class:`NewsSource`.
        hours_lookback: Only include articles published within this window.
                        6 hours is a tighter window than equities (24 h)
                        because crypto markets react faster.
        vote_weight:   How much crowd votes contribute to rank (0.0–1.0).

    Returns:
        Deduplicated, ranked list — most important / most recent first.
    """
    if not sources:
        return []

    now = time.time()
    cutoff = now - hours_lookback * 3600
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

    if not merged:
        return []

    # Normalise votes to [0, 1]
    max_votes = max((a.get("votes", 0) for a in merged), default=1) or 1

    def _score(article: dict) -> float:
        age = now - article.get("datetime", now)
        recency = max(0.0, 1.0 - age / (hours_lookback * 3600))
        votes_norm = article.get("votes", 0) / max_votes
        return recency + vote_weight * votes_norm

    merged.sort(key=_score, reverse=True)
    return merged


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_iso(ts_str: str) -> float:
    """Parse an ISO 8601 datetime string to a unix timestamp. Returns 0 on failure."""
    if not ts_str:
        return 0.0
    try:
        # Handle both "2024-01-15T12:30:00Z" and "2024-01-15T12:30:00+00:00"
        ts_str = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_str).astimezone(UTC).timestamp()
    except Exception:
        return 0.0


def _parse_rss_date(published: str) -> float:
    """Parse an RFC 2822 RSS date string to a unix timestamp. Returns now() on failure."""
    if not published:
        return time.time()
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(published).timestamp()
    except Exception:
        return time.time()
