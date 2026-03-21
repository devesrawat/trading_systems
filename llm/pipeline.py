"""
Sentiment pipeline orchestrator.

fetch news → FinBERT score → Redis cache → TimescaleDB
Runs every 30 minutes during NSE market hours (9:15–15:30 IST).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import structlog
from sqlalchemy import text

from config.settings import settings
from data.store import get_engine, get_redis
from llm.sentiment import FinBERTScorer
from llm.sources import FinnhubFetcher, MoneycontrolRSS, merge_and_rank

log = structlog.get_logger(__name__)

_MARKET_OPEN_HOUR = 9
_MARKET_OPEN_MIN = 15
_MARKET_CLOSE_HOUR = 15
_MARKET_CLOSE_MIN = 30

_INSERT_SQL = text("""
    INSERT INTO sentiment_scores (time, symbol, score, headline_count)
    VALUES (:time, :symbol, :score, :headline_count)
""")


# ---------------------------------------------------------------------------
# DB helpers (module-level so tests can patch them easily)
# ---------------------------------------------------------------------------

def _write_scores(scores: dict[str, float], headline_counts: dict[str, int] | None = None) -> None:
    """Bulk-insert sentiment scores into TimescaleDB."""
    if not scores:
        return

    now = datetime.now(tz=timezone.utc)
    rows = [
        {
            "time": now,
            "symbol": symbol,
            "score": float(score),
            "headline_count": (headline_counts or {}).get(symbol, 0),
        }
        for symbol, score in scores.items()
    ]

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(_INSERT_SQL, rows)
        conn.commit()
    log.debug("sentiment_scores_written", count=len(rows))


def _fetch_latest_score_from_db(symbol: str) -> float | None:
    """Query TimescaleDB for the most recent sentiment score for *symbol*."""
    query = text("""
        SELECT score FROM sentiment_scores
        WHERE symbol = :symbol
        ORDER BY time DESC
        LIMIT 1
    """)
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(query, {"symbol": symbol}).fetchone()
    if row is None:
        return None
    return float(row[0])


# ---------------------------------------------------------------------------
# SentimentPipeline
# ---------------------------------------------------------------------------

class SentimentPipeline:
    """
    Orchestrates: fetch headlines → score with FinBERT → cache + persist.
    """

    def __init__(self) -> None:
        self._scorer = FinBERTScorer()
        self._fetcher = FinnhubFetcher(api_key=settings.finnhub_api_key or "")
        self._rss = MoneycontrolRSS()

    # ------------------------------------------------------------------
    # run_daily
    # ------------------------------------------------------------------

    def run_daily(self, universe: list[str]) -> dict[str, float]:
        """
        Score every symbol in *universe* against the last 24h of news.

        Returns {symbol: score} for all symbols.
        Symbols with no news return 0.0 (neutral).
        Errors per-symbol are caught and logged — never crash the full run.
        """
        if not universe:
            return {}

        scores: dict[str, float] = {}
        headline_counts: dict[str, int] = {}
        now = int(time.time())
        yesterday = now - 86_400

        for symbol in universe:
            try:
                headlines = self._fetch_headlines(symbol, from_ts=yesterday, to_ts=now)
                headline_counts[symbol] = len(headlines)

                if not headlines:
                    scores[symbol] = 0.0
                    continue

                today_str = datetime.utcnow().strftime("%Y-%m-%d")
                score = self._scorer.score_aggregate_cached(
                    [h["headline"] for h in headlines],
                    symbol=symbol,
                    date=today_str,
                )
                scores[symbol] = score

            except Exception as exc:
                log.error("sentiment_run_error", symbol=symbol, error=str(exc))
                scores[symbol] = 0.0

        _write_scores(scores, headline_counts)
        log.info("sentiment_run_daily_complete", symbols=len(scores))
        return scores

    # ------------------------------------------------------------------
    # get_latest_score
    # ------------------------------------------------------------------

    def get_latest_score(self, symbol: str) -> float:
        """
        Return the most recent sentiment score for *symbol*.

        Lookup order: Redis cache → TimescaleDB → 0.0 (neutral fallback).
        """
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        cache_key = f"sentiment:{symbol}:{today_str}"

        r = get_redis()
        cached = r.get(cache_key)
        if cached is not None:
            return float(cached)

        db_score = _fetch_latest_score_from_db(symbol)
        if db_score is not None:
            return db_score

        return 0.0

    # ------------------------------------------------------------------
    # run_continuous
    # ------------------------------------------------------------------

    async def run_continuous(
        self,
        universe: list[str],
        interval_minutes: int = 30,
    ) -> None:
        """
        Async loop: refresh sentiment every *interval_minutes* during market hours.
        Runs indefinitely until cancelled.
        """
        log.info("sentiment_continuous_start", interval_min=interval_minutes)
        while True:
            if _is_market_hours():
                self.run_daily(universe)
            else:
                log.debug("sentiment_skipped_outside_market_hours")
            await asyncio.sleep(interval_minutes * 60)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_headlines(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
    ) -> list[dict]:
        """Fetch from Finnhub + Moneycontrol, merge, deduplicate."""
        finnhub_news = self._fetcher.fetch_news(symbol, from_ts=from_ts, to_ts=to_ts)
        market_news = self._rss.fetch("markets", max_items=10)
        return merge_and_rank([finnhub_news, market_news], hours_lookback=24)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_market_hours() -> bool:
    """Return True if current IST time is within NSE normal market hours."""
    from zoneinfo import ZoneInfo
    now_ist = datetime.now(tz=ZoneInfo("Asia/Kolkata"))
    open_time = now_ist.replace(
        hour=_MARKET_OPEN_HOUR, minute=_MARKET_OPEN_MIN, second=0, microsecond=0
    )
    close_time = now_ist.replace(
        hour=_MARKET_CLOSE_HOUR, minute=_MARKET_CLOSE_MIN, second=0, microsecond=0
    )
    return open_time <= now_ist <= close_time
