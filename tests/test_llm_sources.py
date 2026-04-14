"""Unit tests for llm/sources.py — mocks HTTP/RSS, no live network calls."""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm.sources import FinnhubFetcher, MoneycontrolRSS, merge_and_rank

# ---------------------------------------------------------------------------
# FinnhubFetcher
# ---------------------------------------------------------------------------


def _finnhub_news_item(ts_offset: int = 0) -> dict:
    return {
        "headline": "Test headline",
        "summary": "Test summary",
        "datetime": int(time.time()) - ts_offset,
        "source": "Reuters",
        "url": f"https://example.com/news/{ts_offset}",
    }


class TestFinnhubFetcher:
    def test_fetch_news_returns_list_of_dicts(self):
        with patch("llm.sources.finnhub.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.company_news.return_value = [_finnhub_news_item()]
            mock_cls.return_value = mock_client

            fetcher = FinnhubFetcher(api_key="test_key")
            result = fetcher.fetch_news("RELIANCE", from_ts=0, to_ts=int(time.time()))

        assert isinstance(result, list)
        assert len(result) == 1
        assert "headline" in result[0]
        assert "datetime" in result[0]

    def test_fetch_news_required_fields_present(self):
        with patch("llm.sources.finnhub.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.company_news.return_value = [_finnhub_news_item()]
            mock_cls.return_value = mock_client

            fetcher = FinnhubFetcher(api_key="test_key")
            result = fetcher.fetch_news("TCS", from_ts=0, to_ts=int(time.time()))

        required = {"headline", "summary", "datetime", "source", "url"}
        assert required.issubset(set(result[0].keys()))

    def test_fetch_news_empty_response(self):
        with patch("llm.sources.finnhub.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.company_news.return_value = []
            mock_cls.return_value = mock_client

            fetcher = FinnhubFetcher(api_key="test_key")
            result = fetcher.fetch_news("UNKNOWN", from_ts=0, to_ts=int(time.time()))

        assert result == []

    def test_fetch_market_news_returns_list(self):
        with patch("llm.sources.finnhub.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.general_news.return_value = [_finnhub_news_item()]
            mock_cls.return_value = mock_client

            fetcher = FinnhubFetcher(api_key="test_key")
            result = fetcher.fetch_market_news(category="general")

        assert isinstance(result, list)

    def test_api_error_returns_empty_list(self):
        with patch("llm.sources.finnhub.Client") as mock_cls:
            mock_client = MagicMock()
            mock_client.company_news.side_effect = Exception("API error")
            mock_cls.return_value = mock_client

            fetcher = FinnhubFetcher(api_key="test_key")
            result = fetcher.fetch_news("ERR", from_ts=0, to_ts=int(time.time()))

        assert result == []


# ---------------------------------------------------------------------------
# MoneycontrolRSS
# ---------------------------------------------------------------------------


def _fake_feed(n: int = 3) -> MagicMock:
    feed = MagicMock()
    feed.entries = []
    for i in range(n):
        entry = MagicMock()
        entry.title = f"Market news headline {i}"
        entry.summary = f"Summary {i}"
        entry.link = f"https://moneycontrol.com/news/{i}"
        entry.published = "Mon, 15 Jan 2024 10:00:00 +0530"
        feed.entries.append(entry)
    return feed


class TestMoneycontrolRSS:
    def test_fetch_returns_list_of_dicts(self):
        with patch("llm.sources.feedparser.parse", return_value=_fake_feed(3)):
            rss = MoneycontrolRSS()
            result = rss.fetch("markets")

        assert isinstance(result, list)
        assert len(result) == 3

    def test_fetch_required_fields(self):
        with patch("llm.sources.feedparser.parse", return_value=_fake_feed(1)):
            rss = MoneycontrolRSS()
            result = rss.fetch("markets")

        item = result[0]
        assert "headline" in item
        assert "url" in item
        assert "datetime" in item

    def test_fetch_respects_max_items(self):
        with patch("llm.sources.feedparser.parse", return_value=_fake_feed(10)):
            rss = MoneycontrolRSS()
            result = rss.fetch("markets", max_items=3)

        assert len(result) <= 3

    def test_deduplicates_by_url(self):
        feed = _fake_feed(3)
        # Make two entries with same URL
        feed.entries[1].link = feed.entries[0].link
        with patch("llm.sources.feedparser.parse", return_value=feed):
            rss = MoneycontrolRSS()
            result = rss.fetch("markets")

        urls = [item["url"] for item in result]
        assert len(urls) == len(set(urls))

    def test_unknown_feed_raises(self):
        rss = MoneycontrolRSS()
        with pytest.raises(KeyError):
            rss.fetch("nonexistent_feed")

    def test_feeds_dict_contains_required_keys(self):
        rss = MoneycontrolRSS()
        for key in ("markets", "economy", "companies"):
            assert key in rss.FEEDS


# ---------------------------------------------------------------------------
# merge_and_rank
# ---------------------------------------------------------------------------


class TestMergeAndRank:
    def _make_items(self, n: int, hours_old: float = 1.0) -> list[dict]:
        now = time.time()
        return [
            {
                "headline": f"News {i}",
                "url": f"https://example.com/{hours_old}-{i}",
                "datetime": now - hours_old * 3600 - i,
                "source": "test",
                "summary": "",
            }
            for i in range(n)
        ]

    def test_merges_multiple_sources(self):
        source_a = self._make_items(3, hours_old=1)
        source_b = self._make_items(3, hours_old=2)
        result = merge_and_rank([source_a, source_b], hours_lookback=24)
        assert len(result) == 6

    def test_filters_old_items(self):
        recent = self._make_items(3, hours_old=1)
        old = self._make_items(3, hours_old=48)
        result = merge_and_rank([recent, old], hours_lookback=24)
        assert len(result) == 3

    def test_sorted_by_datetime_desc(self):
        source = self._make_items(5, hours_old=1)
        result = merge_and_rank([source], hours_lookback=24)
        datetimes = [item["datetime"] for item in result]
        assert datetimes == sorted(datetimes, reverse=True)

    def test_empty_sources_returns_empty(self):
        result = merge_and_rank([], hours_lookback=24)
        assert result == []

    def test_deduplicates_across_sources(self):
        item = self._make_items(1, hours_old=1)[0]
        result = merge_and_rank([[item], [item]], hours_lookback=24)
        assert len(result) == 1
