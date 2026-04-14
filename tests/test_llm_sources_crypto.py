"""Unit tests for llm/sources_crypto.py — mocks HTTP/RSS, no live network calls."""

import time
from unittest.mock import MagicMock, patch

import pytest

from llm.sources_crypto import (
    CoinDeskRSS,
    CoinTelegraphRSS,
    CryptoPanicFetcher,
    DecryptRSS,
    NewsSource,
    RedditCryptoRSS,
    _parse_iso,
    _parse_rss_date,
    merge_and_rank_crypto,
)

# ---------------------------------------------------------------------------
# NewsSource — ABC contract
# ---------------------------------------------------------------------------


class TestNewsSourceABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            NewsSource()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_fetch(self):
        class Incomplete(NewsSource):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_can_instantiate(self):
        class Complete(NewsSource):
            def fetch(self, currencies=None, from_ts=None, to_ts=None, max_items=30):
                return []

        assert Complete().fetch() == []


# ---------------------------------------------------------------------------
# CryptoPanicFetcher
# ---------------------------------------------------------------------------


def _cp_result(
    title: str = "BTC rallies",
    url: str = "https://cp.example/1",
    published_at: str = "2024-01-15T12:00:00Z",
    currencies: list | None = None,
    votes: dict | None = None,
) -> dict:
    return {
        "title": title,
        "url": url,
        "published_at": published_at,
        "currencies": [{"code": c} for c in (currencies or ["BTC"])],
        "votes": votes or {"important": 5, "liked": 10, "disliked": 1, "toxic": 0},
    }


class TestCryptoPanicFetcher:
    def _mock_response(self, results: list) -> MagicMock:
        resp = MagicMock()
        resp.json.return_value = {"results": results}
        resp.raise_for_status = MagicMock()
        return resp

    @patch("llm.sources_crypto.requests.get")
    def test_returns_normalised_dicts(self, mock_get):
        mock_get.return_value = self._mock_response([_cp_result()])
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch()
        assert len(result) == 1
        item = result[0]
        assert item["headline"] == "BTC rallies"
        assert item["source"] == "cryptopanic"
        assert item["currencies"] == ["BTC"]

    @patch("llm.sources_crypto.requests.get")
    def test_vote_score_computed_correctly(self, mock_get):
        # important=5, liked=10, disliked=1, toxic=0 → score = 14
        mock_get.return_value = self._mock_response(
            [_cp_result(votes={"important": 5, "liked": 10, "disliked": 1, "toxic": 0})]
        )
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch()
        assert result[0]["votes"] == 14

    @patch("llm.sources_crypto.requests.get")
    def test_negative_vote_score_clamped_to_zero(self, mock_get):
        mock_get.return_value = self._mock_response(
            [_cp_result(votes={"important": 0, "liked": 0, "disliked": 10, "toxic": 5})]
        )
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch()
        assert result[0]["votes"] == 0

    @patch("llm.sources_crypto.requests.get")
    def test_respects_max_items(self, mock_get):
        items = [_cp_result(url=f"https://cp.example/{i}") for i in range(10)]
        mock_get.return_value = self._mock_response(items)
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch(max_items=3)
        assert len(result) <= 3

    @patch("llm.sources_crypto.requests.get")
    def test_from_ts_filter_excludes_old_articles(self, mock_get):
        old_ts = "2020-01-01T00:00:00Z"
        mock_get.return_value = self._mock_response([_cp_result(published_at=old_ts)])
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch(from_ts=int(time.time()) - 3600)  # only last 1h
        assert result == []

    @patch("llm.sources_crypto.requests.get")
    def test_api_error_returns_empty_list(self, mock_get):
        mock_get.side_effect = Exception("network error")
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch()
        assert result == []

    @patch("llm.sources_crypto.requests.get")
    def test_required_fields_present(self, mock_get):
        mock_get.return_value = self._mock_response([_cp_result()])
        fetcher = CryptoPanicFetcher(api_key="testkey")
        result = fetcher.fetch()
        required = {"headline", "summary", "url", "datetime", "source", "currencies", "votes"}
        assert required.issubset(set(result[0].keys()))

    @patch("llm.sources_crypto.requests.get")
    def test_currency_filter_passed_to_api(self, mock_get):
        mock_get.return_value = self._mock_response([])
        fetcher = CryptoPanicFetcher(api_key="testkey")
        fetcher.fetch(currencies=["BTC", "ETH"])
        call_params = mock_get.call_args[1]["params"]
        assert call_params["currencies"] == "BTC,ETH"


# ---------------------------------------------------------------------------
# RSS sources — shared behaviour via _RSSSource
# ---------------------------------------------------------------------------


def _fake_rss_feed(n: int = 3, base_url: str = "https://example.com") -> MagicMock:
    feed = MagicMock()
    feed.entries = []
    for i in range(n):
        entry = MagicMock()
        entry.title = f"Crypto headline {i}"
        entry.summary = f"Summary of article {i}"
        entry.link = f"{base_url}/article/{i}"
        entry.published = "Mon, 15 Jan 2024 10:00:00 +0000"
        feed.entries.append(entry)
    return feed


class TestRSSSources:
    @pytest.mark.parametrize(
        "source_cls", [CoinDeskRSS, CoinTelegraphRSS, DecryptRSS, RedditCryptoRSS]
    )
    def test_returns_list_of_dicts(self, source_cls):
        with patch("llm.sources_crypto.feedparser.parse", return_value=_fake_rss_feed(3)):
            result = source_cls().fetch()
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.parametrize("source_cls", [CoinDeskRSS, CoinTelegraphRSS, DecryptRSS])
    def test_required_fields_present(self, source_cls):
        with patch("llm.sources_crypto.feedparser.parse", return_value=_fake_rss_feed(1)):
            result = source_cls().fetch()
        required = {"headline", "summary", "url", "datetime", "source", "currencies", "votes"}
        assert required.issubset(set(result[0].keys()))

    @pytest.mark.parametrize("source_cls", [CoinDeskRSS, CoinTelegraphRSS, DecryptRSS])
    def test_respects_max_items(self, source_cls):
        with patch("llm.sources_crypto.feedparser.parse", return_value=_fake_rss_feed(20)):
            result = source_cls().fetch(max_items=5)
        assert len(result) <= 5

    @pytest.mark.parametrize("source_cls", [CoinDeskRSS, CoinTelegraphRSS, DecryptRSS])
    def test_deduplicates_by_url(self, source_cls):
        feed = _fake_rss_feed(3)
        feed.entries[1].link = feed.entries[0].link  # duplicate URL
        with patch("llm.sources_crypto.feedparser.parse", return_value=feed):
            result = source_cls().fetch()
        urls = [item["url"] for item in result]
        assert len(urls) == len(set(urls))

    @pytest.mark.parametrize("source_cls", [CoinDeskRSS, CoinTelegraphRSS, DecryptRSS])
    def test_votes_always_zero(self, source_cls):
        """RSS sources don't have vote data."""
        with patch("llm.sources_crypto.feedparser.parse", return_value=_fake_rss_feed(1)):
            result = source_cls().fetch()
        assert result[0]["votes"] == 0

    def test_reddit_fetches_multiple_feeds(self):
        """RedditCryptoRSS has 2 feeds — both are fetched."""
        with patch(
            "llm.sources_crypto.feedparser.parse", return_value=_fake_rss_feed(2)
        ) as mock_parse:
            RedditCryptoRSS().fetch()
        assert mock_parse.call_count == len(RedditCryptoRSS._FEEDS)

    def test_rss_parse_error_returns_empty_not_raises(self):
        with patch("llm.sources_crypto.feedparser.parse", side_effect=Exception("parse error")):
            result = CoinDeskRSS().fetch()
        assert result == []

    def test_currency_filter_skips_unrelated_articles(self):
        feed = _fake_rss_feed(2)
        feed.entries[0].title = "Bitcoin hits new high BTC surge"
        feed.entries[1].title = "General market overview with no crypto mention"
        with patch("llm.sources_crypto.feedparser.parse", return_value=feed):
            result = CoinDeskRSS().fetch(currencies=["BTC"])
        assert len(result) == 1
        assert "BTC" in result[0]["headline"].upper()

    def test_old_articles_filtered_by_from_ts(self):
        feed = _fake_rss_feed(1)
        feed.entries[0].published = "Mon, 01 Jan 2020 00:00:00 +0000"
        with patch("llm.sources_crypto.feedparser.parse", return_value=feed):
            result = CoinDeskRSS().fetch(from_ts=int(time.time()) - 3600)
        assert result == []


# ---------------------------------------------------------------------------
# merge_and_rank_crypto
# ---------------------------------------------------------------------------


def _make_articles(
    n: int, hours_old: float = 1.0, votes: int = 0, url_prefix: str = "a"
) -> list[dict]:
    now = time.time()
    return [
        {
            "headline": f"Article {url_prefix}-{i}",
            "summary": "",
            "url": f"https://example.com/{url_prefix}-{i}",
            "datetime": now - hours_old * 3600 - i,
            "source": "test",
            "currencies": [],
            "votes": votes,
        }
        for i in range(n)
    ]


class TestMergeAndRankCrypto:
    def test_merges_multiple_sources(self):
        a = _make_articles(3, hours_old=1, url_prefix="a")
        b = _make_articles(3, hours_old=2, url_prefix="b")
        result = merge_and_rank_crypto([a, b], hours_lookback=6)
        assert len(result) == 6

    def test_filters_old_articles(self):
        recent = _make_articles(3, hours_old=1)
        old = _make_articles(3, hours_old=24)
        result = merge_and_rank_crypto([recent, old], hours_lookback=6)
        assert len(result) == 3

    def test_deduplicates_across_sources(self):
        article = _make_articles(1, url_prefix="dup")[0]
        result = merge_and_rank_crypto([[article], [article]])
        assert len(result) == 1

    def test_high_vote_articles_ranked_above_low_vote(self):
        """An older article with many votes should beat a newer article with zero votes."""
        now = time.time()
        high_vote = [
            {
                "headline": "Important news",
                "url": "https://example.com/high",
                "datetime": now - 3 * 3600,  # 3h old
                "source": "test",
                "summary": "",
                "currencies": [],
                "votes": 100,
            }
        ]
        low_vote = [
            {
                "headline": "Minor update",
                "url": "https://example.com/low",
                "datetime": now - 1 * 3600,  # 1h old (more recent)
                "source": "test",
                "summary": "",
                "currencies": [],
                "votes": 0,
            }
        ]
        result = merge_and_rank_crypto([high_vote, low_vote], hours_lookback=6, vote_weight=1.0)
        assert result[0]["url"] == "https://example.com/high"

    def test_empty_sources_returns_empty(self):
        assert merge_and_rank_crypto([]) == []

    def test_all_old_articles_returns_empty(self):
        old = _make_articles(5, hours_old=48)
        result = merge_and_rank_crypto([old], hours_lookback=6)
        assert result == []

    def test_returns_list(self):
        result = merge_and_rank_crypto([_make_articles(2)])
        assert isinstance(result, list)

    def test_shorter_lookback_than_equities(self):
        """Default crypto lookback is 6h (equities use 24h) — articles at 7h must be excluded."""
        articles_7h = _make_articles(3, hours_old=7)
        result = merge_and_rank_crypto([articles_7h], hours_lookback=6)
        assert result == []


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


class TestParseHelpers:
    def test_parse_iso_utc_z(self):
        ts = _parse_iso("2024-01-15T12:00:00Z")
        assert ts > 0

    def test_parse_iso_with_offset(self):
        ts = _parse_iso("2024-01-15T12:00:00+00:00")
        assert ts > 0

    def test_parse_iso_empty_returns_zero(self):
        assert _parse_iso("") == 0.0

    def test_parse_iso_invalid_returns_zero(self):
        assert _parse_iso("not-a-date") == 0.0

    def test_parse_rss_date_valid(self):
        ts = _parse_rss_date("Mon, 15 Jan 2024 12:00:00 +0000")
        assert ts > 0

    def test_parse_rss_date_empty_returns_approx_now(self):
        before = time.time()
        ts = _parse_rss_date("")
        after = time.time()
        assert before <= ts <= after

    def test_parse_rss_date_invalid_returns_approx_now(self):
        before = time.time()
        ts = _parse_rss_date("garbage")
        after = time.time()
        assert before <= ts <= after
