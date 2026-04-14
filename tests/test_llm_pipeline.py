"""Unit tests for llm/pipeline.py — mocks scorer, sources, and DB."""

import time
from unittest.mock import MagicMock, patch

from llm.pipeline import SentimentPipeline

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pipeline(mock_scorer=None, mock_fetcher=None) -> SentimentPipeline:
    scorer = mock_scorer or MagicMock()
    scorer.score_aggregate_cached.return_value = 0.5
    fetcher = mock_fetcher or MagicMock()
    fetcher.fetch_news.return_value = [
        {
            "headline": "Test headline",
            "url": "https://example.com/1",
            "datetime": time.time(),
            "source": "test",
            "summary": "",
        },
    ]
    with (
        patch("llm.pipeline.FinnhubFetcher", return_value=fetcher),
        patch("llm.pipeline.FinBERTScorer", return_value=scorer),
    ):
        return SentimentPipeline()


# ---------------------------------------------------------------------------
# run_daily
# ---------------------------------------------------------------------------


class TestRunDaily:
    def test_returns_dict_keyed_by_symbol(self):
        pipeline = _make_pipeline()
        with patch("llm.pipeline._write_scores"):
            result = pipeline.run_daily(["RELIANCE", "TCS", "INFY"])
        assert set(result.keys()) == {"RELIANCE", "TCS", "INFY"}

    def test_scores_are_floats(self):
        pipeline = _make_pipeline()
        with patch("llm.pipeline._write_scores"):
            result = pipeline.run_daily(["RELIANCE"])
        assert isinstance(result["RELIANCE"], float)

    def test_calls_scorer_for_each_symbol(self):
        mock_scorer = MagicMock()
        mock_scorer.score_aggregate_cached.return_value = 0.3
        pipeline = _make_pipeline(mock_scorer=mock_scorer)
        with patch("llm.pipeline._write_scores"):
            pipeline.run_daily(["RELIANCE", "TCS"])
        assert mock_scorer.score_aggregate_cached.call_count == 2

    def test_writes_scores_to_db(self):
        pipeline = _make_pipeline()
        with patch("llm.pipeline._write_scores") as mock_write:
            pipeline.run_daily(["RELIANCE"])
        mock_write.assert_called_once()

    def test_empty_universe_returns_empty_dict(self):
        pipeline = _make_pipeline()
        with patch("llm.pipeline._write_scores"):
            result = pipeline.run_daily([])
        assert result == {}

    def test_symbol_with_no_news_scores_zero(self):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_news.return_value = []
        pipeline = _make_pipeline(mock_fetcher=mock_fetcher)

        mock_scorer = MagicMock()
        mock_scorer.score_aggregate.return_value = 0.0
        mock_scorer.score_aggregate_cached.return_value = 0.0
        pipeline._scorer = mock_scorer

        with patch("llm.pipeline._write_scores"):
            result = pipeline.run_daily(["UNKNOWNSYMBOL"])

        assert result["UNKNOWNSYMBOL"] == 0.0

    def test_fetcher_error_returns_zero_not_crash(self):
        mock_fetcher = MagicMock()
        mock_fetcher.fetch_news.side_effect = Exception("network error")
        pipeline = _make_pipeline(mock_fetcher=mock_fetcher)
        with patch("llm.pipeline._write_scores"):
            result = pipeline.run_daily(["RELIANCE"])
        assert result["RELIANCE"] == 0.0


# ---------------------------------------------------------------------------
# get_latest_score
# ---------------------------------------------------------------------------


class TestGetLatestScore:
    @patch("llm.pipeline.get_redis")
    def test_returns_cached_score(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = "0.65"
        mock_get_redis.return_value = mock_redis

        pipeline = _make_pipeline()
        score = pipeline.get_latest_score("RELIANCE")
        assert score == 0.65

    @patch("llm.pipeline.get_redis")
    def test_returns_zero_on_cache_miss_and_no_db(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        pipeline = _make_pipeline()
        with patch("llm.pipeline._fetch_latest_score_from_db", return_value=None):
            score = pipeline.get_latest_score("NOSCORE")
        assert score == 0.0

    @patch("llm.pipeline.get_redis")
    def test_falls_back_to_db_on_cache_miss(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        pipeline = _make_pipeline()
        with patch("llm.pipeline._fetch_latest_score_from_db", return_value=0.42):
            score = pipeline.get_latest_score("RELIANCE")
        assert score == 0.42


# ---------------------------------------------------------------------------
# _write_scores (DB schema contract)
# ---------------------------------------------------------------------------


class TestWriteScores:
    def test_write_scores_calls_db_with_correct_columns(self):
        from llm.pipeline import _write_scores

        with patch("llm.pipeline.get_engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.return_value.connect.return_value.__enter__ = MagicMock(
                return_value=mock_conn
            )
            mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
            _write_scores({"RELIANCE": 0.5, "TCS": -0.2})
            mock_conn.execute.assert_called_once()
            sql_arg = str(mock_conn.execute.call_args[0][0])
            assert "sentiment_scores" in sql_arg
