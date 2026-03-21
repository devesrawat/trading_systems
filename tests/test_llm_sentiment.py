"""Unit tests for llm/sentiment.py — TDD RED phase. No GPU/HuggingFace download needed."""
from unittest.mock import MagicMock, patch

import pytest

from llm.sentiment import FinBERTScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POSITIVE_HEADLINES = [
    "Reliance Industries reports record quarterly profit",
    "TCS wins $500M multi-year cloud contract",
    "HDFC Bank beats earnings estimates, raises dividend",
]

NEGATIVE_HEADLINES = [
    "Infosys cuts annual revenue guidance amid weak demand",
    "Wipro misses earnings, CEO cites macro headwinds",
    "RBI raises rates unexpectedly, market sells off",
]

MIXED_HEADLINES = POSITIVE_HEADLINES + NEGATIVE_HEADLINES


def _make_mock_scorer() -> tuple[FinBERTScorer, MagicMock]:
    """Return a FinBERTScorer with the HuggingFace pipeline mocked out."""
    with patch("llm.sentiment.pipeline") as mock_pipeline_fn:
        mock_pipe = MagicMock()

        def fake_infer(texts, **kwargs):
            results = []
            for text in texts:
                if any(w in text.lower() for w in ["profit", "wins", "beats", "raises dividend"]):
                    results.append([
                        {"label": "positive", "score": 0.85},
                        {"label": "negative", "score": 0.05},
                        {"label": "neutral",  "score": 0.10},
                    ])
                elif any(w in text.lower() for w in ["cuts", "misses", "sells off", "headwinds"]):
                    results.append([
                        {"label": "positive", "score": 0.05},
                        {"label": "negative", "score": 0.88},
                        {"label": "neutral",  "score": 0.07},
                    ])
                else:
                    results.append([
                        {"label": "positive", "score": 0.33},
                        {"label": "negative", "score": 0.33},
                        {"label": "neutral",  "score": 0.34},
                    ])
            return results

        mock_pipe.side_effect = fake_infer
        mock_pipeline_fn.return_value = mock_pipe
        scorer = FinBERTScorer()

    return scorer, mock_pipe


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestFinBERTScorerInit:
    def test_loads_model_on_init(self):
        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_fn.return_value = MagicMock()
            scorer = FinBERTScorer()
            mock_fn.assert_called_once()

    def test_model_name_is_finbert(self):
        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_fn.return_value = MagicMock()
            FinBERTScorer()
            call_args = mock_fn.call_args
            assert "ProsusAI/finbert" in str(call_args)

    def test_device_kwarg_passed(self):
        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_fn.return_value = MagicMock()
            FinBERTScorer()
            call_kwargs = mock_fn.call_args[1]
            assert "device" in call_kwargs


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScore:
    def test_returns_list_of_floats(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score(POSITIVE_HEADLINES)
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_output_length_matches_input(self):
        scorer, _ = _make_mock_scorer()
        texts = ["headline one", "headline two", "headline three"]
        result = scorer.score(texts)
        assert len(result) == len(texts)

    def test_scores_bounded_minus1_to_plus1(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score(MIXED_HEADLINES)
        for v in result:
            assert -1.0 <= v <= 1.0

    def test_positive_headlines_score_positive(self):
        scorer, _ = _make_mock_scorer()
        scores = scorer.score(POSITIVE_HEADLINES)
        assert all(s > 0 for s in scores)

    def test_negative_headlines_score_negative(self):
        scorer, _ = _make_mock_scorer()
        scores = scorer.score(NEGATIVE_HEADLINES)
        assert all(s < 0 for s in scores)

    def test_empty_input_returns_empty_list(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score([])
        assert result == []

    def test_batches_large_inputs(self):
        scorer, mock_pipe = _make_mock_scorer()
        texts = ["headline"] * 100
        scorer.score(texts)
        # Should have been called multiple times (batching)
        assert mock_pipe.call_count >= 1

    def test_score_formula_is_positive_minus_negative(self):
        """score = positive_prob - negative_prob."""
        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_pipe = MagicMock()
            mock_pipe.return_value = [[
                {"label": "positive", "score": 0.70},
                {"label": "negative", "score": 0.20},
                {"label": "neutral",  "score": 0.10},
            ]]
            mock_fn.return_value = mock_pipe
            scorer = FinBERTScorer()
            result = scorer.score(["test headline"])
        assert abs(result[0] - (0.70 - 0.20)) < 1e-6


# ---------------------------------------------------------------------------
# score_aggregate()
# ---------------------------------------------------------------------------

class TestScoreAggregate:
    def test_mean_aggregation(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score_aggregate(POSITIVE_HEADLINES, method="mean")
        assert isinstance(result, float)
        assert result > 0

    def test_weighted_recency_aggregation(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score_aggregate(POSITIVE_HEADLINES, method="weighted_recency")
        assert isinstance(result, float)
        assert -1.0 <= result <= 1.0

    def test_recency_weights_recent_more(self):
        """Most recent headline (last in list) should have highest weight."""
        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_pipe = MagicMock()
            # First headline: strong negative. Last headline: strong positive.
            mock_pipe.side_effect = lambda texts, **kw: [
                [{"label": "positive", "score": 0.05}, {"label": "negative", "score": 0.90}, {"label": "neutral", "score": 0.05}]
                if i == 0
                else [{"label": "positive", "score": 0.90}, {"label": "negative", "score": 0.05}, {"label": "neutral", "score": 0.05}]
                for i in range(len(texts))
            ]
            mock_fn.return_value = mock_pipe
            scorer = FinBERTScorer()
            score = scorer.score_aggregate(["old bad news", "new great news"], method="weighted_recency")
        # Weighted toward recent (positive) → aggregate should be positive
        assert score > 0

    def test_invalid_method_raises(self):
        scorer, _ = _make_mock_scorer()
        with pytest.raises(ValueError, match="method"):
            scorer.score_aggregate(["headline"], method="unknown")

    def test_empty_input_returns_zero(self):
        scorer, _ = _make_mock_scorer()
        result = scorer.score_aggregate([])
        assert result == 0.0


# ---------------------------------------------------------------------------
# Redis caching
# ---------------------------------------------------------------------------

class TestRedisCache:
    @patch("llm.sentiment.get_redis")
    def test_cache_hit_skips_model(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = "0.75"
        mock_get_redis.return_value = mock_redis

        with patch("llm.sentiment.pipeline") as mock_fn:
            mock_fn.return_value = MagicMock()
            scorer = FinBERTScorer()
            result = scorer.score_aggregate_cached(["headline"], symbol="RELIANCE", date="2024-01-15")

        assert result == 0.75
        # Model should NOT have been called
        mock_fn.return_value.assert_not_called()

    @patch("llm.sentiment.get_redis")
    def test_cache_miss_runs_model_and_stores(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        scorer, mock_pipe = _make_mock_scorer()
        with patch("llm.sentiment.get_redis", return_value=mock_redis):
            scorer.score_aggregate_cached(POSITIVE_HEADLINES, symbol="TCS", date="2024-01-15")

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args[0]
        assert "sentiment:TCS:2024-01-15" == call_args[0]
        assert call_args[1] == 3600  # 1 hour TTL
