"""
FinBERT sentiment scoring engine.

Model: ProsusAI/finbert (cached locally in .models/)
Score: positive_prob - negative_prob  →  float in [-1, +1]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import structlog
import torch
from transformers import pipeline

from data.redis_keys import RedisKeys
from data.store import get_redis

log = structlog.get_logger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_MODEL_CACHE = str(Path(__file__).parent.parent / ".models")
_CACHE_TTL = 3600  # 1 hour
_BATCH_GPU = 32
_BATCH_CPU = 8
_MAX_TOKENS = 512


def _detect_device() -> int:
    """Return torch device index: CUDA > MPS > CPU (-1)."""
    if torch.cuda.is_available():
        log.info("device_selected", device="cuda")
        return 0
    if torch.backends.mps.is_available():
        log.info("device_selected", device="mps")
        return 0  # transformers maps device=0 to MPS on Apple Silicon
    log.info("device_selected", device="cpu")
    return -1


class FinBERTScorer:
    """
    Wraps ProsusAI/finbert for batch headline scoring.

    Usage
    -----
    scorer = FinBERTScorer()
    scores = scorer.score(["headline 1", "headline 2"])   # [-1, +1] per headline
    agg    = scorer.score_aggregate(headlines)             # single float
    """

    def __init__(self) -> None:
        device = _detect_device()
        self._batch_size = _BATCH_GPU if device >= 0 else _BATCH_CPU
        self._pipe = pipeline(
            task="text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=None,  # return all 3 class scores
            device=device,
            model_kwargs={"cache_dir": _MODEL_CACHE},
            truncation=True,
            max_length=_MAX_TOKENS,
        )
        log.info("finbert_loaded", model=_MODEL_NAME, batch_size=self._batch_size)

    # ------------------------------------------------------------------
    # score()
    # ------------------------------------------------------------------

    def score(self, texts: list[str]) -> list[float]:
        """
        Score each headline independently.

        Returns a list of floats in [-1, +1]:
            score = positive_prob - negative_prob
        """
        if not texts:
            return []

        results: list[float] = []
        for batch_start in range(0, len(texts), self._batch_size):
            batch = texts[batch_start : batch_start + self._batch_size]
            raw = self._pipe(batch, truncation=True, max_length=_MAX_TOKENS)
            for label_scores in raw:
                score = _extract_score(label_scores)
                results.append(score)

        return results

    # ------------------------------------------------------------------
    # score_aggregate()
    # ------------------------------------------------------------------

    def score_aggregate(
        self,
        texts: list[str],
        method: str = "mean",
    ) -> float:
        """
        Aggregate multiple headlines into a single sentiment float.

        method='mean'              — simple average
        method='weighted_recency'  — later headlines get higher weight
                                     (linear ramp: weight[i] = i+1)
        """
        if not texts:
            return 0.0

        scores = self.score(texts)

        if method == "mean":
            return float(np.mean(scores))

        if method == "weighted_recency":
            n = len(scores)
            weights = np.arange(1, n + 1, dtype=float)  # [1, 2, 3, …, n]
            weights /= weights.sum()
            return float(np.dot(weights, scores))

        raise ValueError(
            f"Unknown aggregation method '{method}'. Use 'mean' or 'weighted_recency'."
        )

    # ------------------------------------------------------------------
    # Cached variant
    # ------------------------------------------------------------------

    def score_aggregate_cached(
        self,
        texts: list[str],
        symbol: str,
        date: str,
        method: str = "mean",
    ) -> float:
        """
        score_aggregate() with Redis caching.

        Cache key: ``RedisKeys.sentiment(symbol, date)``  TTL: 1 hour
        """
        cache_key = RedisKeys.sentiment(symbol, date)
        r = get_redis()

        cached = r.get(cache_key)
        if cached is not None:
            log.debug("sentiment_cache_hit", symbol=symbol, date=date)
            return float(cached)

        score = self.score_aggregate(texts, method=method)
        r.setex(cache_key, _CACHE_TTL, str(score))
        log.debug("sentiment_cached", symbol=symbol, date=date, score=score)
        return score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_score(label_scores: list[dict]) -> float:
    """Convert FinBERT 3-class output to a single [-1, +1] float."""
    scores_map: dict[str, float] = {d["label"].lower(): d["score"] for d in label_scores}
    positive = scores_map.get("positive", 0.0)
    negative = scores_map.get("negative", 0.0)
    return float(positive - negative)
