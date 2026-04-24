"""Sentiment indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_sentiment_score(raw_score: float | None) -> float:
    if raw_score is None:
        return np.nan
    return max(-1.0, min(1.0, float(raw_score)))


def compute_finbert_sentiment_series(sentiment_scores: pd.Series | None = None) -> pd.Series:
    if sentiment_scores is None or sentiment_scores.empty:
        return pd.Series(np.nan)
    return sentiment_scores.fillna(method="ffill")


def compute_news_sentiment_series(
    news_scores: pd.Series | None = None, period: int = 5
) -> pd.Series:
    if news_scores is None or news_scores.empty:
        return pd.Series(np.nan)
    return news_scores.rolling(period, min_periods=1).mean()


def compute_social_sentiment_series(
    social_scores: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if social_scores is None or social_scores.empty:
        return pd.Series(np.nan)
    return social_scores.rolling(period, min_periods=1).mean()


def compute_sentiment_divergence(
    finbert: pd.Series | None = None, news: pd.Series | None = None
) -> pd.Series:
    if finbert is None or news is None:
        if finbert is not None:
            return pd.Series(np.nan, index=finbert.index)
        elif news is not None:
            return pd.Series(np.nan, index=news.index)
        else:
            return pd.Series(np.nan)
    return (finbert - news).abs()


def compute_sentiment_momentum(sentiment: pd.Series | None = None, period: int = 5) -> pd.Series:
    if sentiment is None or sentiment.empty:
        return pd.Series(np.nan)
    return sentiment - sentiment.shift(period)
