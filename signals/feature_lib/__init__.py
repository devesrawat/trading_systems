"""
Feature engineering framework — modular indicators across 5 categories.

Public API exports all feature functions for use in feature building pipelines.

Categories:
1. core_indicators: Momentum (RSI, MACD, ROC)
2. volatility: ATR, Bollinger Bands, realized/Parkinson/Garman-Klass vol
3. sentiment: FinBERT, news, social sentiment
4. flow: FII/DII, MF inflows, institutional participation
5. advanced: Correlation, sector relative strength, regime, acceleration
"""

from __future__ import annotations

from signals.feature_lib.advanced import (
    compute_atr_trailing_stop,
    compute_correlation_to_benchmark,
    compute_price_acceleration,
    compute_regime_indicator,
    compute_sector_relative_strength,
    compute_support_resistance_levels,
)
from signals.feature_lib.core_indicators import (
    compute_macd,
    compute_macd_cross,
    compute_momentum,
    compute_roc,
    compute_rsi,
)
from signals.feature_lib.flow import (
    compute_dii_participation_series,
    compute_fii_net_cash_normalized,
    compute_fii_participation_series,
    compute_institutional_holding_trend,
    compute_mf_inflow_trend,
    compute_net_flow_to_volume_ratio,
    compute_retail_participation,
)
from signals.feature_lib.sentiment import (
    compute_finbert_sentiment_series,
    compute_news_sentiment_series,
    compute_sentiment_divergence,
    compute_sentiment_momentum,
    compute_social_sentiment_series,
    normalize_sentiment_score,
)
from signals.feature_lib.volatility import (
    compute_atr,
    compute_atr_pct,
    compute_bollinger_bands,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_realized_volatility,
    compute_volatility_of_volatility,
)

__all__ = [
    # Core indicators
    "compute_rsi",
    "compute_macd",
    "compute_macd_cross",
    "compute_momentum",
    "compute_roc",
    # Volatility
    "compute_atr",
    "compute_atr_pct",
    "compute_bollinger_bands",
    "compute_realized_volatility",
    "compute_parkinson_volatility",
    "compute_garman_klass_volatility",
    "compute_volatility_of_volatility",
    # Sentiment
    "normalize_sentiment_score",
    "compute_finbert_sentiment_series",
    "compute_news_sentiment_series",
    "compute_social_sentiment_series",
    "compute_sentiment_divergence",
    "compute_sentiment_momentum",
    # Flow
    "compute_fii_net_cash_normalized",
    "compute_fii_participation_series",
    "compute_dii_participation_series",
    "compute_net_flow_to_volume_ratio",
    "compute_mf_inflow_trend",
    "compute_institutional_holding_trend",
    "compute_retail_participation",
    # Advanced
    "compute_correlation_to_benchmark",
    "compute_sector_relative_strength",
    "compute_regime_indicator",
    "compute_price_acceleration",
    "compute_atr_trailing_stop",
    "compute_support_resistance_levels",
]
