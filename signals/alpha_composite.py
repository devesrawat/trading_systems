"""
Composite Alpha Multiplier Engine.

Combines Sector Alpha, Options Alpha, and Regime into a single multiplier.
"""

from __future__ import annotations

from typing import Any

import structlog

from signals.options_alpha import OptionsFlowAnalyzer
from signals.sector_alpha import SectorRanker

log = structlog.get_logger(__name__)


class AlphaEngine:
    """Combines multiple alpha signals into a single multiplier."""

    def __init__(self, kite: Any | None = None):
        self.sector_ranker = SectorRanker()
        self.options_analyzer = OptionsFlowAnalyzer(kite)

    def calculate_multiplier(
        self, symbol: str, sector: str, current_regime: str | int, side: str = "BUY"
    ) -> float:
        """
        Calculate the Alpha Multiplier (0.5 to 1.5).

        Parameters
        ----------
        symbol : str
            Stock symbol.
        sector : str
            Sector name.
        current_regime : str or int
            Current market regime (e.g. from HMM).
        side : str
            'BUY' or 'SELL'.

        Returns
        -------
        float
            Multiplier to be applied to the baseline signal probability.
        """
        multiplier = 1.0

        # 1. Sector Filter (+0.2 if in top 3 sectors)
        if side == "BUY":
            if self.sector_ranker.is_top_sector(sector, top_n=3):
                multiplier += 0.2
                log.debug("alpha_boost_sector", symbol=symbol, sector=sector, boost=0.2)
        elif side == "SELL" and self.sector_ranker.is_bottom_sector(sector, bottom_n=3):
            multiplier += 0.2
            log.debug("alpha_boost_sector_short", symbol=symbol, sector=sector, boost=0.2)

        # 2. Options Sentiment (+0.2 max)
        opt_sentiment = self.options_analyzer.get_sentiment_score(symbol)
        if side == "BUY" and opt_sentiment > 0.5:
            multiplier += 0.2
            log.debug("alpha_boost_options", symbol=symbol, boost=0.2)
        elif side == "SELL" and opt_sentiment < -0.5:
            multiplier += 0.2
            log.debug("alpha_boost_options_short", symbol=symbol, boost=0.2)

        # 3. Regime Alignment
        # trending_bull → boost BUY / suppress SELL
        # trending_bear → suppress BUY (avoid chasing longs in a downtrend) / boost SELL
        # choppy        → suppress both sides (low signal quality)
        # high_vol      → mild caution both sides
        _BUY_DELTA: dict[str, float] = {
            "trending_bull": 0.1,
            "trending_bear": -0.3,
            "high_vol": -0.1,
            "choppy": -0.2,
            "normal": 0.0,
        }
        _SELL_DELTA: dict[str, float] = {
            "trending_bull": -0.1,
            "trending_bear": 0.1,
            "high_vol": 0.0,
            "choppy": -0.1,
            "normal": 0.0,
        }
        regime_key = str(current_regime).lower() if current_regime is not None else "normal"
        regime_delta = (
            _BUY_DELTA.get(regime_key, 0.0) if side == "BUY" else _SELL_DELTA.get(regime_key, 0.0)
        )
        if regime_delta != 0.0:
            multiplier += regime_delta
            log.debug(
                "alpha_regime_adjustment",
                symbol=symbol,
                regime=regime_key,
                side=side,
                delta=regime_delta,
            )

        # Clip multiplier to reasonable bounds
        final_multiplier = max(0.5, min(1.5, multiplier))

        log.info(
            "alpha_multiplier_calculated",
            symbol=symbol,
            multiplier=round(final_multiplier, 2),
            base_prob_impact=f"x{final_multiplier:.2f}",
        )

        return final_multiplier
