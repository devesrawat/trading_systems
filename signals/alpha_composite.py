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
        # If high vol and we are doing momentum, maybe boost?
        # If low vol and we are doing mean reversion, maybe boost?
        # This part can be refined as we define the "strategies" better.

        # Clip multiplier to reasonable bounds
        final_multiplier = max(0.5, min(1.5, multiplier))

        log.info(
            "alpha_multiplier_calculated",
            symbol=symbol,
            multiplier=round(final_multiplier, 2),
            base_prob_impact=f"x{final_multiplier:.2f}",
        )

        return final_multiplier
