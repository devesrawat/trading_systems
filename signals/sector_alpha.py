"""
Sector Relative Strength Alpha.

Ranks NSE sectors by 5-day and 20-day relative strength against Nifty 50.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import structlog

from data.store import get_ohlcv

log = structlog.get_logger(__name__)

# Sector Index Token Mapping
# These tokens are specific to Zerodha Kite.
# Use data/universe.py refresh_instruments() or Kite API to get the correct tokens.
SECTOR_INDICES = {
    "NIFTY 50": 256265,
    "NIFTY BANK": 260105,
    "NIFTY IT": 262665,
    "NIFTY AUTO": 261641,
    "NIFTY PHARMA": 261897,
    "NIFTY FMCG": 262153,
    "NIFTY METAL": 262409,
    "NIFTY REALTY": 262921,
    "NIFTY MEDIA": 263177,
    "NIFTY PSU BANK": 263433,
    "NIFTY PVT BANK": 263689,
    "NIFTY FIN SERVICE": 264713,
}


class SectorRanker:
    """Ranks sectors by relative strength against a benchmark."""

    def __init__(self, benchmark: str = "NIFTY 50"):
        self.benchmark = benchmark
        self.benchmark_token = SECTOR_INDICES.get(benchmark, 256265)
        self._cached_ranks: pd.DataFrame | None = None
        self._last_calc_date: date | None = None

    def get_ranks(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Get current sector rankings.
        Ranks are cached for the day.
        """
        today = datetime.utcnow().date()
        if not force_refresh and self._cached_ranks is not None and self._last_calc_date == today:
            return self._cached_ranks

        log.info("calculating_sector_ranks", benchmark=self.benchmark)

        # Calculate RS for each sector
        rs_data = []
        end_date = today
        start_date = end_date - timedelta(days=40)  # Enough for 20-day returns

        # Fetch benchmark data
        try:
            benchmark_df = get_ohlcv(self.benchmark_token, start_date, end_date, "day")
        except Exception as e:
            log.error("benchmark_fetch_failed", symbol=self.benchmark, error=str(e))
            return pd.DataFrame()

        if benchmark_df.empty or len(benchmark_df) < 21:
            log.warning(
                "benchmark_data_insufficient", symbol=self.benchmark, rows=len(benchmark_df)
            )
            return pd.DataFrame()

        benchmark_ret_5 = benchmark_df["close"].pct_change(5).iloc[-1]
        benchmark_ret_20 = benchmark_df["close"].pct_change(20).iloc[-1]

        for sector, token in SECTOR_INDICES.items():
            if sector == self.benchmark:
                continue

            try:
                df = get_ohlcv(token, start_date, end_date, "day")
            except Exception as e:
                log.debug("sector_fetch_failed", symbol=sector, error=str(e))
                continue

            if df.empty or len(df) < 21:
                continue

            ret_5 = df["close"].pct_change(5).iloc[-1]
            ret_20 = df["close"].pct_change(20).iloc[-1]

            # RS = Difference in returns
            rs_5 = ret_5 - benchmark_ret_5
            rs_20 = ret_20 - benchmark_ret_20

            rs_data.append(
                {
                    "sector": sector,
                    "rs_5": rs_5,
                    "rs_20": rs_20,
                    "composite_rs": (rs_5 * 0.4) + (rs_20 * 0.6),
                }
            )

        df_ranks = pd.DataFrame(rs_data)
        if df_ranks.empty:
            log.warning("no_sector_data_found")
            return df_ranks

        df_ranks["rank"] = df_ranks["composite_rs"].rank(ascending=False)
        df_ranks.sort_values("rank", inplace=True)

        self._cached_ranks = df_ranks
        self._last_calc_date = today
        log.info("sector_ranks_calculated", top_sectors=df_ranks.head(3)["sector"].tolist())

        return df_ranks

    def is_top_sector(self, sector_name: str, top_n: int = 3) -> bool:
        """Check if a sector is in the top N."""
        ranks = self.get_ranks()
        if ranks.empty:
            return False

        top_sectors = ranks.head(top_n)["sector"].tolist()
        normalized_name = self._normalize_sector_name(sector_name)
        return normalized_name in top_sectors

    def is_bottom_sector(self, sector_name: str, bottom_n: int = 3) -> bool:
        """Check if a sector is in the bottom N."""
        ranks = self.get_ranks()
        if ranks.empty:
            return False

        bottom_sectors = ranks.tail(bottom_n)["sector"].tolist()
        normalized_name = self._normalize_sector_name(sector_name)
        return normalized_name in bottom_sectors

    def _normalize_sector_name(self, name: str) -> str:
        """Map generic sector names to index names."""
        mapping = {
            "Banking": "NIFTY BANK",
            "IT": "NIFTY IT",
            "Automobiles": "NIFTY AUTO",
            "Energy": "NIFTY 50",  # Placeholder
            "FMCG": "NIFTY FMCG",
            "Financials": "NIFTY FIN SERVICE",
            "Pharma": "NIFTY PHARMA",
            "Metal": "NIFTY METAL",
            "Realty": "NIFTY REALTY",
        }
        return mapping.get(name, name)
