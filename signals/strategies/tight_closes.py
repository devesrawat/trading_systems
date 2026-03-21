"""
Tight Closes — 3+ consecutive closes within a narrow range (Minervini).

This pattern marks institutional accumulation: price barely moves for
several sessions, indicating sellers are exhausted and big money is
absorbing supply quietly.

Criteria
--------
- 3 or more consecutive closes within 1.5 % of each other
- The tight area sits within 10 % of the 52-week high (near the top of a base)
- Volume contracting over the tight closes (supply drying up)
- Price above 50-day SMA (stock must already be in a valid uptrend)

Entry
-----
Buy on a decisive range expansion day (close outside the tight range + volume surge).
This scanner flags the tight zone; the breakout is confirmed in real time via LiveFeed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals.base_strategy import BaseStrategy


class TightClosesStrategy(BaseStrategy):
    name: str          = "tight_closes"
    lookback_days: int = 200
    interval: str      = "day"
    min_bars: int      = 60

    tight_range_pct: float = 1.5    # max % spread across the tight closes
    min_tight_bars: int    = 3      # minimum number of consecutive tight bars

    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        close    = df["close"]
        volume   = df["volume"]
        sma50    = close.rolling(50).mean()
        high52   = close.rolling(252).max().iloc[-1]

        cur = close.iloc[-1]

        # Must be above 50-day SMA
        if cur <= sma50.iloc[-1]:
            return None

        # Tight area must be in the upper part of the 52-week range
        if cur < high52 * 0.90:
            return None

        # Find longest recent streak of tight closes
        streak_len, streak_start = self._find_tight_streak(close)
        if streak_len < self.min_tight_bars:
            return None

        # Tight range — pct spread from min to max close within streak
        streak_closes = close.iloc[streak_start : streak_start + streak_len]
        tight_range   = float((streak_closes.max() - streak_closes.min()) / streak_closes.mean() * 100)
        if tight_range > self.tight_range_pct:
            return None

        # Volume should be contracting over the streak
        streak_vols = volume.iloc[streak_start : streak_start + streak_len]
        vol_trend   = self._volume_contracting(streak_vols)

        # Breakout level = top of the tight range + 0.5 %
        breakout_level = round(float(streak_closes.max()) * 1.005, 2)

        return {
            "symbol":          symbol,
            "strategy":        self.name,
            "current_price":   cur,
            "breakout_level":  breakout_level,
            "tight_bars":      streak_len,
            "tight_range_pct": round(tight_range, 3),
            "volume_contracting": vol_trend,
            "distance_to_breakout_pct": round((breakout_level - cur) / cur * 100, 2),
        }

    def sort_key(self, result: dict) -> Any:
        # Tightest ranges first, then most bars
        return (result.get("tight_range_pct", 99.0), -result.get("tight_bars", 0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_tight_streak(self, close: pd.Series) -> tuple[int, int]:
        """
        Scan from the most recent bar backwards to find the longest
        consecutive streak where all closes stay within tight_range_pct.
        Returns (streak_length, start_index_in_series).
        """
        closes = close.values
        n      = len(closes)
        best_len   = 0
        best_start = n - 1

        for end in range(n - 1, -1, -1):
            # Expand window backwards until the range exceeds threshold
            for start in range(end, max(end - 30, -1), -1):
                window = closes[start : end + 1]
                if len(window) < 2:
                    continue
                rng = (window.max() - window.min()) / window.mean() * 100
                if rng > self.tight_range_pct:
                    break
                streak = end - start + 1
                if streak > best_len:
                    best_len   = streak
                    best_start = start
            break   # only check the most recent endpoint

        return best_len, best_start

    @staticmethod
    def _volume_contracting(vols: pd.Series) -> bool:
        if len(vols) < 2:
            return False
        # Simple linear regression slope — negative = contracting
        x = np.arange(len(vols), dtype=float)
        slope = float(np.polyfit(x, vols.values.astype(float), 1)[0])
        return slope < 0
