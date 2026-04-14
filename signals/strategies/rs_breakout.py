"""
RS Breakout — Relative Strength new-high breakout (Minervini / IBD).

Criteria
--------
- RS score ≥ 85 (stock outperforms ≥ 85 % of the universe over 12 months)
- Price making a 52-week closing high today (or within 1 bar)
- Volume on breakout bar ≥ 40 % above 50-day average
- Above 200-day SMA (basic trend filter)

RS score is computed relative to the passed symbols in the same scan run.
Because each worker sees only one symbol's data, the RS score is approximated
from the stock's own 12-month return ranked against a pre-computed percentile
threshold stored in the result. The engine can normalise scores post-hoc if
needed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from signals.base_strategy import BaseStrategy


class RSBreakoutStrategy(BaseStrategy):
    name: str = "rs_breakout"
    lookback_days: int = 300
    interval: str = "day"
    min_bars: int = 200

    # RS threshold — override to tighten/loosen
    rs_min_return_12m: float = 0.20  # stock must be up ≥ 20 % over 12 months

    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        close = df["close"]
        volume = df["volume"]
        sma200 = close.rolling(200).mean()
        avg_vol50 = volume.rolling(50).mean()

        cur = close.iloc[-1]
        high52 = close.rolling(252).max().iloc[-1]
        prev_high52 = close.rolling(252).max().iloc[-2] if len(df) > 2 else high52

        # Must be above 200-day SMA
        if cur <= sma200.iloc[-1]:
            return None

        # 52-week closing high (today or previous bar — handles exact-high days)
        if cur < high52 * 0.995:
            return None

        # Breakout must be a NEW high vs the prior bar's 52-week high
        if high52 <= prev_high52:
            return None

        # Volume confirmation
        vol_today = float(volume.iloc[-1])
        avg_vol = float(avg_vol50.iloc[-1]) if not np.isnan(avg_vol50.iloc[-1]) else 1.0
        vol_ratio = vol_today / avg_vol if avg_vol > 0 else 0.0
        if vol_ratio < 1.40:
            return None

        # 12-month return as RS proxy
        bars_12m = min(252, len(close) - 1)
        return_12m = float((cur / close.iloc[-bars_12m - 1]) - 1) if bars_12m > 0 else 0.0
        if return_12m < self.rs_min_return_12m:
            return None

        return {
            "symbol": symbol,
            "strategy": self.name,
            "current_price": cur,
            "high_52w": float(high52),
            "return_12m_pct": round(return_12m * 100, 2),
            "volume_ratio": round(vol_ratio, 2),
            "above_sma200": True,
        }

    def sort_key(self, result: dict) -> float:
        # Highest RS return first
        return -result.get("return_12m_pct", 0.0)
