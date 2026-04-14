"""
VCP — Volatility Contraction Pattern (Minervini).

Criteria
--------
- Stage 2 trend template (5 conditions)
- ≥ 2 progressively tighter swing contractions in base (last 60 bars)
- Final contraction range < 10 %
- Volume dry-up in the most recent 10 bars
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from signals.base_strategy import BaseStrategy


class VCPStrategy(BaseStrategy):
    name: str = "vcp"
    lookback_days: int = 400
    interval: str = "day"
    min_bars: int = 200

    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        if not self._trend_template(df):
            return None

        ranges = self._swing_ranges(df, lookback=60)
        contractions = sum(1 for i in range(1, len(ranges)) if ranges[i] < ranges[i - 1])
        final_range = ranges[-1] if ranges else float("inf")

        if contractions < 2 or final_range >= 10.0:
            return None

        current = float(df["close"].iloc[-1])
        pivot = round(df.tail(15)["high"].max() * 1.005, 2)

        return {
            "symbol": symbol,
            "strategy": self.name,
            "current_price": current,
            "pivot_buy": pivot,
            "distance_to_pivot_pct": round((pivot - current) / current * 100, 2),
            "contractions": contractions,
            "swing_ranges": ranges,
            "volume_dry_up": self._volume_dry_up(df),
        }

    def sort_key(self, result: dict) -> float:
        return result.get("distance_to_pivot_pct", 99.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_template(df: pd.DataFrame) -> bool:
        close = df["close"]
        sma150 = close.rolling(150).mean()
        sma200 = close.rolling(200).mean()
        cur = close.iloc[-1]
        high52 = close.rolling(252).max().iloc[-1]
        low52 = close.rolling(252).min().iloc[-1]
        return (
            cur > sma150.iloc[-1]
            and cur > sma200.iloc[-1]
            and sma150.iloc[-1] > sma200.iloc[-1]
            and sma200.iloc[-1] > sma200.iloc[-22]
            and cur >= high52 * 0.75
            and cur >= low52 * 1.30
        )

    @staticmethod
    def _swing_ranges(df: pd.DataFrame, lookback: int = 60) -> list[float]:
        base = df.tail(lookback)
        high, low = base["high"], base["low"]
        ph_idx = high[(high.shift(1) < high) & (high.shift(-1) < high)].index.tolist()
        pl_idx = low[(low.shift(1) > low) & (low.shift(-1) > low)].index.tolist()
        ranges: list[float] = []
        for ph in ph_idx:
            subsequent = [pl for pl in pl_idx if pl > ph]
            if not subsequent:
                continue
            pl = subsequent[0]
            sh, sl = high.loc[ph], low.loc[pl]
            if sh > 0:
                ranges.append(round((sh - sl) / sh * 100, 2))
        return ranges

    @staticmethod
    def _volume_dry_up(df: pd.DataFrame) -> bool:
        if len(df) < 20:
            return False
        vol = df["volume"]
        return bool(vol.iloc[-10:].mean() < vol.iloc[-20:-10].mean())
