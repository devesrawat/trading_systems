"""
Momentum Sentinel — Hermes dual-engine, aggressive leg.

Criteria
--------
- Close > 50-day SMA (trend filter)
- Latest bar volume > 2× 20-day average volume (institutional accumulation)
- RSI(14) > 60 (momentum confirmation)

Produces the highest volume-ratio signals first so the strongest breakouts
are actioned before the move exhausts.

No 52-week high requirement (contrast with RSBreakout). This catches
early-stage momentum before the stock makes a new annual high.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from signals.base_strategy import BaseStrategy


class MomentumSentinelStrategy(BaseStrategy):
    name: str = "momentum_sentinel"
    lookback_days: int = 120
    interval: str = "day"
    min_bars: int = 60

    def scan(self, symbol: str, df: pd.DataFrame) -> dict[str, Any] | None:
        close = df["close"]
        volume = df["volume"]

        # 1. Above 50-day SMA
        sma50 = float(close.rolling(50).mean().iloc[-1])
        current = float(close.iloc[-1])
        if current <= sma50:
            return None

        # 2. Volume surge: latest bar > 2× 20-day average
        vol_avg_20 = float(volume.rolling(20).mean().iloc[-1])
        if vol_avg_20 <= 0:
            return None
        volume_ratio = float(volume.iloc[-1]) / vol_avg_20
        if volume_ratio < 2.0:
            return None

        # 3. RSI(14) > 60 — computed inline to stay pure CPU (no I/O)
        rsi = self._rsi(close, period=14)
        if rsi <= 60.0:
            return None

        return {
            "symbol": symbol,
            "strategy": self.name,
            "current_price": round(current, 2),
            "volume_ratio": round(volume_ratio, 2),
            "rsi_14": round(rsi, 2),
            "distance_to_sma50_pct": round((current - sma50) / sma50 * 100, 2),
        }

    def sort_key(self, result: dict[str, Any]) -> float:
        # Highest volume ratio first — most aggressive accumulation signal
        return -result.get("volume_ratio", 0.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> float:
        """
        Wilder-smoothed RSI via EWM (alpha=1/period).
        Returns 0.0 if insufficient data to stay strict.
        """
        delta = close.diff().dropna()
        if len(delta) < period:
            return 0.0
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = float(gain.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
        avg_loss = float(loss.ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1])
        if avg_loss == 0.0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)
