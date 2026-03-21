"""
Delta-neutral F&O strategy engine for NSE options.

Signal Rules (from project spec)
---------------------------------
  SELL_PREMIUM : iv_rank > 0.70 AND iv_premium > 0.05
                 → iron condor or straddle (collect elevated premium)
  BUY_PREMIUM  : iv_rank < 0.30
                 → long straddle before expected event (cheap vol)
  No signal    : otherwise

Position Limits
---------------
  max 3 concurrent F&O positions (configurable)
  1 lot per signal

Delta Hedge
-----------
  If net portfolio delta > ±0.10, hedge by buying/selling the underlying
  to flatten delta within the threshold.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Any

import structlog

from options.iv_features import IVFeatures

log = structlog.get_logger(__name__)

_IV_RANK_SELL_THRESHOLD = 0.70
_IV_PREMIUM_SELL_THRESHOLD = 0.05
_IV_RANK_BUY_THRESHOLD = 0.30
_DELTA_HEDGE_THRESHOLD = 0.10


class SignalType(str, Enum):
    SELL_PREMIUM = "SELL_PREMIUM"
    BUY_PREMIUM = "BUY_PREMIUM"


class FoStrategyEngine:
    """
    Generates delta-neutral options signals and manages concurrent positions.

    Usage
    -----
    engine = FoStrategyEngine(max_concurrent_positions=3)
    signal = engine.generate_signal(iv_features)
    if signal:
        engine.add_position(position_id)
    hedge = engine.compute_delta_hedge(net_delta, underlying_price, lot_size)
    """

    def __init__(self, max_concurrent_positions: int = 3) -> None:
        self._max_positions = max_concurrent_positions
        self._open_positions: set[str] = set()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def position_count(self) -> int:
        return len(self._open_positions)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def add_position(self, position_id: str) -> None:
        """Register a new open position. Raises ValueError if at capacity."""
        if len(self._open_positions) >= self._max_positions:
            raise ValueError(
                f"Cannot add position '{position_id}': "
                f"already at max ({self._max_positions}) concurrent positions."
            )
        self._open_positions.add(position_id)
        log.info("fo_position_added", position_id=position_id, total=self.position_count)

    def remove_position(self, position_id: str) -> None:
        """Deregister a closed position."""
        self._open_positions.discard(position_id)
        log.info("fo_position_removed", position_id=position_id, total=self.position_count)

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(self, features: IVFeatures) -> dict[str, Any] | None:
        """
        Evaluate IV features and return a signal dict or None.

        Returns None when:
          - IV rank is in the neutral zone (0.30–0.70)
          - Sell condition met but iv_premium ≤ 0.05
          - Already at max_concurrent_positions
        """
        if self.position_count >= self._max_positions:
            log.info(
                "fo_signal_blocked_capacity",
                symbol=features.symbol,
                capacity=self._max_positions,
            )
            return None

        signal_type: SignalType | None = None

        if (
            features.iv_rank > _IV_RANK_SELL_THRESHOLD
            and features.iv_premium > _IV_PREMIUM_SELL_THRESHOLD
        ):
            signal_type = SignalType.SELL_PREMIUM
        elif features.iv_rank < _IV_RANK_BUY_THRESHOLD:
            signal_type = SignalType.BUY_PREMIUM

        if signal_type is None:
            return None

        signal: dict[str, Any] = {
            "type": signal_type,
            "symbol": features.symbol,
            "expiry_date": features.expiry_date.isoformat(),
            "iv_rank": round(features.iv_rank, 4),
            "iv_percentile": round(features.iv_percentile, 2),
            "iv_premium": round(features.iv_premium, 4),
            "put_call_ratio": round(features.put_call_ratio, 3),
            "max_pain": features.max_pain,
            "days_to_expiry": features.days_to_expiry,
            "current_iv": round(features.current_iv, 4),
            "realized_vol": round(features.realized_vol, 4),
            "lots": 1,  # always 1 lot per signal (project spec)
        }

        log.info(
            "fo_signal_generated",
            signal_type=signal_type.value,
            symbol=features.symbol,
            iv_rank=signal["iv_rank"],
            days_to_expiry=features.days_to_expiry,
        )
        return signal

    # ------------------------------------------------------------------
    # Delta hedging
    # ------------------------------------------------------------------

    def compute_delta_hedge(
        self,
        net_delta: float,
        underlying_price: float,
        lot_size: int,
    ) -> dict[str, Any] | None:
        """
        Compute the hedge order needed to flatten portfolio delta within ±0.10.

        Parameters
        ----------
        net_delta         : current net portfolio delta (in underlying share equiv.)
        underlying_price  : current spot price of the underlying
        lot_size          : number of shares per futures/options lot

        Returns
        -------
        dict with keys {action, qty, underlying_price} or None if no hedge needed.
        """
        if abs(net_delta) <= _DELTA_HEDGE_THRESHOLD:
            return None

        # Buy or sell underlying shares to neutralise the delta
        hedge_qty = max(1, math.ceil(abs(net_delta)))
        action = "SELL" if net_delta > 0 else "BUY"

        log.info(
            "delta_hedge_required",
            net_delta=round(net_delta, 4),
            action=action,
            qty=hedge_qty,
            underlying_price=underlying_price,
        )

        return {
            "action": action,
            "qty": hedge_qty,
            "underlying_price": underlying_price,
        }
