"""
Half-Kelly position sizing with volatility scaling.

Formula:
  edge          = signal_probability - 0.5
  kelly         = edge / (1 - edge)          ← full Kelly
  half_kelly    = kelly * 0.5                ← always use half
  vol_scalar    = min(1.0, 0.20 / max(vol, 0.05))
  base_size     = current_capital * max_position_pct
  final_size    = base_size * min(half_kelly, 1.0) * vol_scalar
"""
from __future__ import annotations

import math

import structlog

log = structlog.get_logger(__name__)

_HARD_CAP_PCT = 0.02    # 2% — non-negotiable
_VOL_TARGET = 0.20      # target annualised volatility for scaling
_VOL_FLOOR = 0.05       # prevent division by near-zero vol


class PositionSizer:
    def __init__(
        self,
        total_capital: float,
        max_position_pct: float = 0.02,
    ) -> None:
        if max_position_pct > _HARD_CAP_PCT:
            raise ValueError(
                f"max_position_pct {max_position_pct:.1%} exceeds the hard cap of 2%. "
                "This limit is non-negotiable."
            )
        self._total_capital = total_capital
        self._max_position_pct = max_position_pct

    # ------------------------------------------------------------------
    # size()
    # ------------------------------------------------------------------

    def size(
        self,
        signal_probability: float,
        asset_volatility: float,
        current_capital: float,
        correlation_penalty: float = 0.0,
    ) -> float:
        """
        Return position size in INR.

        Parameters
        ----------
        signal_probability  : Model-predicted P(positive outcome).
        asset_volatility    : Annualised volatility of the asset.
        current_capital     : Available capital (INR or USD).
        correlation_penalty : Fraction [0, 1] to reduce size by when the new
                              position is correlated with existing open positions.
                              E.g. 0.3 → final_size *= 0.7. Clamped to [0, 1].

        Returns 0.0 for signals with no edge (probability ≤ 0.5).
        """
        edge = signal_probability - 0.5
        if edge <= 0:
            return 0.0

        kelly = edge / max(1 - edge, 1e-9)
        half_kelly = kelly * 0.5

        vol_safe = max(asset_volatility, _VOL_FLOOR)
        vol_scalar = min(1.0, _VOL_TARGET / vol_safe)

        penalty = max(0.0, min(1.0, correlation_penalty))

        base_size = current_capital * self._max_position_pct
        final_size = base_size * min(half_kelly, 1.0) * vol_scalar * (1.0 - penalty)

        result = round(final_size, 2)
        log.debug(
            "position_sized",
            prob=signal_probability,
            vol=asset_volatility,
            half_kelly=round(half_kelly, 4),
            vol_scalar=round(vol_scalar, 4),
            correlation_penalty=round(penalty, 4),
            size_inr=result,
        )
        return result

    # ------------------------------------------------------------------
    # shares()
    # ------------------------------------------------------------------

    def shares(
        self,
        rupee_amount: float,
        current_price: float,
        lot_size: int = 1,
    ) -> int:
        """
        Convert an INR amount to a share/lot count.

        Equities : floor(rupee_amount / current_price)
        F&O      : floor(rupee_amount / (current_price * lot_size)) * lot_size
        """
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")

        if lot_size == 1:
            return int(math.floor(rupee_amount / current_price))

        lots = int(math.floor(rupee_amount / (current_price * lot_size)))
        return lots * lot_size

    # ------------------------------------------------------------------
    # max_allowed()
    # ------------------------------------------------------------------

    def max_allowed(self, current_capital: float) -> float:
        """Hard cap: 2% of current capital."""
        return round(current_capital * self._max_position_pct, 2)
