"""
A/B test signal router.

Routes a configurable percentage of signals to a "challenger" model slot
for live comparison against the production (champion) model.

Routing is deterministic: the same symbol+date always maps to the same
slot, so a position opened under the challenger is always closed under
the challenger. Uses MD5 (not Python ``hash()``) to ensure stability
across restarts.

Usage
-----
    router = SignalRouter(challenger_pct=0.20)

    slot = router.route(symbol="RELIANCE", date="2026-04-14")
    if slot == "challenger":
        prob = challenger_model.predict(features)
    else:
        prob = champion_model.predict(features)

    router.record_outcome(symbol, date, slot, outcome_pnl)
"""

from __future__ import annotations

import hashlib
from contextlib import suppress
from datetime import date as date_type
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

Slot = Literal["champion", "challenger"]

_REDIS_SLOT_KEY = "trading:ab:slot:{symbol}:{date}"
_REDIS_OUTCOME_KEY = "trading:ab:outcome:{symbol}:{date}"


class SignalRouter:
    """
    Deterministic A/B test router using MD5 hash of (symbol, date).

    Parameters
    ----------
    challenger_pct : float
        Fraction [0, 1] of signals routed to the challenger slot.
        Default 0.20 (20 %). Set to 0 to disable A/B testing.
    """

    def __init__(self, challenger_pct: float = 0.20) -> None:
        if not 0.0 <= challenger_pct <= 1.0:
            raise ValueError(f"challenger_pct must be in [0, 1], got {challenger_pct}")
        self._challenger_pct = challenger_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, symbol: str, date: str | date_type | None = None) -> Slot:
        """
        Return the slot ("champion" or "challenger") for this symbol+date.

        The decision is based on a stable MD5 hash so it never changes
        between process restarts. Redis is written for audit purposes only.

        Parameters
        ----------
        symbol : Stock/crypto symbol, e.g. "RELIANCE" or "BTCUSDT".
        date   : Trading date as ISO string or ``datetime.date``.
                 Defaults to today if omitted.
        """
        if self._challenger_pct == 0.0:
            return "champion"

        date_str = str(date) if date else str(date_type.today())
        slot = self._stable_slot(symbol, date_str)
        self._persist_slot(symbol, date_str, slot)
        return slot

    def record_outcome(
        self,
        symbol: str,
        date: str | date_type,
        slot: Slot,
        pnl_pct: float,
    ) -> None:
        """
        Store realised P&L for a completed signal in Redis.

        Used offline to compute champion vs challenger Sharpe over a rolling
        window. Key TTL is 90 days (enough for 3-month comparison).
        """
        date_str = str(date)
        key = _REDIS_OUTCOME_KEY.format(symbol=symbol, date=date_str)
        value = f"{slot}:{pnl_pct:.6f}"
        try:
            from data.store import get_redis

            get_redis().setex(key, 60 * 60 * 24 * 90, value)
            log.debug("ab_outcome_recorded", symbol=symbol, date=date_str, slot=slot, pnl=pnl_pct)
        except Exception as exc:
            log.warning("ab_outcome_persist_failed", error=str(exc))

    def get_slot(self, symbol: str, date: str | date_type | None = None) -> Slot | None:
        """
        Retrieve a previously recorded routing decision from Redis.

        Returns None if no decision is stored for this symbol+date.
        """
        date_str = str(date) if date else str(date_type.today())
        key = _REDIS_SLOT_KEY.format(symbol=symbol, date=date_str)
        try:
            from data.store import get_redis

            raw = get_redis().get(key)
            if raw:
                val = raw.decode() if isinstance(raw, bytes) else raw
                return "champion" if val == "champion" else "challenger"
        except Exception as exc:
            log.warning("ab_slot_read_failed", error=str(exc))
        return None

    def summary(self, window_days: int = 90) -> dict[str, dict[str, float]]:
        """
        Compute per-slot P&L statistics from Redis outcome keys.

        Returns ``{"champion": {"count": n, "mean_pnl": x}, "challenger": {...}}``.
        """
        from datetime import timedelta

        stats: dict[str, list[float]] = {"champion": [], "challenger": []}

        try:
            from data.store import get_redis

            r = get_redis()
            today = date_type.today()
            for delta in range(window_days):
                d = (today - timedelta(days=delta)).isoformat()
                for key in r.scan_iter(f"trading:ab:outcome:*:{d}"):
                    raw = r.get(key)
                    if not raw:
                        continue
                    val = raw.decode() if isinstance(raw, bytes) else raw
                    parts = val.split(":", 1)
                    if len(parts) == 2 and parts[0] in stats:
                        with suppress(ValueError):
                            stats[parts[0]].append(float(parts[1]))
        except Exception as exc:
            log.warning("ab_summary_failed", error=str(exc))

        return {
            slot: {
                "count": len(pnls),
                "mean_pnl": sum(pnls) / len(pnls) if pnls else 0.0,
            }
            for slot, pnls in stats.items()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stable_slot(self, symbol: str, date_str: str) -> Slot:
        """Map (symbol, date) → slot using MD5 for process-stable routing."""
        raw = f"{symbol}:{date_str}".encode()
        digest = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return "challenger" if bucket < int(self._challenger_pct * 100) else "champion"

    def _persist_slot(self, symbol: str, date_str: str, slot: Slot) -> None:
        """Write routing decision to Redis for audit trail. Fire-and-forget."""
        key = _REDIS_SLOT_KEY.format(symbol=symbol, date=date_str)
        try:
            from data.store import get_redis

            get_redis().setex(key, 60 * 60 * 24 * 90, slot)
        except Exception as exc:
            log.debug("ab_slot_persist_failed", error=str(exc))
