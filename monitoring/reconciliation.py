"""
Daily P&L reconciliation.

Compares paper portfolio positions vs current market prices.
Sends a Telegram alert if the unrealised P&L drift exceeds 0.5%.

The drift threshold catches data inconsistencies (stale prices, missed fills)
rather than serving as a risk limit — the circuit breaker handles that.

Usage (called from post_market_summary()):
    reconciler = DailyReconciler(broker)
    reconciler.reconcile(open_positions)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from execution.broker import BrokerAdapter

log = structlog.get_logger(__name__)

_DRIFT_THRESHOLD = 0.005  # 0.5%


class DailyReconciler:
    """Compare paper positions vs live prices; alert on significant drift."""

    def __init__(self, broker: BrokerAdapter) -> None:  # type: ignore[name-defined]
        self._broker = broker

    def reconcile(self, open_positions: set[str]) -> dict[str, float]:
        """
        For each open position, compare recorded entry price vs current quote.

        Returns {symbol: pnl_pct} for all positions checked.
        Sends alert if any position has P&L drift > _DRIFT_THRESHOLD.
        """
        if not open_positions:
            log.debug("reconciliation_skipped_no_positions")
            return {}

        results: dict[str, float] = {}
        try:
            quotes = self._broker.get_quote(list(open_positions))
        except Exception as exc:
            log.warning("reconciliation_quote_failed", error=str(exc))
            return {}

        for symbol, quote in quotes.items():
            try:
                entry = self._get_recorded_entry(symbol)
                if entry is None or entry <= 0:
                    continue
                current = float(quote.get("last_price", 0))
                if current <= 0:
                    continue
                pnl_pct = (current - entry) / entry
                results[symbol] = pnl_pct
                if abs(pnl_pct) > _DRIFT_THRESHOLD:
                    log.warning(
                        "reconciliation_drift",
                        symbol=symbol,
                        entry=round(entry, 2),
                        current=round(current, 2),
                        pnl_pct=round(pnl_pct, 4),
                    )
            except Exception as exc:
                log.debug("reconciliation_symbol_error", symbol=symbol, error=str(exc))

        large_drift = {s: p for s, p in results.items() if abs(p) > _DRIFT_THRESHOLD}
        if large_drift:
            self._alert(large_drift)

        log.info("reconciliation_complete", positions=len(results), drifting=len(large_drift))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_recorded_entry(self, symbol: str) -> float | None:
        """Read last recorded entry price from Redis (written by OrderExecutor)."""
        try:
            from data.store import get_redis

            raw = get_redis().get(f"trading:position:entry:{symbol}")
            return float(raw) if raw else None
        except Exception:
            return None

    def _alert(self, drifting: dict[str, float]) -> None:
        lines = "\n".join(f"  {s}: {p:+.2%}" for s, p in drifting.items())
        msg = f"⚠️ <b>P&L Reconciliation Alert</b>\nDrift > 0.5% detected:\n{lines}"
        try:
            from monitoring.alerts import TelegramAlerter

            TelegramAlerter().send(msg)
        except Exception:
            pass
