"""
Signal filters applied before order emission.

EarningsFilter — suppress signals for instruments in ±1 day earnings blackout.

Status: DISABLED by default (feature flag EARNINGS_FILTER_ENABLED=false).
Real NSE earnings scraping is deferred to Phase 4. The filter always returns
False (no blackout) until a reliable data source is wired in.

Usage:
    f = EarningsFilter()
    if f.is_blackout(symbol):
        return  # skip signal
"""

from __future__ import annotations

import structlog

try:
    from config.settings import settings as _settings
except Exception:
    _settings = None  # type: ignore[assignment]

log = structlog.get_logger(__name__)

# Allow tests to inject a mock via patch("signals.filters.settings")
settings = _settings

# Feature flag — set EARNINGS_FILTER_ENABLED=true in .env to enable.
# This has no effect until a real scraper is wired in (Phase 4 item).
_ENABLED_BY_DEFAULT = False


class EarningsFilter:
    """
    Earnings announcement blackout filter.

    Suppresses signals for instruments with earnings ±1 trading day from today.
    Requires earnings dates to be loaded into Redis (not yet implemented).
    When disabled or data unavailable, allows all signals through.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            try:
                import signals.filters as _m

                self._enabled = getattr(_m.settings, "earnings_filter_enabled", _ENABLED_BY_DEFAULT)
            except Exception:
                self._enabled = _ENABLED_BY_DEFAULT
        else:
            self._enabled = enabled

    def is_blackout(self, symbol: str) -> bool:
        """
        Return True if *symbol* is in an earnings blackout window.

        Always returns False until real earnings data is wired in.
        """
        if not self._enabled:
            return False

        try:
            return self._check_redis(symbol)
        except Exception as exc:
            log.debug("earnings_filter_check_failed", symbol=symbol, error=str(exc))
            return False  # fail open — allow signal if data unavailable

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_redis(self, symbol: str) -> bool:
        """
        Check Redis for an earnings date within ±1 day.

        Key format: trading:earnings:{symbol}  →  "YYYY-MM-DD"
        """
        from datetime import date

        from data.store import get_redis

        r = get_redis()
        raw = r.get(f"trading:earnings:{symbol}")
        if not raw:
            return False

        earnings_date_str = raw if isinstance(raw, str) else raw.decode()
        try:
            earnings_date = date.fromisoformat(earnings_date_str)
            today = date.today()
            return abs((earnings_date - today).days) <= 1
        except ValueError:
            return False

    def load_earnings_dates(self, dates: dict[str, str]) -> None:
        """
        Load earnings dates into Redis.

        Args:
            dates: {symbol: "YYYY-MM-DD"} mapping.
        """
        try:
            from data.store import get_redis

            r = get_redis()
            for symbol, date_str in dates.items():
                r.set(f"trading:earnings:{symbol}", date_str, ex=86400 * 7)
            log.info("earnings_dates_loaded", count=len(dates))
        except Exception as exc:
            log.warning("earnings_dates_load_failed", error=str(exc))
