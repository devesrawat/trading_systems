"""
Wealth Architect Scanner — Hermes dual-engine, conservative leg.

Screens a symbol universe for blue-chip compounding candidates by reading
pre-cached fundamentals from Redis (key ``FUND:{symbol}``).

Criteria (Hermes Wealth Architect skill):
- PE ratio < sector average PE (fallback cap: 25 when sector avg unavailable)
- ROE > 15%

This is a standalone class, NOT a BaseStrategy subclass, because it must
read from Redis — which is forbidden inside BaseStrategy.scan() (worker
process constraint). Call it from a scheduled job or TradingSystem method
running in the main process.

Redis cache format (JSON):
    {
        "pe": 18.5,
        "roe": 22.4,
        "sector": "Banking",
        "sector_avg_pe": 21.0   # optional; falls back to FALLBACK_PE_CAP
    }
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from data.store import get_redis

log = structlog.get_logger(__name__)

_ROE_THRESHOLD = 15.0
_FALLBACK_PE_CAP = 25.0


class WealthArchitectScanner:
    """
    Screens symbols for value + quality compounders.

    Parameters
    ----------
    roe_threshold : float
        Minimum ROE (%). Default: 15.
    fallback_pe_cap : float
        PE ceiling used when sector average is unavailable. Default: 25.
    """

    def __init__(
        self,
        roe_threshold: float = _ROE_THRESHOLD,
        fallback_pe_cap: float = _FALLBACK_PE_CAP,
    ) -> None:
        self.roe_threshold = roe_threshold
        self.fallback_pe_cap = fallback_pe_cap

    def run(self, symbols: list[str]) -> list[dict[str, Any]]:
        """
        Scan *symbols* and return those passing all criteria, sorted by ROE desc.

        Symbols without a Redis cache entry are silently skipped.

        Parameters
        ----------
        symbols : list[str]
            NSE trading symbols to evaluate.

        Returns
        -------
        list[dict[str, Any]]
            Each entry contains: symbol, pe, roe, sector_avg_pe,
            pe_discount_pct, sector.
        """
        redis = get_redis()
        results: list[dict[str, Any]] = []

        for symbol in symbols:
            try:
                raw = redis.get(f"FUND:{symbol}")
                if raw is None:
                    continue

                data: dict[str, Any] = json.loads(raw)
                pe = data.get("pe")
                roe = data.get("roe")

                if pe is None or roe is None:
                    continue

                pe = float(pe)
                roe = float(roe)
                sector_avg_pe = float(data.get("sector_avg_pe") or self.fallback_pe_cap)

                if not (pe < sector_avg_pe and roe > self.roe_threshold):
                    continue

                results.append(
                    {
                        "symbol": symbol,
                        "pe": round(pe, 2),
                        "roe": round(roe, 2),
                        "sector_avg_pe": round(sector_avg_pe, 2),
                        "pe_discount_pct": round((sector_avg_pe - pe) / sector_avg_pe * 100, 1),
                        "sector": data.get("sector", "Unknown"),
                    }
                )

            except Exception as exc:
                log.debug("wealth_architect_symbol_skip", symbol=symbol, error=str(exc))

        results.sort(key=lambda r: r["roe"], reverse=True)
        log.info("wealth_architect_scan_complete", candidates=len(results))
        return results
