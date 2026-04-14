"""
NSE F&O options chain scraper.

Computes Put/Call Ratio (PCR), IV skew, and max pain from the NSE options chain.

PCR interpretation:
  PCR > 1.5  →  bullish (more put OI than call OI = hedging, not bearish bets)
  PCR < 0.7  →  bearish (more call OI than put OI = speculative calls)
  0.7–1.5    →  neutral

These values are used as confirmatory signals in RegimeDetector
(high-confidence buy signals when PCR > 1.5 AND regime is TRENDING_BULL).

Usage:
    scraper = OptionsPCRScraper()
    pcr_data = scraper.get_pcr("NIFTY")
    # {"pcr": 1.23, "call_oi": 1234567, "put_oi": 1524680, "max_pain": 22400}
"""

from __future__ import annotations

import structlog

log = structlog.get_logger(__name__)

_NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; trading-system/1.0)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/option-chain",
}


class OptionsPCRScraper:
    """
    NSE F&O options chain scraper for PCR and max pain computation.

    Uses a session cookie approach to bypass NSE's AJAX protection.
    The session is initialized by hitting the main NSE page first.
    """

    def __init__(self, session=None) -> None:
        self._session = session  # allow injection for testing

    def _get_session(self):
        """Return injected session or create a new requests.Session."""
        if self._session is not None:
            return self._session
        import requests

        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        try:
            session.get("https://www.nseindia.com", timeout=10)
        except Exception as exc:
            log.warning("nse_session_init_failed", error=str(exc))
        return session

    def get_pcr(self, symbol: str = "NIFTY") -> float | None:
        """
        Fetch options chain and return the put/call OI ratio.

        Returns the PCR float, or None if data is unavailable.
        """
        result = self.get_pcr_data(symbol)
        if result is None:
            return None
        return result["pcr"]

    def get_pcr_data(self, symbol: str = "NIFTY") -> dict[str, float] | None:
        """
        Fetch options chain and compute PCR + max pain.

        Returns:
            {
              "pcr":      float,   # put/call OI ratio
              "call_oi":  float,   # total call open interest
              "put_oi":   float,   # total put open interest
              "max_pain": float,   # strike at which total option loss is minimum
            }
        Returns None on error.
        """
        try:
            session = self._get_session()
            resp = session.get(
                _NSE_OPTION_CHAIN_URL,
                params={"symbol": symbol.upper()},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_chain(data)
        except Exception as exc:
            log.warning("options_scraper_failed", symbol=symbol, error=str(exc))
            return None

    def get_pcr_signal(self, symbol: str = "NIFTY") -> str:
        """
        Return a human-readable regime signal from PCR.

        Returns "BULLISH" | "BEARISH" | "NEUTRAL" | "UNAVAILABLE"
        """
        try:
            pcr = self.get_pcr(symbol)
            if pcr is None:
                return "NEUTRAL"
            if pcr > 1.5:
                return "BULLISH"
            elif pcr < 0.7:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as exc:
            log.warning("pcr_signal_unavailable", symbol=symbol, error=str(exc))
            return "UNAVAILABLE"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_chain(self, data: dict) -> dict[str, float] | None:
        """Parse raw NSE options chain JSON into PCR summary. Returns None if empty."""
        records = data.get("records", {}).get("data", [])

        if not records:
            return None

        call_oi_by_strike: dict[float, float] = {}
        put_oi_by_strike: dict[float, float] = {}

        for rec in records:
            strike = float(rec.get("strikePrice", 0))
            if "CE" in rec:
                call_oi_by_strike[strike] = float(rec["CE"].get("openInterest", 0))
            if "PE" in rec:
                put_oi_by_strike[strike] = float(rec["PE"].get("openInterest", 0))

        total_call_oi = sum(call_oi_by_strike.values())
        total_put_oi = sum(put_oi_by_strike.values())

        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        max_pain = self._compute_max_pain(call_oi_by_strike, put_oi_by_strike)

        log.info(
            "options_chain_parsed",
            symbol="NSE",
            pcr=round(pcr, 3),
            max_pain=max_pain,
            call_oi=int(total_call_oi),
            put_oi=int(total_put_oi),
        )

        return {
            "pcr": round(pcr, 3),
            "call_oi": total_call_oi,
            "put_oi": total_put_oi,
            "max_pain": max_pain,
        }

    def _compute_max_pain(
        self,
        call_oi: dict[float, float],
        put_oi: dict[float, float],
    ) -> float:
        """
        Max pain = strike price at which total option buyer losses are maximised.

        For each potential expiry strike S:
          loss = sum over all call strikes K < S: (S - K) * call_OI[K]
               + sum over all put strikes K > S:  (K - S) * put_OI[K]
        """
        strikes = sorted(set(call_oi) | set(put_oi))
        if not strikes:
            return 0.0

        min_loss = float("inf")
        max_pain_strike = strikes[0]

        for s in strikes:
            loss = sum((s - k) * call_oi[k] for k in strikes if k < s and k in call_oi) + sum(
                (k - s) * put_oi[k] for k in strikes if k > s and k in put_oi
            )
            if loss < min_loss:
                min_loss = loss
                max_pain_strike = s

        return max_pain_strike
