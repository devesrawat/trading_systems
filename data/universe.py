"""
NSE500 instrument universe management.

Provides:
  - load_nse500_tokens     — read from config/instruments.json
  - refresh_instruments    — pull fresh dump from Kite and persist
  - get_fo_instruments     — active F&O instruments with lot sizes
  - filter_liquid          — keep only instruments above volume threshold
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import structlog

if TYPE_CHECKING:
    from kiteconnect import KiteConnect

log = structlog.get_logger(__name__)

_INSTRUMENTS_PATH = Path(__file__).parent.parent / "config" / "instruments.json"

# Kite exchange strings
_NSE_EXCHANGE = "NSE"
_NFO_EXCHANGE = "NFO"

# Minimum 20-day average daily volume to be considered liquid
DEFAULT_MIN_AVG_VOLUME = 500_000


# ---------------------------------------------------------------------------
# Load from disk
# ---------------------------------------------------------------------------


def load_nse500_tokens() -> list[dict[str, Any]]:
    """
    Load the NSE500 instrument list from config/instruments.json.

    Returns a list of instrument dicts, each containing at minimum:
      token, symbol, name, segment, lot_size, exchange
    """
    with open(_INSTRUMENTS_PATH) as f:
        data = json.load(f)

    instruments: list[dict[str, Any]] = data.get("instruments", [])
    if not instruments:
        log.warning(
            "instruments_json_empty",
            path=str(_INSTRUMENTS_PATH),
            hint="Run refresh_instruments(kite) to populate.",
        )
    log.debug("instruments_loaded", count=len(instruments))
    return instruments


# ---------------------------------------------------------------------------
# Refresh from Kite API
# ---------------------------------------------------------------------------


def refresh_instruments(kite: KiteConnect) -> list[dict[str, Any]]:
    """
    Pull the full instrument dump from Kite, filter to NSE equities,
    and persist to config/instruments.json.

    Returns the new instrument list.
    """
    log.info("refreshing_instruments_from_kite")
    all_instruments = kite.instruments(exchange=_NSE_EXCHANGE)

    # Keep only EQ segment instruments
    eq_instruments = [
        {
            "token": int(inst["instrument_token"]),
            "symbol": inst["tradingsymbol"],
            "name": inst.get("name", ""),
            "exchange": inst["exchange"],
            "segment": inst.get("segment", "EQ"),
            "instrument_type": inst.get("instrument_type", "EQ"),
            "lot_size": int(inst.get("lot_size", 1)),
            "tick_size": float(inst.get("tick_size", 0.05)),
        }
        for inst in all_instruments
        if inst.get("instrument_type") == "EQ"
    ]

    payload = {
        "last_updated": datetime.utcnow().isoformat(),
        "count": len(eq_instruments),
        "instruments": eq_instruments,
    }
    with open(_INSTRUMENTS_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    log.info("instruments_refreshed", count=len(eq_instruments))
    return eq_instruments


# ---------------------------------------------------------------------------
# F&O instruments
# ---------------------------------------------------------------------------


def get_fo_instruments(kite: KiteConnect) -> list[dict[str, Any]]:
    """
    Return active F&O (NFO) instruments with lot sizes.

    Filters to near-expiry futures and options contracts.
    """
    log.info("fetching_fo_instruments")
    all_nfo = kite.instruments(exchange=_NFO_EXCHANGE)
    today = datetime.utcnow().date()

    active = []
    for inst in all_nfo:
        expiry = inst.get("expiry")
        if not expiry:
            continue
        if isinstance(expiry, str):
            expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
        if expiry < today:
            continue

        active.append(
            {
                "token": int(inst["instrument_token"]),
                "symbol": inst["tradingsymbol"],
                "name": inst.get("name", ""),
                "exchange": _NFO_EXCHANGE,
                "instrument_type": inst.get("instrument_type", ""),  # CE, PE, FUT
                "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                "strike": float(inst.get("strike", 0)),
                "lot_size": int(inst.get("lot_size", 1)),
                "underlying": inst.get("name", ""),
            }
        )

    log.info("fo_instruments_fetched", count=len(active))
    return active


# ---------------------------------------------------------------------------
# Liquidity filter
# ---------------------------------------------------------------------------


def filter_liquid(
    universe: list[dict[str, Any]],
    ohlcv_df_map: dict[int, pd.DataFrame],
    min_avg_volume: int = DEFAULT_MIN_AVG_VOLUME,
) -> list[dict[str, Any]]:
    """
    Filter *universe* to instruments whose 20-day average daily volume
    exceeds *min_avg_volume*.

    *ohlcv_df_map* maps instrument_token → DataFrame with a 'volume' column.
    Instruments without OHLCV data are excluded.
    """
    liquid = []
    for inst in universe:
        token = inst["token"]
        df = ohlcv_df_map.get(token)
        if df is None or df.empty or "volume" not in df.columns:
            continue
        avg_vol = df["volume"].tail(20).mean()
        if avg_vol >= min_avg_volume:
            liquid.append(inst)

    log.info(
        "liquidity_filter_applied",
        before=len(universe),
        after=len(liquid),
        min_avg_volume=min_avg_volume,
    )
    return liquid
