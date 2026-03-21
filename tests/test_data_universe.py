"""Unit tests for data/universe.py — mocks Kite API, no live calls."""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.universe import (
    DEFAULT_MIN_AVG_VOLUME,
    filter_liquid,
    get_fo_instruments,
    load_nse500_tokens,
    refresh_instruments,
)


# ---------------------------------------------------------------------------
# load_nse500_tokens
# ---------------------------------------------------------------------------

class TestLoadNse500Tokens:
    def test_loads_instruments_from_file(self, tmp_path, monkeypatch):
        instruments = [{"token": 1, "symbol": "RELIANCE", "segment": "EQ"}]
        f = tmp_path / "instruments.json"
        f.write_text(json.dumps({"instruments": instruments}))

        import data.universe as univ_mod
        monkeypatch.setattr(univ_mod, "_INSTRUMENTS_PATH", f)

        result = load_nse500_tokens()
        assert len(result) == 1
        assert result[0]["symbol"] == "RELIANCE"

    def test_empty_instruments_list_returns_empty(self, tmp_path, monkeypatch):
        f = tmp_path / "instruments.json"
        f.write_text(json.dumps({"instruments": []}))

        import data.universe as univ_mod
        monkeypatch.setattr(univ_mod, "_INSTRUMENTS_PATH", f)

        result = load_nse500_tokens()
        assert result == []


# ---------------------------------------------------------------------------
# refresh_instruments
# ---------------------------------------------------------------------------

class TestRefreshInstruments:
    def test_writes_eq_instruments_to_file(self, tmp_path, monkeypatch):
        f = tmp_path / "instruments.json"
        f.write_text(json.dumps({"instruments": []}))

        import data.universe as univ_mod
        monkeypatch.setattr(univ_mod, "_INSTRUMENTS_PATH", f)

        kite = MagicMock()
        kite.instruments.return_value = [
            {"instrument_token": 256265, "tradingsymbol": "RELIANCE", "name": "Reliance Industries",
             "exchange": "NSE", "segment": "EQ", "instrument_type": "EQ", "lot_size": 1, "tick_size": 0.05},
            {"instrument_token": 265, "tradingsymbol": "NIFTY24JAN18000CE", "name": "NIFTY",
             "exchange": "NSE", "segment": "NFO", "instrument_type": "CE", "lot_size": 50, "tick_size": 0.05},
        ]
        result = refresh_instruments(kite)

        assert len(result) == 1
        assert result[0]["symbol"] == "RELIANCE"
        written = json.loads(f.read_text())
        assert written["count"] == 1
        assert written["instruments"][0]["token"] == 256265

    def test_sets_last_updated_timestamp(self, tmp_path, monkeypatch):
        f = tmp_path / "instruments.json"
        f.write_text(json.dumps({"instruments": []}))

        import data.universe as univ_mod
        monkeypatch.setattr(univ_mod, "_INSTRUMENTS_PATH", f)

        kite = MagicMock()
        kite.instruments.return_value = []
        refresh_instruments(kite)

        written = json.loads(f.read_text())
        assert written["last_updated"] is not None


# ---------------------------------------------------------------------------
# get_fo_instruments
# ---------------------------------------------------------------------------

class TestGetFoInstruments:
    def test_filters_expired_instruments(self):
        kite = MagicMock()
        kite.instruments.return_value = [
            # Active contract
            {"instrument_token": 1, "tradingsymbol": "NIFTY24FEB18000CE", "name": "NIFTY",
             "instrument_type": "CE", "expiry": "2099-12-31", "strike": 18000, "lot_size": 50},
            # Expired contract
            {"instrument_token": 2, "tradingsymbol": "NIFTY22JAN17000CE", "name": "NIFTY",
             "instrument_type": "CE", "expiry": "2020-01-01", "strike": 17000, "lot_size": 50},
        ]
        result = get_fo_instruments(kite)
        assert len(result) == 1
        assert result[0]["symbol"] == "NIFTY24FEB18000CE"

    def test_no_expiry_excluded(self):
        kite = MagicMock()
        kite.instruments.return_value = [
            {"instrument_token": 3, "tradingsymbol": "NIFTY_FUT", "name": "NIFTY",
             "instrument_type": "FUT", "expiry": None, "strike": 0, "lot_size": 50},
        ]
        result = get_fo_instruments(kite)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# filter_liquid
# ---------------------------------------------------------------------------

class TestFilterLiquid:
    def _make_volume_df(self, avg_volume: float, n: int = 25) -> pd.DataFrame:
        return pd.DataFrame(
            {"volume": [int(avg_volume)] * n},
            index=pd.date_range("2024-01-01", periods=n, freq="B"),
        )

    def test_liquid_instrument_kept(self):
        universe = [{"token": 1, "symbol": "RELIANCE", "segment": "EQ"}]
        ohlcv_map = {1: self._make_volume_df(600_000)}
        result = filter_liquid(universe, ohlcv_map, min_avg_volume=DEFAULT_MIN_AVG_VOLUME)
        assert len(result) == 1

    def test_illiquid_instrument_excluded(self):
        universe = [{"token": 1, "symbol": "SMALLCO", "segment": "EQ"}]
        ohlcv_map = {1: self._make_volume_df(100_000)}
        result = filter_liquid(universe, ohlcv_map, min_avg_volume=DEFAULT_MIN_AVG_VOLUME)
        assert len(result) == 0

    def test_missing_ohlcv_excluded(self):
        universe = [{"token": 99, "symbol": "NOOHLCV", "segment": "EQ"}]
        result = filter_liquid(universe, {}, min_avg_volume=DEFAULT_MIN_AVG_VOLUME)
        assert len(result) == 0

    def test_uses_last_20_bars_only(self):
        """Instrument with recent low volume should be excluded even if older bars are high."""
        universe = [{"token": 1, "symbol": "DECLINING", "segment": "EQ"}]
        volumes = [2_000_000] * 100 + [10_000] * 20
        df = pd.DataFrame(
            {"volume": volumes},
            index=pd.date_range("2024-01-01", periods=120, freq="B"),
        )
        ohlcv_map = {1: df}
        result = filter_liquid(universe, ohlcv_map, min_avg_volume=DEFAULT_MIN_AVG_VOLUME)
        assert len(result) == 0
