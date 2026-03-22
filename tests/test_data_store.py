"""Unit tests for data/store.py — mocks DB and Redis, no real connections."""
import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.store import _df_to_records, get_latest_tick, get_universe, write_tick


# ---------------------------------------------------------------------------
# _df_to_records
# ---------------------------------------------------------------------------

class TestDfToRecords:
    def test_index_named_time_is_reset(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "token": [1, 1, 1],
                "symbol": ["RELIANCE", "RELIANCE", "RELIANCE"],
                "open": [100.0, 101.0, 102.0],
                "high": [105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 101.0],
                "close": [103.0, 104.0, 105.0],
                "volume": [1_000_000, 1_100_000, 1_200_000],
                "interval": ["day", "day", "day"],
            },
            index=idx,
        )
        df.index.name = "time"
        records = _df_to_records(df)
        assert len(records) == 3
        assert "time" in records[0]
        assert records[0]["symbol"] == "RELIANCE"

    def test_non_time_index_handled(self):
        df = pd.DataFrame(
            {
                "time": pd.date_range("2024-01-01", periods=2, freq="D"),
                "token": [1, 1],
                "symbol": ["TCS", "TCS"],
                "open": [100.0, 101.0],
                "high": [105.0, 106.0],
                "low": [99.0, 100.0],
                "close": [103.0, 104.0],
                "volume": [500_000, 600_000],
                "interval": ["day", "day"],
            }
        )
        records = _df_to_records(df)
        assert len(records) == 2


# ---------------------------------------------------------------------------
# Redis tick cache
# ---------------------------------------------------------------------------

class TestTickCache:
    @patch("data.store.get_redis")
    def test_write_tick_sets_key_with_ttl(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        write_tick(256265, {"last_price": 2500.5, "volume": 1000})

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "trading:tick:256265"
        assert args[1] == 5  # TTL
        assert "last_price" in args[2]

    @patch("data.store.get_redis")
    def test_get_latest_tick_returns_dict(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps({"last_price": 2500.5})
        mock_get_redis.return_value = mock_redis

        result = get_latest_tick(256265)
        assert result == {"last_price": 2500.5}

    @patch("data.store.get_redis")
    def test_get_latest_tick_returns_none_if_missing(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis

        result = get_latest_tick(999999)
        assert result is None


# ---------------------------------------------------------------------------
# get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    def test_filters_by_segment(self, tmp_path, monkeypatch):
        instruments = [
            {"token": 1, "symbol": "RELIANCE", "segment": "EQ"},
            {"token": 2, "symbol": "NIFTY24JAN", "segment": "INDICES"},
        ]
        instruments_file = tmp_path / "instruments.json"
        instruments_file.write_text(json.dumps({"instruments": instruments}))

        import data.store as store_mod
        monkeypatch.setattr(store_mod, "_INSTRUMENTS_PATH", instruments_file)

        result = get_universe(segment="EQ")
        assert len(result) == 1
        assert result[0]["symbol"] == "RELIANCE"

    def test_empty_instruments_returns_empty_list(self, tmp_path, monkeypatch):
        instruments_file = tmp_path / "instruments.json"
        instruments_file.write_text(json.dumps({"instruments": []}))

        import data.store as store_mod
        monkeypatch.setattr(store_mod, "_INSTRUMENTS_PATH", instruments_file)

        result = get_universe()
        assert result == []
