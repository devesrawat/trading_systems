"""Unit tests for data/ingest.py — mocks KiteConnect, no live API calls."""
from datetime import date, datetime
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from data.ingest import VALID_INTERVALS, KiteIngestor
from data.providers.base import OHLCVProvider


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kite_cls():
    with patch("data.ingest.KiteConnect") as mock_cls:
        instance = MagicMock()
        instance.access_token = "test_token"
        mock_cls.return_value = instance
        yield mock_cls, instance


@pytest.fixture
def ingestor(mock_kite_cls):
    _, kite_instance = mock_kite_cls
    return KiteIngestor(api_key="test_key", access_token="test_token")


# ---------------------------------------------------------------------------
# _date_chunks
# ---------------------------------------------------------------------------

class TestDateChunks:
    def test_day_interval_single_chunk(self):
        chunks = OHLCVProvider._date_chunks(date(2022, 1, 1), date(2024, 1, 1), "day")
        assert len(chunks) == 1
        assert chunks[0][0] == date(2022, 1, 1)
        assert chunks[0][1] == date(2024, 1, 1)

    def test_minute_interval_splits_into_chunks(self):
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 6, 1)
        chunks = OHLCVProvider._date_chunks(from_date, to_date, "minute", chunk_days=60)
        assert len(chunks) > 1
        # Each chunk covers at most 60 days
        for start, end in chunks:
            delta = (end - start).days
            assert delta <= 60

    def test_chunks_cover_full_range(self):
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 6, 1)
        chunks = OHLCVProvider._date_chunks(from_date, to_date, "5minute", chunk_days=60)
        assert chunks[0][0] == from_date
        assert chunks[-1][1] >= to_date - pd.Timedelta(days=1)

    def test_short_range_single_chunk(self):
        from_date = datetime(2024, 1, 1)
        to_date = datetime(2024, 1, 30)
        chunks = OHLCVProvider._date_chunks(from_date, to_date, "minute", chunk_days=60)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# fetch_historical
# ---------------------------------------------------------------------------

class TestFetchHistorical:
    def test_invalid_interval_raises(self, ingestor):
        with pytest.raises(ValueError, match="Invalid interval"):
            ingestor.fetch_historical(256265, date(2024, 1, 1), date(2024, 3, 1), "2minute")

    def test_valid_intervals_accepted(self, ingestor, mock_kite_cls):
        _, kite_instance = mock_kite_cls
        kite_instance.historical_data.return_value = []
        for interval in VALID_INTERVALS:
            ingestor.fetch_historical(256265, date(2024, 1, 1), date(2024, 1, 10), interval)

    def test_empty_api_response_returns_empty_df(self, ingestor, mock_kite_cls):
        _, kite_instance = mock_kite_cls
        kite_instance.historical_data.return_value = []
        result = ingestor.fetch_historical(256265, date(2024, 1, 1), date(2024, 1, 10), "day")
        assert result.empty

    @patch("data.ingest.write_ohlcv")
    def test_writes_to_db_on_success(self, mock_write, ingestor, mock_kite_cls):
        _, kite_instance = mock_kite_cls
        kite_instance.historical_data.return_value = [
            {"date": datetime(2024, 1, 2), "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1_000_000},
            {"date": datetime(2024, 1, 3), "open": 103.0, "high": 107.0, "low": 102.0, "close": 106.0, "volume": 1_200_000},
        ]
        ingestor.fetch_historical(256265, date(2024, 1, 1), date(2024, 1, 10), "day", symbol="RELIANCE")
        mock_write.assert_called_once()
        written_df = mock_write.call_args[0][0]
        assert len(written_df) == 2
        assert "symbol" in written_df.columns
        assert "token" in written_df.columns

    @patch("data.ingest.write_ohlcv")
    def test_deduplicates_overlapping_chunks(self, mock_write, ingestor, mock_kite_cls):
        _, kite_instance = mock_kite_cls
        # Return the same bar twice (simulating overlap)
        row = {"date": datetime(2024, 1, 2), "open": 100.0, "high": 105.0, "low": 99.0, "close": 103.0, "volume": 1_000_000}
        kite_instance.historical_data.return_value = [row]
        result = ingestor.fetch_historical(256265, date(2024, 1, 1), date(2024, 1, 10), "day")
        assert not result.index.duplicated().any()


# ---------------------------------------------------------------------------
# refresh_access_token
# ---------------------------------------------------------------------------

class TestRefreshAccessToken:
    @patch("data.ingest.get_redis")
    def test_stores_token_in_redis(self, mock_get_redis, ingestor, mock_kite_cls):
        _, kite_instance = mock_kite_cls
        kite_instance.generate_session.return_value = {"access_token": "new_token_abc"}
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        with patch("data.ingest.settings") as mock_settings:
            mock_settings.kite_api_secret = "secret"
            token = ingestor.refresh_access_token("request_token_xyz")

        assert token == "new_token_abc"
        mock_redis.set.assert_called_with("trading:auth:kite:access_token", "new_token_abc")
        kite_instance.set_access_token.assert_called_with("new_token_abc")
