"""Unit tests for data/providers/binance.py — mocks HTTP and WebSocket, no live calls."""

import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.providers.binance import (
    BinanceProvider,
    _klines_to_df,
    _symbol_hash,
    _to_ms,
)

# ---------------------------------------------------------------------------
# _symbol_hash
# ---------------------------------------------------------------------------


class TestSymbolHash:
    def test_deterministic(self):
        assert _symbol_hash("BTCUSDT") == _symbol_hash("BTCUSDT")

    def test_always_positive(self):
        for pair in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"):
            assert _symbol_hash(pair) > 0

    def test_different_pairs_produce_different_hashes(self):
        hashes = {_symbol_hash(p) for p in ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")}
        assert len(hashes) == 4

    def test_case_insensitive(self):
        assert _symbol_hash("btcusdt") == _symbol_hash("BTCUSDT")


# ---------------------------------------------------------------------------
# _to_ms
# ---------------------------------------------------------------------------


class TestToMs:
    def test_date_converts_to_midnight_utc_ms(self):
        ms = _to_ms(date(2024, 1, 1))
        assert ms == int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1000)

    def test_datetime_converts_correctly(self):
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
        assert _to_ms(dt) == int(dt.timestamp() * 1000)

    def test_naive_datetime_treated_as_utc(self):
        naive = datetime(2024, 1, 1, 0, 0, 0)
        ms = _to_ms(naive)
        assert ms > 0


# ---------------------------------------------------------------------------
# _klines_to_df
# ---------------------------------------------------------------------------


def _fake_kline_row(open_time_ms: int = 1704067200000) -> list:
    """Return a single Binance kline row with close_time at open+60s."""
    return [
        open_time_ms,  # 0  open_time
        "42000.0",  # 1  open
        "42500.0",  # 2  high
        "41800.0",  # 3  low
        "42200.0",  # 4  close
        "150.5",  # 5  volume
        open_time_ms + 59999,  # 6  close_time
        "6321000.0",  # 7  quote_asset_volume
        1234,  # 8  number_of_trades
        "75.25",  # 9  taker_buy_base
        "3160500.0",  # 10 taker_buy_quote
        "0",  # 11 ignore
    ]


class TestKlinesToDf:
    def test_returns_dataframe(self):
        raw = [_fake_kline_row(1704067200000), _fake_kline_row(1704067260000)]
        df = _klines_to_df(raw, "BTC", "BTCUSDT", "minute", 12345)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_index_is_datetime(self):
        raw = [_fake_kline_row()]
        df = _klines_to_df(raw, "BTC", "BTCUSDT", "day", 12345)
        assert df.index.name == "time"
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_required_columns_present(self):
        raw = [_fake_kline_row()]
        df = _klines_to_df(raw, "BTC", "BTCUSDT", "day", 12345)
        for col in ("open", "high", "low", "close", "volume", "symbol", "interval", "token"):
            assert col in df.columns

    def test_numeric_columns_are_float(self):
        raw = [_fake_kline_row()]
        df = _klines_to_df(raw, "BTC", "BTCUSDT", "day", 12345)
        for col in ("open", "high", "low", "close", "volume"):
            assert pd.api.types.is_float_dtype(df[col])

    def test_symbol_and_interval_set_correctly(self):
        raw = [_fake_kline_row()]
        df = _klines_to_df(raw, "BTC", "BTCUSDT", "5minute", 99)
        assert df["symbol"].iloc[0] == "BTC"
        assert df["interval"].iloc[0] == "5minute"
        assert df["token"].iloc[0] == 99


# ---------------------------------------------------------------------------
# BinanceProvider — instrument registry
# ---------------------------------------------------------------------------


class TestBinanceProviderRegistry:
    def test_register_and_resolve_pair(self):
        p = BinanceProvider()
        p.register_instruments({"BTC": "BTCUSDT"})
        assert p._resolve_pair("BTC") == "BTCUSDT"

    def test_resolve_case_insensitive(self):
        p = BinanceProvider()
        p.register_instruments({"BTC": "BTCUSDT"})
        assert p._resolve_pair("btc") == "BTCUSDT"

    def test_pair_itself_also_resolves(self):
        """register_instruments auto-registers the pair string as a key too."""
        p = BinanceProvider()
        p.register_instruments({"BTC": "BTCUSDT"})
        assert p._resolve_pair("BTCUSDT") == "BTCUSDT"

    def test_unknown_symbol_raises_value_error(self):
        p = BinanceProvider()
        with pytest.raises(ValueError, match="No Binance pair registered"):
            p._resolve_pair("UNKNOWN")

    def test_multiple_instruments(self):
        p = BinanceProvider()
        p.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"})
        assert p._resolve_pair("ETH") == "ETHUSDT"
        assert p._resolve_pair("SOL") == "SOLUSDT"


# ---------------------------------------------------------------------------
# BinanceProvider — fetch_historical
# ---------------------------------------------------------------------------


def _make_provider(*pairs: str) -> BinanceProvider:
    p = BinanceProvider()
    p.register_instruments({pair.replace("USDT", ""): pair for pair in pairs})
    return p


class TestFetchHistorical:
    def _raw_one_bar(self, open_time_ms: int = 1704067200000) -> list[list]:
        return [_fake_kline_row(open_time_ms)]

    @patch("data.providers.binance.write_ohlcv")
    @patch.object(BinanceProvider, "_get")
    def test_returns_dataframe_with_rows(self, mock_get, mock_write):
        mock_get.return_value = self._raw_one_bar()
        p = _make_provider("BTCUSDT")
        df = p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 1, 2), "day")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    @patch("data.providers.binance.write_ohlcv")
    @patch.object(BinanceProvider, "_get")
    def test_persists_to_db(self, mock_get, mock_write):
        mock_get.return_value = self._raw_one_bar()
        p = _make_provider("BTCUSDT")
        p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 1, 2), "day")
        mock_write.assert_called_once()

    @patch("data.providers.binance.write_ohlcv")
    @patch.object(BinanceProvider, "_get")
    def test_empty_api_response_returns_empty_dataframe(self, mock_get, mock_write):
        mock_get.return_value = []
        p = _make_provider("BTCUSDT")
        df = p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 1, 2), "day")
        assert df.empty
        mock_write.assert_not_called()

    @patch("data.providers.binance.write_ohlcv")
    @patch.object(BinanceProvider, "_get")
    def test_paginates_when_full_page_returned(self, mock_get, mock_write):
        """When Binance returns exactly MAX_BARS (1000), a second request is made."""
        from data.providers.binance import _MAX_BARS_PER_REQUEST

        # First page: exactly 1000 bars ending at a known close_time
        first_page = [
            _fake_kline_row(1704067200000 + i * 60000) for i in range(_MAX_BARS_PER_REQUEST)
        ]
        # Second page: fewer than 1000 bars → stop
        second_page = [_fake_kline_row(1704067200000 + (_MAX_BARS_PER_REQUEST + 1) * 60000)]
        mock_get.side_effect = [first_page, second_page]

        p = _make_provider("BTCUSDT")
        df = p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 6, 1), "minute")
        assert mock_get.call_count == 2
        assert len(df) == _MAX_BARS_PER_REQUEST + 1

    def test_unknown_symbol_raises(self):
        p = BinanceProvider()  # no instruments registered
        with pytest.raises(ValueError):
            p.fetch_historical("UNKNOWN", date(2024, 1, 1), date(2024, 1, 2), "day")

    def test_invalid_interval_raises(self):
        p = _make_provider("BTCUSDT")
        with pytest.raises(ValueError, match="Unknown interval"):
            p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 1, 2), "2hour")

    @patch("data.providers.binance.write_ohlcv")
    @patch.object(BinanceProvider, "_get")
    def test_deduplicates_index(self, mock_get, mock_write):
        """Duplicate rows from overlapping pages are dropped."""
        bar = _fake_kline_row(1704067200000)
        # Three side effects: two pages with the same bar, then empty to halt loop
        mock_get.side_effect = [[bar], [bar], []]
        from data.providers import binance as bmod

        original_max = bmod._MAX_BARS_PER_REQUEST
        bmod._MAX_BARS_PER_REQUEST = 1
        try:
            p = _make_provider("BTCUSDT")
            df = p.fetch_historical("BTC", date(2024, 1, 1), date(2024, 1, 2), "day")
            assert not df.index.duplicated().any()
        finally:
            bmod._MAX_BARS_PER_REQUEST = original_max


# ---------------------------------------------------------------------------
# BinanceProvider — get_quote
# ---------------------------------------------------------------------------


class TestGetQuote:
    @patch.object(BinanceProvider, "_get")
    def test_returns_friendly_name_keyed_dict(self, mock_get):
        """get_quote maps Binance pairs back to the friendly names used in register_instruments."""
        mock_get.return_value = [
            {"symbol": "BTCUSDT", "price": "42000.00"},
            {"symbol": "ETHUSDT", "price": "2500.00"},
        ]
        p = BinanceProvider()
        p.register_instruments({"BTC": "BTCUSDT", "ETH": "ETHUSDT"})
        result = p.get_quote(["BTC", "ETH"])
        assert "BTC" in result
        assert "ETH" in result
        assert result["BTC"]["last_price"] == 42000.0

    @patch.object(BinanceProvider, "_get")
    def test_returns_pair_key_when_no_friendly_name(self, mock_get):
        """When the pair is registered as its own key, the result key is the pair itself."""
        mock_get.return_value = [{"symbol": "BTCUSDT", "price": "42000.00"}]
        p = BinanceProvider()
        p.register_instruments({"BTCUSDT": "BTCUSDT"})
        result = p.get_quote(["BTCUSDT"])
        assert "BTCUSDT" in result

    @patch.object(BinanceProvider, "_get")
    def test_empty_symbols_returns_empty(self, mock_get):
        mock_get.return_value = []
        p = _make_provider("BTCUSDT")
        result = p.get_quote([])
        assert result == {}


# ---------------------------------------------------------------------------
# BinanceProvider — stream_live
# ---------------------------------------------------------------------------


class TestStreamLive:
    def _mock_websocket_module(self) -> MagicMock:
        """Return a mock websocket module so tests don't need websocket-client installed."""
        mock_ws_mod = MagicMock()
        mock_ws_app_instance = MagicMock()
        mock_ws_app_instance.run_forever = MagicMock()
        mock_ws_mod.WebSocketApp.return_value = mock_ws_app_instance
        return mock_ws_mod

    @patch("data.providers.binance.write_crypto_tick")
    def test_on_tick_called_for_agg_trade(self, mock_write):
        """WebSocket on_message handler correctly parses aggTrade and calls on_tick."""
        received: list[dict] = []

        agg_trade_msg = json.dumps(
            {
                "stream": "btcusdt@aggTrade",
                "data": {
                    "e": "aggTrade",
                    "E": 1704067200000,
                    "s": "BTCUSDT",
                    "a": 99,
                    "p": "42100.00",
                    "q": "0.005",
                    "f": 100,
                    "l": 100,
                    "T": 1704067200000,
                    "m": False,
                    "M": True,
                },
            }
        )

        captured_on_message = {}

        def fake_ws_app(url, on_open, on_message, on_error, on_close):
            captured_on_message["fn"] = on_message
            return MagicMock()

        mock_ws_mod = MagicMock()
        mock_ws_mod.WebSocketApp.side_effect = fake_ws_app

        import sys

        with patch.dict(sys.modules, {"websocket": mock_ws_mod}):
            p = _make_provider("BTCUSDT")
            p.stream_live(["BTC"], on_tick=lambda ticks: received.extend(ticks))

        # Directly invoke the captured on_message callback
        captured_on_message["fn"](MagicMock(), agg_trade_msg)

        mock_write.assert_called_once_with(
            "BTCUSDT",
            {
                "symbol": "BTCUSDT",
                "last_price": 42100.0,
                "quantity": 0.005,
                "timestamp": 1704067200000,
                "is_buyer_maker": False,
            },
        )
        assert len(received) == 1
        assert received[0]["last_price"] == 42100.0

    def test_stop_stream_closes_ws(self):
        p = _make_provider("BTCUSDT")
        mock_ws = MagicMock()
        p._ws_app = mock_ws
        p.stop_stream()
        mock_ws.close.assert_called_once()

    def test_stop_stream_no_ws_is_noop(self):
        p = _make_provider("BTCUSDT")
        p._ws_app = None
        p.stop_stream()  # must not raise

    def test_unknown_symbol_raises_before_connecting(self):
        import sys

        mock_ws_mod = MagicMock()
        with patch.dict(sys.modules, {"websocket": mock_ws_mod}):
            p = BinanceProvider()
            with pytest.raises(ValueError, match="No Binance pair registered"):
                p.stream_live(["UNKNOWN"])


# ---------------------------------------------------------------------------
# BinanceProvider — auth helpers raise NotImplementedError
# ---------------------------------------------------------------------------


class TestAuthHelpers:
    def test_get_login_url_raises(self):
        with pytest.raises(NotImplementedError):
            BinanceProvider().get_login_url()

    def test_refresh_access_token_raises(self):
        with pytest.raises(NotImplementedError):
            BinanceProvider().refresh_access_token("some_token")
