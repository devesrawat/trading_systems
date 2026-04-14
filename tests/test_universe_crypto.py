"""Unit tests for data/universe_crypto.py — mocks HTTP and Redis, no live calls."""

import json
from unittest.mock import MagicMock, patch

from data.universe_crypto import (
    CryptoUniverse,
    _apply_filters,
    _include,
    _to_instrument,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _btc_item(
    rank: int = 1,
    volume: float = 20_000_000_000,
    market_cap: float = 800_000_000_000,
) -> dict:
    return {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 42000.0,
        "market_cap": market_cap,
        "total_volume": volume,
        "market_cap_rank": rank,
    }


def _eth_item(rank: int = 2) -> dict:
    return {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "current_price": 2500.0,
        "market_cap": 300_000_000_000,
        "total_volume": 10_000_000_000,
        "market_cap_rank": rank,
    }


def _stablecoin_item() -> dict:
    return {
        "id": "tether",
        "symbol": "usdt",
        "name": "Tether",
        "current_price": 1.0,
        "market_cap": 90_000_000_000,
        "total_volume": 50_000_000_000,
        "market_cap_rank": 3,
    }


# ---------------------------------------------------------------------------
# _include
# ---------------------------------------------------------------------------


class TestInclude:
    def test_accepts_bitcoin(self):
        assert _include(_btc_item())

    def test_accepts_ethereum(self):
        assert _include(_eth_item())

    def test_rejects_tether(self):
        assert not _include(_stablecoin_item())

    def test_rejects_all_known_stablecoins(self):
        stablecoin_ids = [
            "tether",
            "usd-coin",
            "dai",
            "binance-usd",
            "trueusd",
            "first-digital-usd",
            "usdd",
            "frax",
            "pax-dollar",
        ]
        for sid in stablecoin_ids:
            item = {**_btc_item(), "id": sid}
            assert not _include(item), f"{sid} should be excluded"

    def test_rejects_missing_market_cap(self):
        item = {**_btc_item(), "market_cap": None}
        assert not _include(item)

    def test_rejects_missing_volume(self):
        item = {**_btc_item(), "total_volume": None}
        assert not _include(item)


# ---------------------------------------------------------------------------
# _to_instrument
# ---------------------------------------------------------------------------


class TestToInstrument:
    def test_symbol_is_uppercase(self):
        result = _to_instrument(_btc_item())
        assert result["symbol"] == "BTC"

    def test_pair_appends_usdt(self):
        result = _to_instrument(_btc_item())
        assert result["pair"] == "BTCUSDT"

    def test_asset_class_is_crypto(self):
        result = _to_instrument(_btc_item())
        assert result["asset_class"] == "crypto"

    def test_required_fields_present(self):
        result = _to_instrument(_btc_item())
        for field in (
            "symbol",
            "pair",
            "name",
            "market_cap_usd",
            "volume_24h_usd",
            "price_usd",
            "rank",
            "asset_class",
        ):
            assert field in result

    def test_numeric_fields_are_float(self):
        result = _to_instrument(_btc_item())
        assert isinstance(result["market_cap_usd"], float)
        assert isinstance(result["volume_24h_usd"], float)
        assert isinstance(result["price_usd"], float)

    def test_rank_is_int(self):
        result = _to_instrument(_btc_item(rank=5))
        assert result["rank"] == 5
        assert isinstance(result["rank"], int)

    def test_coingecko_id_preserved(self):
        result = _to_instrument(_btc_item())
        assert result["coingecko_id"] == "bitcoin"


# ---------------------------------------------------------------------------
# _apply_filters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def _instruments(self, n: int) -> list[dict]:
        return [
            {
                "symbol": f"COIN{i}",
                "pair": f"COIN{i}USDT",
                "volume_24h_usd": float(10_000_000 - i * 1_000_000),
                "market_cap_usd": float(100_000_000 - i * 1_000_000),
                "rank": i + 1,
                "asset_class": "crypto",
            }
            for i in range(n)
        ]

    def test_respects_top_n(self):
        instruments = self._instruments(20)
        result = _apply_filters(instruments, top_n=5, min_volume_usd=0)
        assert len(result) == 5

    def test_filters_below_min_volume(self):
        instruments = self._instruments(5)
        # Only first two have volume ≥ 9M
        result = _apply_filters(instruments, top_n=10, min_volume_usd=9_000_000)
        assert all(i["volume_24h_usd"] >= 9_000_000 for i in result)

    def test_returns_empty_when_all_below_volume(self):
        instruments = self._instruments(5)
        result = _apply_filters(instruments, top_n=10, min_volume_usd=999_999_999)
        assert result == []

    def test_top_n_applied_after_volume_filter(self):
        instruments = self._instruments(10)
        result = _apply_filters(instruments, top_n=3, min_volume_usd=0)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# CryptoUniverse — cache behaviour
# ---------------------------------------------------------------------------


class TestCryptoUniverseCache:
    def _universe(self) -> CryptoUniverse:
        return CryptoUniverse(api_key=None)

    @patch("data.universe_crypto.get_redis")
    def test_get_tradeable_serves_from_cache(self, mock_get_redis):
        cached_data = [_to_instrument(_btc_item()), _to_instrument(_eth_item())]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached_data)
        mock_get_redis.return_value = mock_redis

        u = self._universe()
        result = u.get_tradeable(top_n=10, min_volume_usd=0)
        assert len(result) == 2
        assert result[0]["symbol"] == "BTC"

    @patch("data.universe_crypto.get_redis")
    def test_get_tradeable_calls_refresh_on_cache_miss(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # cache miss
        mock_redis.setex = MagicMock()
        mock_get_redis.return_value = mock_redis

        u = self._universe()
        with patch.object(u, "_fetch_markets", return_value=[_btc_item()]) as mock_fetch:
            result = u.get_tradeable(top_n=5, min_volume_usd=0)
        mock_fetch.assert_called_once()
        assert len(result) == 1

    @patch("data.universe_crypto.get_redis")
    def test_cache_error_does_not_raise(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.side_effect = Exception("redis error")
        mock_get_redis.return_value = mock_redis

        u = self._universe()
        with patch.object(u, "_fetch_markets", return_value=[_btc_item()]):
            result = u.get_tradeable()  # should not raise
        assert isinstance(result, list)

    @patch("data.universe_crypto.get_redis")
    def test_save_cache_error_does_not_raise(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_redis.get.return_value = None
        mock_redis.setex.side_effect = Exception("redis write error")
        mock_get_redis.return_value = mock_redis

        u = self._universe()
        with patch.object(u, "_fetch_markets", return_value=[_btc_item()]):
            result = u.get_tradeable()  # should not raise
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# CryptoUniverse — refresh
# ---------------------------------------------------------------------------


class TestCryptoUniverseRefresh:
    @patch("data.universe_crypto.get_redis")
    def test_refresh_excludes_stablecoins(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        u = CryptoUniverse()
        with patch.object(u, "_fetch_markets", return_value=[_btc_item(), _stablecoin_item()]):
            result = u.refresh(min_volume_usd=0)

        symbols = [i["symbol"] for i in result]
        assert "BTC" in symbols
        assert "USDT" not in symbols

    @patch("data.universe_crypto.get_redis")
    def test_refresh_writes_to_cache(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        u = CryptoUniverse()
        with patch.object(u, "_fetch_markets", return_value=[_btc_item()]):
            u.refresh()

        mock_redis.setex.assert_called_once()

    @patch("data.universe_crypto.get_redis")
    def test_refresh_empty_fetch_returns_empty(self, mock_get_redis):
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        u = CryptoUniverse()
        with patch.object(u, "_fetch_markets", return_value=[]):
            result = u.refresh()
        assert result == []


# ---------------------------------------------------------------------------
# CryptoUniverse — convenience methods
# ---------------------------------------------------------------------------


class TestCryptoUniverseConvenience:
    @patch("data.universe_crypto.get_redis")
    def test_get_instrument_map_returns_symbol_to_pair(self, mock_get_redis):
        cached = [_to_instrument(_btc_item()), _to_instrument(_eth_item())]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached)
        mock_get_redis.return_value = mock_redis

        u = CryptoUniverse()
        result = u.get_instrument_map(min_volume_usd=0)
        assert result["BTC"] == "BTCUSDT"
        assert result["ETH"] == "ETHUSDT"

    @patch("data.universe_crypto.get_redis")
    def test_get_binance_pairs_returns_list_of_pair_strings(self, mock_get_redis):
        cached = [_to_instrument(_btc_item()), _to_instrument(_eth_item())]
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(cached)
        mock_get_redis.return_value = mock_redis

        u = CryptoUniverse()
        pairs = u.get_binance_pairs(min_volume_usd=0)
        assert "BTCUSDT" in pairs
        assert "ETHUSDT" in pairs
        assert all(p.endswith("USDT") for p in pairs)
