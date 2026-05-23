"""
Tests for WealthArchitectScanner (Hermes dual-engine, conservative leg).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from signals.wealth_architect_scanner import WealthArchitectScanner


@pytest.fixture
def scanner() -> WealthArchitectScanner:
    return WealthArchitectScanner()


def _redis_with_data(entries: dict[str, dict]) -> MagicMock:
    """Build a mock Redis client whose .get() returns JSON for known keys."""
    mock_redis = MagicMock()

    def _get(key: str):
        symbol = key.removeprefix("FUND:")
        if symbol in entries:
            return json.dumps(entries[symbol]).encode()
        return None

    mock_redis.get.side_effect = _get
    return mock_redis


# ------------------------------------------------------------------
# Inclusion / exclusion criteria
# ------------------------------------------------------------------


def test_includes_low_pe_high_roe(scanner):
    """PE < sector_avg AND ROE > 15 → included."""
    redis = _redis_with_data(
        {
            "INFY": {"pe": 18.0, "roe": 25.0, "sector_avg_pe": 22.0, "sector": "IT"},
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["INFY"])
    assert len(results) == 1
    r = results[0]
    assert r["symbol"] == "INFY"
    assert r["roe"] == 25.0
    assert r["pe"] == 18.0
    assert r["pe_discount_pct"] == pytest.approx(18.2, abs=0.1)


def test_excludes_high_pe(scanner):
    """PE >= sector_avg → excluded."""
    redis = _redis_with_data(
        {
            "INFY": {"pe": 28.0, "roe": 25.0, "sector_avg_pe": 22.0, "sector": "IT"},
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["INFY"])
    assert results == []


def test_excludes_low_roe(scanner):
    """ROE < 15 → excluded even if PE is attractive."""
    redis = _redis_with_data(
        {
            "SBIN": {"pe": 8.0, "roe": 10.0, "sector_avg_pe": 14.0, "sector": "Banking"},
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["SBIN"])
    assert results == []


def test_skips_missing_cache_entry(scanner):
    """Symbols with no Redis entry are silently skipped."""
    redis = _redis_with_data({})  # empty — all symbols missing
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["RELIANCE", "TCS"])
    assert results == []


def test_fallback_pe_cap_when_sector_avg_absent(scanner):
    """sector_avg_pe absent → use FALLBACK_PE_CAP=25. PE < 25 passes."""
    redis = _redis_with_data(
        {
            "TCS": {"pe": 22.0, "roe": 35.0, "sector": "IT"},  # no sector_avg_pe
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["TCS"])
    assert len(results) == 1
    assert results[0]["sector_avg_pe"] == scanner.fallback_pe_cap


def test_fallback_pe_cap_blocks_expensive_stock(scanner):
    """PE > FALLBACK_PE_CAP when no sector avg provided → excluded."""
    redis = _redis_with_data(
        {
            "TCS": {"pe": 30.0, "roe": 35.0, "sector": "IT"},
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["TCS"])
    assert results == []


# ------------------------------------------------------------------
# Sorting
# ------------------------------------------------------------------


def test_results_sorted_by_roe_descending(scanner):
    """Results must be sorted by ROE descending."""
    redis = _redis_with_data(
        {
            "A": {"pe": 12.0, "roe": 18.0, "sector_avg_pe": 20.0, "sector": "X"},
            "B": {"pe": 10.0, "roe": 32.0, "sector_avg_pe": 20.0, "sector": "X"},
            "C": {"pe": 14.0, "roe": 22.0, "sector_avg_pe": 20.0, "sector": "X"},
        }
    )
    with patch("signals.wealth_architect_scanner.get_redis", return_value=redis):
        results = scanner.run(["A", "B", "C"])
    assert [r["symbol"] for r in results] == ["B", "C", "A"]


# ------------------------------------------------------------------
# Resilience
# ------------------------------------------------------------------


def test_handles_corrupt_redis_value(scanner):
    """Corrupt JSON in Redis → symbol skipped, no exception propagated."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = b"NOT_JSON"
    with patch("signals.wealth_architect_scanner.get_redis", return_value=mock_redis):
        results = scanner.run(["BAD"])
    assert results == []


def test_handles_redis_connection_error(scanner):
    """Redis .get() raising an exception → symbol skipped gracefully."""
    mock_redis = MagicMock()
    mock_redis.get.side_effect = ConnectionError("Redis down")
    with patch("signals.wealth_architect_scanner.get_redis", return_value=mock_redis):
        results = scanner.run(["X"])
    assert results == []
