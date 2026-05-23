"""
Tests for regime alignment logic in AlphaEngine.

The regime stub was filled in as part of the Hermes dual-engine integration.
These tests verify that different regime states correctly adjust the alpha
multiplier in the expected direction.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from signals.alpha_composite import AlphaEngine


@pytest.fixture
def engine_no_sector_no_options():
    """
    AlphaEngine with sector and options signals disabled (return neutral 0.0).
    This isolates regime logic.
    """
    with (
        patch("signals.alpha_composite.SectorRanker.is_top_sector", return_value=False),
        patch("signals.alpha_composite.SectorRanker.is_bottom_sector", return_value=False),
        patch(
            "signals.alpha_composite.OptionsFlowAnalyzer.get_sentiment_score",
            return_value=0.0,
        ),
    ):
        yield AlphaEngine()


# ------------------------------------------------------------------
# BUY side regime
# ------------------------------------------------------------------


def test_trending_bull_boosts_buy(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bull", "BUY")
    assert mult > 1.0, "trending_bull should boost BUY multiplier above 1.0"


def test_trending_bear_suppresses_buy(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bear", "BUY")
    assert mult < 1.0, "trending_bear should suppress BUY multiplier below 1.0"


def test_choppy_suppresses_buy(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "choppy", "BUY")
    assert mult < 1.0, "choppy regime should suppress BUY multiplier"


def test_high_vol_reduces_buy(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "high_vol", "BUY")
    assert mult < 1.0, "high_vol regime should reduce BUY multiplier"


def test_normal_regime_neutral_buy(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "normal", "BUY")
    assert mult == pytest.approx(1.0), "normal regime should leave BUY multiplier unchanged"


# ------------------------------------------------------------------
# SELL side regime
# ------------------------------------------------------------------


def test_trending_bear_boosts_sell(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bear", "SELL")
    assert mult > 1.0, "trending_bear should boost SELL multiplier"


def test_trending_bull_suppresses_sell(engine_no_sector_no_options):
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bull", "SELL")
    assert mult < 1.0, "trending_bull should suppress SELL multiplier"


# ------------------------------------------------------------------
# Bounds
# ------------------------------------------------------------------


def test_multiplier_clipped_to_minimum(engine_no_sector_no_options):
    """Multiplier must never fall below 0.5 even in worst-case regime."""
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bear", "BUY")
    assert mult >= 0.5


def test_multiplier_clipped_to_maximum(engine_no_sector_no_options):
    """Multiplier must never exceed 1.5."""
    mult = engine_no_sector_no_options.calculate_multiplier("TEST", "IT", "trending_bull", "BUY")
    assert mult <= 1.5


def test_unknown_regime_neutral(engine_no_sector_no_options):
    """Unknown regime key → no adjustment (default 0.0 delta)."""
    mult = engine_no_sector_no_options.calculate_multiplier(
        "TEST", "IT", "unknown_regime_xyz", "BUY"
    )
    assert mult == pytest.approx(1.0)
