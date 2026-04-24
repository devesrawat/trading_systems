"""
Unit tests for signal contracts and router.

Tests:
- Valid signal construction
- Strategy dict to Signal normalization
- ML probability to Signal normalization
- JSON serialization round-trip
- Required field validation
- Invalid mode rejection
"""

import json
from datetime import UTC, datetime

import pytest

from signals.contracts import Direction, EntrySpec, RiskSpec, Signal, SignalType
from signals.signal_router import (
    normalize_fundamental_rank,
    normalize_ml_signal,
    normalize_strategy_result,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_strategy_result() -> dict:
    """Basic VCP strategy result."""
    return {
        "symbol": "INFY",
        "strategy": "vcp",
        "current_price": 1450.0,
        "pivot_buy": 1500.0,
        "distance_to_pivot_pct": 3.45,
        "contractions": 3,
        "swing_ranges": [8.5, 7.2, 5.1],
        "volume_dry_up": True,
    }


@pytest.fixture
def rs_breakout_result() -> dict:
    """RS Breakout strategy result."""
    return {
        "symbol": "RELIANCE",
        "strategy": "rs_breakout",
        "current_price": 2850.0,
        "high_52w": 2900.0,
        "return_12m_pct": 25.5,
        "volume_ratio": 1.68,
        "above_sma200": True,
    }


# ---------------------------------------------------------------------------
# Signal contract validation
# ---------------------------------------------------------------------------


class TestSignalConstruction:
    """Test valid Signal construction."""

    def test_minimal_signal(self):
        """Create signal with only required fields."""
        signal = Signal(
            signal_id="test_1",
            timestamp=datetime.now(tz=UTC),
            symbol="INFY.NS",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.7,
            score=0.75,
        )
        assert signal.signal_id == "test_1"
        assert signal.symbol == "INFY.NS"
        assert signal.confidence == 0.7
        assert signal.score == 0.75

    def test_signal_with_entry_and_risk(self):
        """Create signal with entry and risk specifications."""
        entry = EntrySpec(
            entry_price=1500.0,
            stop_price=1400.0,
            target_price=1700.0,
            invalidation_price=1550.0,
        )
        risk = RiskSpec(size_hint_pct=0.015, liquidity_score=0.8, volatility_score=0.6)

        signal = Signal(
            signal_id="test_2",
            timestamp=datetime.now(tz=UTC),
            symbol="RELIANCE",
            strategy_name="vcp",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.65,
            score=0.70,
            entry=entry,
            risk=risk,
        )

        assert signal.entry is not None
        assert signal.entry.entry_price == 1500.0
        assert signal.risk is not None
        assert signal.risk.size_hint_pct == 0.015

    def test_symbol_uppercased(self):
        """Symbol should be normalized to uppercase."""
        signal = Signal(
            signal_id="test_3",
            timestamp=datetime.now(tz=UTC),
            symbol="infy",
            strategy_name="test",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.5,
            score=0.5,
        )
        assert signal.symbol == "INFY"

    def test_all_directions(self):
        """Test all direction enums."""
        for direction in Direction:
            signal = Signal(
                signal_id="test_dir",
                timestamp=datetime.now(tz=UTC),
                symbol="TCS",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=direction,
                confidence=0.5,
                score=0.5,
            )
            assert signal.direction == direction

    def test_all_signal_types(self):
        """Test all signal type enums."""
        for sig_type in SignalType:
            signal = Signal(
                signal_id="test_type",
                timestamp=datetime.now(tz=UTC),
                symbol="HSBC",
                strategy_name="test",
                signal_type=sig_type,
                direction=Direction.long,
                confidence=0.5,
                score=0.5,
            )
            assert signal.signal_type == sig_type


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestSignalValidation:
    """Test Signal validation rules."""

    def test_invalid_confidence_too_high(self):
        """Confidence > 1 should fail."""
        with pytest.raises(ValueError):
            Signal(
                signal_id="test",
                timestamp=datetime.now(tz=UTC),
                symbol="INFY",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=Direction.long,
                confidence=1.5,  # Invalid
                score=0.5,
            )

    def test_invalid_confidence_negative(self):
        """Confidence < 0 should fail."""
        with pytest.raises(ValueError):
            Signal(
                signal_id="test",
                timestamp=datetime.now(tz=UTC),
                symbol="INFY",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=Direction.long,
                confidence=-0.1,  # Invalid
                score=0.5,
            )

    def test_invalid_mode(self):
        """Invalid mode should fail."""
        with pytest.raises(ValueError):
            Signal(
                signal_id="test",
                timestamp=datetime.now(tz=UTC),
                symbol="INFY",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=Direction.long,
                confidence=0.5,
                score=0.5,
                mode="invalid_mode",
            )

    def test_valid_modes(self):
        """All valid modes should work."""
        for mode in ["research", "watchlist", "paper", "live"]:
            signal = Signal(
                signal_id="test",
                timestamp=datetime.now(tz=UTC),
                symbol="INFY",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=Direction.long,
                confidence=0.5,
                score=0.5,
                mode=mode,
            )
            assert signal.mode == mode

    def test_empty_symbol(self):
        """Empty symbol should fail."""
        with pytest.raises(ValueError):
            Signal(
                signal_id="test",
                timestamp=datetime.now(tz=UTC),
                symbol="",
                strategy_name="test",
                signal_type=SignalType.scanner_hit,
                direction=Direction.long,
                confidence=0.5,
                score=0.5,
            )

    def test_entry_spec_validation(self):
        """EntrySpec must have valid prices."""
        with pytest.raises(ValueError):
            EntrySpec(
                entry_price=-100,  # Invalid
                stop_price=1400,
                target_price=1700,
            )

    def test_risk_spec_size_cap(self):
        """Risk size hint must not exceed 2%."""
        with pytest.raises(ValueError):
            RiskSpec(size_hint_pct=2.5)


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


class TestJSONSerialization:
    """Test Signal serialization round-trips."""

    def test_signal_to_dict(self):
        """Signal.to_dict() should return JSON-serializable dict."""
        signal = Signal(
            signal_id="test_json",
            timestamp=datetime.now(tz=UTC),
            symbol="WIPRO",
            strategy_name="rs_breakout",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.70,
            score=0.75,
            metadata={"note": "test signal"},
        )

        d = signal.to_dict()
        assert isinstance(d, dict)
        assert d["signal_id"] == "test_json"
        assert d["symbol"] == "WIPRO"
        assert json.dumps(d)  # Should be JSON-serializable

    def test_signal_from_dict(self):
        """Signal.from_dict() should reconstruct signal."""
        original_dict = {
            "signal_id": "test_roundtrip",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "symbol": "infy",
            "strategy_name": "vcp",
            "signal_type": "scanner_hit",
            "direction": "long",
            "confidence": 0.65,
            "score": 0.68,
            "mode": "paper",
        }

        signal = Signal.from_dict(original_dict)
        assert signal.signal_id == "test_roundtrip"
        assert signal.symbol == "INFY"
        assert signal.confidence == 0.65
        assert signal.mode == "paper"

    def test_roundtrip_with_entry_and_risk(self):
        """Signal with entry/risk should roundtrip through JSON."""
        entry = EntrySpec(
            entry_price=1500.0,
            stop_price=1400.0,
            target_price=1700.0,
        )
        risk = RiskSpec(size_hint_pct=0.01, liquidity_score=0.75)

        signal = Signal(
            signal_id="roundtrip",
            timestamp=datetime.now(tz=UTC),
            symbol="RELIANCE",
            strategy_name="vcp",
            signal_type=SignalType.scanner_hit,
            direction=Direction.long,
            confidence=0.70,
            score=0.72,
            entry=entry,
            risk=risk,
        )

        d = signal.to_dict()
        reconstructed = Signal.from_dict(d)

        assert reconstructed.entry is not None
        assert reconstructed.entry.entry_price == 1500.0
        assert reconstructed.risk is not None
        assert reconstructed.risk.size_hint_pct == 0.01


# ---------------------------------------------------------------------------
# Strategy normalization
# ---------------------------------------------------------------------------


class TestStrategyNormalization:
    """Test normalize_strategy_result() function."""

    def test_vcp_normalization(self, basic_strategy_result):
        """Normalize VCP strategy result."""
        signal = normalize_strategy_result(
            strategy_result=basic_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            confidence=0.60,
            mode="paper",
        )

        assert signal.signal_type == SignalType.scanner_hit
        assert signal.symbol == "INFY"
        assert signal.strategy_name == "vcp"
        assert signal.direction == Direction.long
        assert signal.mode == "paper"
        assert signal.confidence == 0.60

    def test_rs_breakout_normalization(self, rs_breakout_result):
        """Normalize RS Breakout strategy result."""
        signal = normalize_strategy_result(
            strategy_result=rs_breakout_result,
            symbol="RELIANCE",
            strategy_name="rs_breakout",
            confidence=0.72,
            mode="watchlist",
        )

        assert signal.symbol == "RELIANCE"
        assert signal.strategy_name == "rs_breakout"
        assert signal.confidence == 0.72
        assert signal.mode == "watchlist"
        assert signal.raw_payload == rs_breakout_result

    def test_strategy_result_missing_symbol(self):
        """Missing symbol should raise."""
        with pytest.raises(ValueError):
            normalize_strategy_result(
                strategy_result={},
                symbol="",
                strategy_name="vcp",
            )

    def test_strategy_result_with_custom_entry(self):
        """Custom entry spec should be preserved."""
        entry = EntrySpec(
            entry_price=2900.0,
            stop_price=2800.0,
            target_price=3100.0,
        )
        signal = normalize_strategy_result(
            strategy_result={"symbol": "INFY", "strategy": "vcp"},
            symbol="INFY",
            strategy_name="vcp",
            entry_spec=entry,
        )

        assert signal.entry == entry

    def test_strategy_result_with_custom_risk(self):
        """Custom risk spec should be preserved."""
        risk = RiskSpec(size_hint_pct=0.015, liquidity_score=0.85)
        signal = normalize_strategy_result(
            strategy_result={"symbol": "TCS", "strategy": "vcp"},
            symbol="TCS",
            strategy_name="vcp",
            risk_spec=risk,
        )

        assert signal.risk == risk

    def test_strategy_result_inferred_entry_spec(self):
        """Entry spec should be inferred from strategy result."""
        result = {
            "symbol": "INFY",
            "strategy": "vcp",
            "current_price": 1450.0,
            "pivot_buy": 1500.0,
            "stop_price": 1400.0,
            "target_price": 1700.0,
        }
        signal = normalize_strategy_result(
            strategy_result=result,
            symbol="INFY",
            strategy_name="vcp",
        )

        assert signal.entry is not None
        assert signal.entry.entry_price == 1500.0
        assert signal.entry.stop_price == 1400.0
        assert signal.entry.target_price == 1700.0


# ---------------------------------------------------------------------------
# ML signal normalization
# ---------------------------------------------------------------------------


class TestMLSignalNormalization:
    """Test normalize_ml_signal() function."""

    def test_ml_signal_high_probability(self):
        """High probability should yield long direction."""
        signal = normalize_ml_signal(
            probability=0.75,
            symbol="infy",
            features={"rsi": 65, "macd_sign": 1},
            model_version="2.1",
        )

        assert signal.signal_type == SignalType.ml_prediction
        assert signal.symbol == "INFY"
        assert signal.direction == Direction.long
        assert signal.confidence == 0.75
        assert signal.score == 0.75

    def test_ml_signal_low_probability(self):
        """Low probability should yield neutral direction."""
        signal = normalize_ml_signal(
            probability=0.35,
            symbol="tcs",
            features={},
            model_version="2.1",
        )

        assert signal.direction == Direction.neutral
        assert signal.confidence == 0.35

    def test_ml_signal_probability_boundary_long(self):
        """Probability at 0.65 (threshold) should be long."""
        signal = normalize_ml_signal(
            probability=0.65,
            symbol="hsbc",
            features={},
        )

        assert signal.direction == Direction.long

    def test_ml_signal_probability_just_below_threshold(self):
        """Probability just below 0.65 should be neutral."""
        signal = normalize_ml_signal(
            probability=0.64,
            symbol="bajajfinsv",
            features={},
        )

        assert signal.direction == Direction.neutral

    def test_ml_signal_invalid_probability(self):
        """Probability > 1 should raise."""
        with pytest.raises(ValueError):
            normalize_ml_signal(
                probability=1.5,
                symbol="infy",
                features={},
            )

    def test_ml_signal_custom_timestamp(self):
        """Custom timestamp should be preserved."""
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        signal = normalize_ml_signal(
            probability=0.72,
            symbol="reliance",
            features={},
            timestamp=ts,
        )

        assert signal.timestamp == ts

    def test_ml_signal_custom_entry(self):
        """Custom entry spec should be preserved."""
        entry = EntrySpec(
            entry_price=1500.0,
            stop_price=1400.0,
            target_price=1700.0,
        )
        signal = normalize_ml_signal(
            probability=0.70,
            symbol="infy",
            features={},
            entry_spec=entry,
        )

        assert signal.entry == entry


# ---------------------------------------------------------------------------
# Fundamental rank normalization
# ---------------------------------------------------------------------------


class TestFundamentalRankNormalization:
    """Test normalize_fundamental_rank() function."""

    def test_fundamental_rank_creation(self):
        """Create fundamental rank signal."""
        signal = normalize_fundamental_rank(
            symbol="infy",
            rank=5,
            multibagger_score=0.72,
        )

        assert signal.signal_type == SignalType.fundamental_rank
        assert signal.symbol == "INFY"
        assert signal.direction == Direction.neutral
        assert signal.rank == 5
        assert signal.confidence == 0.72
        assert signal.metadata["universe_rank"] == 5

    def test_fundamental_rank_invalid_rank_zero(self):
        """Rank must be > 0."""
        with pytest.raises(ValueError):
            normalize_fundamental_rank(
                symbol="infy",
                rank=0,
                multibagger_score=0.5,
            )

    def test_fundamental_rank_invalid_rank_negative(self):
        """Rank must be > 0."""
        with pytest.raises(ValueError):
            normalize_fundamental_rank(
                symbol="infy",
                rank=-1,
                multibagger_score=0.5,
            )

    def test_fundamental_rank_invalid_score_too_high(self):
        """Multibagger score must be in [0, 1]."""
        with pytest.raises(ValueError):
            normalize_fundamental_rank(
                symbol="infy",
                rank=1,
                multibagger_score=1.5,
            )

    def test_fundamental_rank_custom_timestamp(self):
        """Custom timestamp should be preserved."""
        ts = datetime(2024, 1, 14, 15, 0, 0, tzinfo=UTC)
        signal = normalize_fundamental_rank(
            symbol="tcs",
            rank=3,
            multibagger_score=0.85,
            timestamp=ts,
        )

        assert signal.timestamp == ts


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_tight_closes_strategy_conversion(self):
        """Simulate tight-closes strategy scan result."""
        tight_closes_result = {
            "symbol": "BAJAJFINSV",
            "strategy": "tight_closes",
            "current_price": 1620.0,
            "close_range": 0.8,
            "days_tight": 5,
            "inside_day_count": 3,
        }

        signal = normalize_strategy_result(
            strategy_result=tight_closes_result,
            symbol="BAJAJFINSV",
            strategy_name="tight_closes",
            confidence=0.55,
        )

        assert signal.symbol == "BAJAJFINSV"
        assert signal.strategy_name == "tight_closes"
        assert json.dumps(signal.to_dict())  # JSON-serializable

    def test_multiple_signals_in_sequence(self):
        """Create multiple signals and serialize/deserialize."""
        signals = []

        # Strategy signal
        s1 = normalize_strategy_result(
            strategy_result={"symbol": "INFY", "strategy": "vcp"},
            symbol="INFY",
            strategy_name="vcp",
            confidence=0.60,
        )
        signals.append(s1)

        # ML signal
        s2 = normalize_ml_signal(
            probability=0.72,
            symbol="RELIANCE",
            features={},
        )
        signals.append(s2)

        # Fundamental rank
        s3 = normalize_fundamental_rank(
            symbol="TCS",
            rank=1,
            multibagger_score=0.85,
        )
        signals.append(s3)

        # Roundtrip all through JSON
        dicts = [s.to_dict() for s in signals]
        reconstructed = [Signal.from_dict(d) for d in dicts]

        assert len(reconstructed) == 3
        assert all(isinstance(s, Signal) for s in reconstructed)
        assert reconstructed[0].symbol == "INFY"
        assert reconstructed[1].symbol == "RELIANCE"
        assert reconstructed[2].symbol == "TCS"
