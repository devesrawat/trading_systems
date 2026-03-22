"""Unit tests for risk/breakers.py — TDD RED phase. Mocks Redis."""
from unittest.mock import MagicMock, call, patch

import pytest

from risk.breakers import CircuitBreaker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_breaker(**kwargs) -> tuple[CircuitBreaker, MagicMock]:
    with patch("risk.breakers.get_redis") as mock_get_redis:
        mock_redis = MagicMock()
        mock_redis.get.return_value = None   # no persisted state
        mock_get_redis.return_value = mock_redis
        cb = CircuitBreaker(**kwargs)
    return cb, mock_redis


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestCircuitBreakerInit:
    def test_default_limits(self):
        cb, _ = _make_breaker()
        assert cb.daily_limit == 0.03
        assert cb.weekly_limit == 0.07
        assert cb.max_consecutive_losses == 5

    def test_custom_limits(self):
        cb, _ = _make_breaker(daily_limit=0.02, weekly_limit=0.05, max_consecutive_losses=3)
        assert cb.daily_limit == 0.02
        assert cb.weekly_limit == 0.05
        assert cb.max_consecutive_losses == 3

    def test_not_halted_on_init(self):
        cb, _ = _make_breaker()
        assert not cb.is_halted()

    def test_loads_persisted_state_from_redis(self):
        import json
        state = {
            "halted": True,
            "reason": "Test halt",
            "daily_start_capital": 100_000.0,
            "peak_capital": 100_000.0,
            "consecutive_losses": 0,
            "weekly_start_capital": 100_000.0,
        }
        with patch("risk.breakers.get_redis") as mock_get_redis:
            mock_redis = MagicMock()
            mock_redis.get.return_value = json.dumps(state)
            mock_get_redis.return_value = mock_redis
            cb = CircuitBreaker()
        assert cb.is_halted()


# ---------------------------------------------------------------------------
# check() — normal operation
# ---------------------------------------------------------------------------

class TestCheck:
    def test_allows_trade_when_healthy(self):
        cb, _ = _make_breaker()
        cb._daily_start_capital = 100_000.0
        cb._peak_capital = 100_000.0
        cb._weekly_start_capital = 100_000.0
        allowed, reason = cb.check(current_capital=100_000.0)
        assert allowed
        assert reason is None

    def test_blocks_when_halted(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"):
            with patch("risk.breakers.TelegramAlerter"):
                cb.halt("manual test")
        allowed, reason = cb.check(current_capital=100_000.0)
        assert not allowed
        assert reason is not None

    def test_daily_drawdown_breach_halts(self):
        cb, _ = _make_breaker(daily_limit=0.03)
        cb._daily_start_capital = 100_000.0
        cb._peak_capital = 100_000.0
        cb._weekly_start_capital = 100_000.0
        # 3.1% daily drawdown — exceeds 3% limit
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            allowed, reason = cb.check(current_capital=96_900.0)
        assert not allowed
        assert "daily" in reason.lower()

    def test_weekly_drawdown_breach_halts(self):
        cb, _ = _make_breaker(weekly_limit=0.07)
        cb._daily_start_capital = 95_000.0
        cb._weekly_start_capital = 100_000.0
        cb._peak_capital = 100_000.0
        # 7.5% weekly drawdown — exceeds 7% limit
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            allowed, reason = cb.check(current_capital=92_500.0)
        assert not allowed
        assert "weekly" in reason.lower()

    def test_consecutive_losses_breach_halts(self):
        cb, _ = _make_breaker(max_consecutive_losses=5)
        cb._daily_start_capital = 100_000.0
        cb._weekly_start_capital = 100_000.0
        cb._peak_capital = 100_000.0
        cb._consecutive_losses = 5
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            allowed, reason = cb.check(current_capital=99_000.0)
        assert not allowed
        assert "consecutive" in reason.lower()

    def test_exactly_at_daily_limit_not_halted(self):
        cb, _ = _make_breaker(daily_limit=0.03)
        cb._daily_start_capital = 100_000.0
        cb._peak_capital = 100_000.0
        cb._weekly_start_capital = 100_000.0
        # exactly 3.0% — should NOT trigger (strictly greater than)
        allowed, _ = cb.check(current_capital=97_000.0)
        assert allowed


# ---------------------------------------------------------------------------
# halt() — one-way latch
# ---------------------------------------------------------------------------

class TestHalt:
    def test_halt_sets_is_halted(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("test reason")
        assert cb.is_halted()

    def test_halt_persists_state(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state") as mock_persist, \
             patch("risk.breakers.TelegramAlerter"):
            cb.halt("drawdown exceeded")
        mock_persist.assert_called()

    def test_halt_reason_stored(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("daily drawdown exceeded 3%")
        assert cb._halt_reason == "daily drawdown exceeded 3%"

    def test_second_halt_does_not_override_first_reason(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("first reason")
            cb.halt("second reason")
        assert cb._halt_reason == "first reason"


# ---------------------------------------------------------------------------
# reset_daily / reset_weekly
# ---------------------------------------------------------------------------

class TestResets:
    def test_reset_daily_updates_start_capital(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"):
            cb.reset_daily(current_capital=98_000.0)
        assert cb._daily_start_capital == 98_000.0

    def test_reset_daily_clears_consecutive_losses(self):
        cb, _ = _make_breaker()
        cb._consecutive_losses = 3
        with patch.object(cb, "_persist_state"):
            cb.reset_daily(current_capital=100_000.0)
        assert cb._consecutive_losses == 0

    def test_reset_daily_does_not_clear_halt(self):
        """Halts caused by drawdown must NEVER auto-clear."""
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("drawdown breach")
        with patch.object(cb, "_persist_state"):
            cb.reset_daily(current_capital=100_000.0)
        assert cb.is_halted()

    def test_reset_weekly_updates_weekly_start_capital(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"):
            cb.reset_weekly(current_capital=95_000.0)
        assert cb._weekly_start_capital == 95_000.0

    def test_reset_weekly_does_not_clear_halt(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("weekly drawdown breach")
        with patch.object(cb, "_persist_state"):
            cb.reset_weekly(current_capital=95_000.0)
        assert cb.is_halted()


# ---------------------------------------------------------------------------
# manual_reset (CLI only — clears halt)
# ---------------------------------------------------------------------------

class TestManualReset:
    def test_manual_reset_clears_halt(self):
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("test")
        with patch.object(cb, "_persist_state"):
            cb.manual_reset(current_capital=100_000.0)
        assert not cb.is_halted()

    def test_manual_reset_resets_all_state(self):
        cb, _ = _make_breaker()
        cb._consecutive_losses = 7
        with patch.object(cb, "_persist_state"), patch("risk.breakers.TelegramAlerter"):
            cb.halt("test")
        with patch.object(cb, "_persist_state"):
            cb.manual_reset(current_capital=100_000.0)
        assert cb._consecutive_losses == 0
        assert cb._daily_start_capital == 100_000.0
        assert cb._weekly_start_capital == 100_000.0


# ---------------------------------------------------------------------------
# Redis state persistence
# ---------------------------------------------------------------------------

class TestStatePersistence:
    def test_persist_state_writes_to_redis(self):
        import json
        cb, mock_redis = _make_breaker()
        with patch("risk.breakers.get_redis", return_value=mock_redis):
            cb._persist_state()
        mock_redis.set.assert_called_once()
        key, value = mock_redis.set.call_args[0]
        assert key == "trading:risk:circuit:state"
        state = json.loads(value)
        assert "halted" in state
        assert "daily_start_capital" in state
        assert "consecutive_losses" in state


# ---------------------------------------------------------------------------
# _write_circuit_event (DB audit trail)
# ---------------------------------------------------------------------------

class TestWriteCircuitEvent:
    def _mock_engine(self):
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine = MagicMock()
        mock_engine.connect.return_value = mock_conn
        return mock_engine, mock_conn

    def test_halt_writes_circuit_event(self):
        cb, _ = _make_breaker()
        engine, conn = self._mock_engine()
        with patch.object(cb, "_persist_state"), \
             patch("risk.breakers.TelegramAlerter"), \
             patch("risk.breakers.get_engine", return_value=engine):
            cb.halt("daily drawdown exceeded")
        conn.execute.assert_called_once()
        conn.commit.assert_called_once()

    def test_reset_daily_writes_circuit_event(self):
        cb, _ = _make_breaker()
        engine, conn = self._mock_engine()
        with patch.object(cb, "_persist_state"), \
             patch("risk.breakers.get_engine", return_value=engine):
            cb.reset_daily(current_capital=100_000.0)
        conn.execute.assert_called_once()

    def test_reset_weekly_writes_circuit_event(self):
        cb, _ = _make_breaker()
        engine, conn = self._mock_engine()
        with patch.object(cb, "_persist_state"), \
             patch("risk.breakers.get_engine", return_value=engine):
            cb.reset_weekly(current_capital=95_000.0)
        conn.execute.assert_called_once()

    def test_manual_reset_writes_circuit_event(self):
        cb, _ = _make_breaker()
        engine, conn = self._mock_engine()
        with patch.object(cb, "_persist_state"), \
             patch("risk.breakers.get_engine", return_value=engine):
            cb.manual_reset(current_capital=100_000.0)
        conn.execute.assert_called_once()

    def test_db_error_does_not_propagate(self):
        """DB write failure must never crash the trading loop."""
        cb, _ = _make_breaker()
        with patch.object(cb, "_persist_state"), \
             patch("risk.breakers.get_engine", side_effect=Exception("DB down")):
            cb._write_circuit_event("halt", "test")  # must not raise
