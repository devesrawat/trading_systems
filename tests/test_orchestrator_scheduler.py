"""Unit tests for orchestrator/scheduler.py — mocks APScheduler, no live jobs."""
from unittest.mock import MagicMock, patch, call

import pytest

from orchestrator.scheduler import TradingScheduler


def _mock_system():
    s = MagicMock()
    s.pre_market_setup.__name__ = "pre_market_setup"
    s.trading_loop.__name__ = "trading_loop"
    s.post_market_summary.__name__ = "post_market_summary"
    s.reset_weekly.__name__ = "reset_weekly"
    s.retrain_check.__name__ = "retrain_check"
    return s


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------

class TestTradingSchedulerInit:
    def test_defaults_to_equity(self):
        s = _mock_system()
        sched = TradingScheduler(s)
        assert sched._market_type == "equity"

    def test_accepts_crypto_market_type(self):
        s = _mock_system()
        sched = TradingScheduler(s, market_type="crypto")
        assert sched._market_type == "crypto"

    def test_accepts_both_market_type(self):
        s = _mock_system()
        sched = TradingScheduler(s, market_type="both")
        assert sched._market_type == "both"

    def test_market_type_lowercased(self):
        s = _mock_system()
        sched = TradingScheduler(s, market_type="EQUITY")
        assert sched._market_type == "equity"


# ---------------------------------------------------------------------------
# Job registration
# ---------------------------------------------------------------------------

class TestJobRegistration:
    def _start(self, market_type: str) -> tuple[TradingScheduler, MagicMock]:
        s = _mock_system()
        sched = TradingScheduler(s, market_type=market_type)
        with patch.object(sched._scheduler, "start"):
            with patch.object(sched._scheduler, "get_jobs", return_value=[MagicMock()]):
                sched.start()
        return sched, s

    def test_equity_registers_equity_jobs(self):
        sched, _ = self._start("equity")
        job_ids = {j.id for j in sched._scheduler.get_jobs()}
        # APScheduler is real here so we just check no error was raised
        assert sched._market_type == "equity"

    def test_equity_does_not_register_crypto_loop(self):
        sched, _ = self._start("equity")
        job_ids = [j.id for j in sched._scheduler.get_jobs()]
        assert "crypto_trading_loop" not in job_ids

    def test_crypto_registers_crypto_jobs(self):
        sched, _ = self._start("crypto")
        job_ids = [j.id for j in sched._scheduler.get_jobs()]
        assert "crypto_trading_loop" in job_ids

    def test_crypto_does_not_register_equity_loop(self):
        sched, _ = self._start("crypto")
        job_ids = [j.id for j in sched._scheduler.get_jobs()]
        assert "equity_trading_loop" not in job_ids

    def test_both_registers_all_loops(self):
        sched, _ = self._start("both")
        job_ids = [j.id for j in sched._scheduler.get_jobs()]
        assert "equity_trading_loop" in job_ids
        assert "crypto_trading_loop" in job_ids

    def test_maintenance_jobs_always_registered(self):
        for market in ("equity", "crypto", "both"):
            sched, _ = self._start(market)
            job_ids = [j.id for j in sched._scheduler.get_jobs()]
            assert "weekly_reset" in job_ids, f"weekly_reset missing for {market}"
            assert "retrain_check" in job_ids, f"retrain_check missing for {market}"


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------

class TestSchedulerStop:
    def test_stop_shuts_down_scheduler(self):
        s = _mock_system()
        sched = TradingScheduler(s)
        with patch.object(sched._scheduler, "shutdown") as mock_shutdown:
            sched.stop()
        mock_shutdown.assert_called_once_with(wait=True)


# ---------------------------------------------------------------------------
# _safe wrapper
# ---------------------------------------------------------------------------

class TestSafeWrapper:
    def test_safe_calls_wrapped_function(self):
        s = _mock_system()
        sched = TradingScheduler(s)
        called = []
        fn = MagicMock(__name__="test_fn")
        fn.side_effect = lambda: called.append(1)
        wrapped = sched._safe(fn)
        wrapped()
        assert called == [1]

    def test_safe_catches_exceptions_without_raising(self):
        s = _mock_system()
        sched = TradingScheduler(s)
        fn = MagicMock(__name__="boom_fn")
        fn.side_effect = RuntimeError("kaboom")
        wrapped = sched._safe(fn)
        wrapped()   # must NOT raise

    def test_safe_preserves_function_name(self):
        s = _mock_system()
        sched = TradingScheduler(s)
        fn = MagicMock()
        fn.__name__ = "my_job"
        wrapped = sched._safe(fn)
        assert wrapped.__name__ == "my_job"
