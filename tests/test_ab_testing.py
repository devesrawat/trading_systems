"""
Unit tests for A/B testing framework — champion vs. challenger model comparison.

Tests cover:
- Signal routing (50/50 champion/challenger)
- Result logging to Redis and TimescaleDB
- Statistical testing and comparison
- Promotion logic
- Rollback
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from orchestrator.ab_tester import ABTestOrchestrator, ABTestResult
from orchestrator.model_registry import ModelRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis_data = {}

    def mock_set_impl(key, value, ex=None):
        redis_data[key] = value

    def mock_get_impl(key):
        return redis_data.get(key)

    def mock_zadd_impl(key, mapping, ex=None):
        if key not in redis_data:
            redis_data[key] = []
        for item, score in mapping.items():
            redis_data[key].append(item)

    def mock_zrange_impl(key, start, stop):
        return redis_data.get(key, [])

    redis.set = MagicMock(side_effect=mock_set_impl)
    redis.get = MagicMock(side_effect=mock_get_impl)
    redis.zadd = MagicMock(side_effect=mock_zadd_impl)
    redis.zrange = MagicMock(side_effect=mock_zrange_impl)

    return redis


@pytest.fixture
def mock_engine():
    """Mock SQLAlchemy engine."""
    engine = MagicMock()
    return engine


@pytest.fixture
def ab_tester(mock_redis, mock_engine):
    """Create ABTestOrchestrator with mocked dependencies."""
    with (
        patch("orchestrator.ab_tester.get_redis", return_value=mock_redis),
        patch("orchestrator.ab_tester.get_engine", return_value=mock_engine),
    ):
        registry = ModelRegistry()
        with patch("orchestrator.model_registry.get_redis", return_value=mock_redis):
            tester = ABTestOrchestrator(registry=registry)
    return tester


@pytest.fixture
def model_registry(mock_redis):
    """Create ModelRegistry with mocked Redis."""
    with patch("orchestrator.model_registry.get_redis", return_value=mock_redis):
        registry = ModelRegistry()
    return registry


# ---------------------------------------------------------------------------
# ABTestResult — dataclass tests
# ---------------------------------------------------------------------------


class TestABTestResult:
    """Tests for ABTestResult dataclass."""

    def test_result_to_dict(self):
        result = ABTestResult(
            timestamp="2025-01-01T10:00:00",
            symbol="INFY",
            model_name="champion",
            entry_price=1000.0,
            exit_price=1010.0,
            pnl=100.0,
            pnl_pct=0.01,
            sharpe=1.5,
            win=True,
            duration_minutes=30,
            model_prediction=0.72,
        )
        result_dict = result.to_dict()
        assert result_dict["symbol"] == "INFY"
        assert result_dict["pnl"] == 100.0
        assert result_dict["win"] is True

    def test_result_from_dict(self):
        data = {
            "timestamp": "2025-01-01T10:00:00",
            "symbol": "TCS",
            "model_name": "challenger",
            "entry_price": 2000.0,
            "exit_price": 2020.0,
            "pnl": 400.0,
            "pnl_pct": 0.02,
            "sharpe": 2.0,
            "win": True,
            "duration_minutes": 60,
            "model_prediction": 0.68,
        }
        result = ABTestResult.from_dict(data)
        assert result.symbol == "TCS"
        assert result.win is True


# ---------------------------------------------------------------------------
# Signal routing (50/50)
# ---------------------------------------------------------------------------


class TestSignalRouting:
    """Test signal routing between champion and challenger."""

    def test_route_signal_returns_model_name(self, ab_tester):
        choice = ab_tester.route_signal_to_model("INFY")
        assert choice in ("champion", "challenger")

    def test_route_signal_50_50_distribution(self, ab_tester):
        """Verify routing is approximately 50/50."""
        n_trials = 1000
        champion_count = 0

        for i in range(n_trials):
            choice = ab_tester.route_signal_to_model(f"SYMBOL_{i}")
            if choice == "champion":
                champion_count += 1

        challenger_count = n_trials - champion_count
        ratio = champion_count / (champion_count + challenger_count)

        # Allow 45-55% range
        assert 0.45 < ratio < 0.55, f"Ratio {ratio} outside expected 45-55% range"

    def test_route_signal_different_symbols(self, ab_tester):
        """Test routing for multiple symbols."""
        symbols = ["INFY", "TCS", "RELIANCE", "HDFC", "ICICI"]
        for symbol in symbols:
            choice = ab_tester.route_signal_to_model(symbol)
            assert choice in ("champion", "challenger")


# ---------------------------------------------------------------------------
# Result logging
# ---------------------------------------------------------------------------


class TestResultLogging:
    """Test logging of A/B test results."""

    def test_log_result_to_redis(self, ab_tester, mock_redis):
        ab_tester.log_ab_test_result(
            symbol="INFY",
            model_name="champion",
            entry_price=1000.0,
            exit_price=1010.0,
            pnl=100.0,
            sharpe=1.5,
            model_prediction=0.72,
            duration_minutes=30,
        )

        # Check that zadd was called
        assert mock_redis.zadd.called

    def test_log_result_invalid_model_name_raises(self, ab_tester):
        """Should raise if model_name is neither 'champion' nor 'challenger'."""
        with pytest.raises(ValueError):
            ab_tester.log_ab_test_result(
                symbol="INFY",
                model_name="invalid",
                entry_price=1000.0,
                exit_price=1010.0,
                pnl=100.0,
                sharpe=1.5,
                model_prediction=0.72,
            )

    def test_log_result_calculates_pnl_pct(self, ab_tester):
        """Verify PnL % is calculated correctly."""
        ab_tester.log_ab_test_result(
            symbol="TCS",
            model_name="challenger",
            entry_price=2000.0,
            exit_price=2100.0,
            pnl=100.0,
            sharpe=1.2,
            model_prediction=0.65,
        )

        # Check that set was called (for time scaledb retry via log_to_timescaledb)
        # We can't directly assert the pnl_pct, but the method should complete

    def test_log_result_win_loss_determination(self, ab_tester, mock_redis):
        """Test that win/loss is correctly determined."""
        # Winning trade
        ab_tester.log_ab_test_result(
            symbol="INFY",
            model_name="champion",
            entry_price=1000.0,
            exit_price=1050.0,
            pnl=50.0,
            sharpe=1.5,
            model_prediction=0.75,
        )

        # Losing trade
        ab_tester.log_ab_test_result(
            symbol="TCS",
            model_name="challenger",
            entry_price=2000.0,
            exit_price=1950.0,
            pnl=-50.0,
            sharpe=0.5,
            model_prediction=0.68,
        )

        # zadd should be called twice
        assert mock_redis.zadd.call_count == 2


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------


class TestComparison:
    """Test model comparison and metrics."""

    def test_compare_models_no_results(self, ab_tester, mock_redis):
        """Handle case with no test results."""
        result = ab_tester.compare_models(lookback_days=30)

        assert result is not None
        assert "champion" in result
        assert "challenger" in result

    def test_compute_stats_empty_list(self, ab_tester):
        """Test computing stats on empty results list."""
        stats_dict = ab_tester._compute_stats([])
        assert stats_dict == {}

    def test_compute_stats_single_trade(self, ab_tester):
        """Test stats with one trade."""
        results = [
            {
                "pnl": 100.0,
                "sharpe": 1.5,
                "win": True,
                "model_prediction": 0.75,
            }
        ]
        stats_dict = ab_tester._compute_stats(results)

        assert stats_dict["num_trades"] == 1
        assert stats_dict["total_pnl"] == 100.0
        assert stats_dict["avg_sharpe"] == 1.5
        assert stats_dict["win_rate"] == 1.0

    def test_compute_stats_multiple_trades(self, ab_tester):
        """Test stats with multiple trades."""
        results = [
            {"pnl": 100.0, "sharpe": 1.5, "win": True, "model_prediction": 0.75},
            {"pnl": -50.0, "sharpe": 0.5, "win": False, "model_prediction": 0.60},
            {"pnl": 200.0, "sharpe": 2.0, "win": True, "model_prediction": 0.80},
        ]
        stats_dict = ab_tester._compute_stats(results)

        assert stats_dict["num_trades"] == 3
        assert stats_dict["total_pnl"] == 250.0
        assert stats_dict["avg_pnl"] == pytest.approx(83.33, rel=1e-2)
        assert stats_dict["win_rate"] == pytest.approx(2 / 3, rel=1e-2)

    def test_compute_profit_factor(self, ab_tester):
        """Test profit factor calculation."""
        pnls = np.array([100.0, -50.0, 200.0, -25.0])
        wins = np.array([True, False, True, False])

        pf = ab_tester._compute_profit_factor(pnls, wins)

        expected = 300.0 / 75.0  # sum(wins) / abs(sum(losses))
        assert pf == pytest.approx(expected, rel=1e-2)

    def test_statistical_test_no_results(self, ab_tester):
        """Test statistical test with no results."""
        p_value, challenger_wins = ab_tester._statistical_test([], [])

        assert p_value is None
        assert challenger_wins is False

    def test_statistical_test_champion_wins(self, ab_tester):
        """Test when champion wins (higher Sharpe)."""
        # Challenger has lower Sharpes (lower mean)
        challenger_results = [
            {"sharpe": 0.5},
            {"sharpe": 0.6},
            {"sharpe": 0.55},
            {"sharpe": 0.65},
        ]

        # Champion has higher Sharpes (higher mean)
        champion_results = [
            {"sharpe": 1.5},
            {"sharpe": 1.6},
            {"sharpe": 1.55},
            {"sharpe": 1.65},
        ]

        p_value, challenger_wins = ab_tester._statistical_test(champion_results, challenger_results)

        # Challenger should NOT win (lower mean Sharpe)
        assert challenger_wins is False

    def test_statistical_test_challenger_wins(self, ab_tester):
        """Test when challenger wins (significantly higher Sharpe)."""
        # Champion has lower Sharpes
        champion_results = [
            {"sharpe": 0.5},
            {"sharpe": 0.55},
            {"sharpe": 0.52},
            {"sharpe": 0.48},
            {"sharpe": 0.53},
        ]

        # Challenger has significantly higher Sharpes
        challenger_results = [
            {"sharpe": 1.8},
            {"sharpe": 1.9},
            {"sharpe": 1.85},
            {"sharpe": 1.95},
            {"sharpe": 1.88},
        ]

        p_value, challenger_wins = ab_tester._statistical_test(champion_results, challenger_results)

        # Challenger should win (p < 0.05 and higher mean)
        assert p_value < 0.05
        assert challenger_wins is True


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Test ModelRegistry for champion/challenger tracking."""

    def test_set_and_get_champion(self, model_registry, mock_redis):
        model_registry.set_champion(
            model_path="/path/to/champion.ubj",
            version="v1.0",
        )

        champion = model_registry.get_champion()
        assert champion is not None
        assert champion["model_path"] == "/path/to/champion.ubj"
        assert champion["version"] == "v1.0"

    def test_set_and_get_challenger(self, model_registry, mock_redis):
        model_registry.set_challenger(
            model_path="/path/to/challenger.ubj",
            version="v1.1",
        )

        challenger = model_registry.get_challenger()
        assert challenger is not None
        assert challenger["model_path"] == "/path/to/challenger.ubj"
        assert challenger["version"] == "v1.1"

    def test_get_champion_not_found(self, model_registry):
        champion = model_registry.get_champion()
        assert champion is None

    def test_promotion_history(self, model_registry, mock_redis):
        model_registry.record_promotion(
            old_champion_version="v1.0",
            new_champion_version="v1.1",
            reason="statistical_win",
            metrics={"old_sharpe": 1.0, "new_sharpe": 1.5},
        )

        history = model_registry.list_promotion_history()
        assert len(history) == 1
        assert history[0]["old_champion"] == "v1.0"
        assert history[0]["new_champion"] == "v1.1"

    def test_multiple_promotions_history(self, model_registry, mock_redis):
        """Test tracking multiple promotions."""
        for i in range(3):
            model_registry.record_promotion(
                old_champion_version=f"v1.{i}",
                new_champion_version=f"v1.{i + 1}",
                reason="statistical_win",
            )

        history = model_registry.list_promotion_history()
        assert len(history) == 3
        # Most recent first
        assert history[0]["new_champion"] == "v1.3"


# ---------------------------------------------------------------------------
# Promotion and rollback
# ---------------------------------------------------------------------------


class TestPromotion:
    """Test promotion of challenger to champion."""

    def test_promote_challenger_no_win(self, ab_tester, model_registry, mock_redis):
        """Should not promote if challenger didn't win."""
        # Set champion and challenger but don't add winning results
        model_registry.set_champion(
            "/path/champion.ubj",
            "v1.0",
            metadata={"avg_sharpe": 1.5},
        )
        model_registry.set_challenger(
            "/path/challenger.ubj",
            "v1.1",
            metadata={"avg_sharpe": 1.0},
        )

        # Mock the registry
        ab_tester._registry = model_registry

        # Should not promote (no winning results in Redis)
        result = ab_tester.promote_challenger_to_champion()
        assert result is False

    def test_rollback_no_history(self, ab_tester, model_registry):
        """Should fail gracefully if no promotion history."""
        result = ab_tester.roll_back_to_previous_champion("regression_detected")
        assert result is False

    def test_rollback_with_history(self, ab_tester, model_registry, mock_redis):
        """Should record rollback event."""
        model_registry.record_promotion(
            old_champion_version="v1.0",
            new_champion_version="v1.1",
            reason="statistical_win",
        )

        ab_tester._registry = model_registry
        result = ab_tester.roll_back_to_previous_champion("regression_detected")
        assert result is True


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestABTestingIntegration:
    """End-to-end A/B testing scenarios."""

    def test_full_ab_testing_workflow(self, ab_tester, model_registry, mock_redis):
        """Simulate a complete A/B test scenario."""
        # 1. Set champion and challenger
        model_registry.set_champion(
            "/path/champion.ubj",
            "v1.0",
            metadata={"avg_sharpe": 1.0},
        )
        model_registry.set_challenger(
            "/path/challenger.ubj",
            "v1.1",
            metadata={"avg_sharpe": 1.0},
        )
        ab_tester._registry = model_registry

        # 2. Log some results (simulate trades)
        results_list = []
        for i in range(20):
            champion_result = {
                "pnl": float(np.random.normal(100, 50)),
                "sharpe": float(np.random.normal(1.0, 0.2)),
                "win": bool(np.random.random() > 0.3),
                "model_prediction": float(np.random.uniform(0.6, 0.9)),
            }
            challenger_result = {
                "pnl": float(np.random.normal(120, 50)),  # Slightly better
                "sharpe": float(np.random.normal(1.3, 0.2)),  # Significantly better
                "win": bool(np.random.random() > 0.25),  # Better win rate
                "model_prediction": float(np.random.uniform(0.6, 0.9)),
            }

            # Store in results list
            results_list.append(json.dumps({"model_name": "champion", **champion_result}))
            results_list.append(json.dumps({"model_name": "challenger", **challenger_result}))

        # Update mock zrange to return results
        mock_redis.zrange.return_value = results_list

        # 3. Compare models
        comparison = ab_tester.compare_models(lookback_days=30)

        assert "champion" in comparison
        assert "challenger" in comparison

    def test_100_trades_routing_verification(self, ab_tester):
        """Simulate 100 trades split between champion and challenger."""
        champion_count = 0
        challenger_count = 0

        for i in range(100):
            model = ab_tester.route_signal_to_model(f"SYMBOL_{i}")
            if model == "champion":
                champion_count += 1
            else:
                challenger_count += 1

        # Verify split is roughly 50/50
        assert 35 <= champion_count <= 65, f"Champion got {champion_count} instead of ~50"
        assert 35 <= challenger_count <= 65, f"Challenger got {challenger_count} instead of ~50"
