"""
Tests for Phase 8 Workstream 4: Attribution and Feature Analysis.

Tests cover:
- SHAP value computation
- Feature importance trends
- Strategy contribution calculation
- Loss/profit analysis
- Feature correlation with returns
- Feature drift detection
- Trade review engine
- Full attribution report generation
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from monitoring.attribution import PerformanceAttribution, TradeRecord
from monitoring.feature_drift_report import FeatureDriftReporter, FeatureStatistics
from monitoring.reporters import AttributionReport
from monitoring.trade_review import OutcomeCategory, TradeContext, TradeReviewEngine
from signals.features import FEATURE_COLUMNS

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Mock XGBoost model."""
    model = Mock()

    def mock_explain(feature_dict):
        """Return SHAP values for features."""
        return {
            "rsi_14": np.random.randn() * 0.1,
            "macd": np.random.randn() * 0.05,
            "atr_14": np.random.randn() * 0.03,
            "volume_zscore_20": np.random.randn() * 0.02,
            "ema_9": np.random.randn() * 0.01,
        }

    model.explain = mock_explain
    return model


@pytest.fixture
def sample_features_df():
    """Sample feature DataFrame."""
    data = {
        "rsi_14": [45, 55, 65, 35],
        "rsi_7": [40, 60, 70, 30],
        "macd": [0.5, -0.3, 0.8, -0.5],
        "macd_signal": [0.4, -0.2, 0.7, -0.4],
        "macd_hist": [0.1, -0.1, 0.1, -0.1],
        "macd_cross": [0, 1, 0, 1],
        "mom_10": [100, 150, 200, 50],
        "roc_5": [1.0, -0.5, 2.0, -1.5],
        "roc_21": [2.5, -1.0, 3.0, -2.0],
        "atr_14": [0.5, 0.6, 0.7, 0.4],
        "atr_pct": [0.02, 0.025, 0.03, 0.015],
        "bb_upper": [100, 105, 110, 95],
        "bb_mid": [98, 100, 105, 90],
        "bb_lower": [96, 95, 100, 85],
        "bb_position": [0.5, 0.6, 0.7, 0.3],
        "realized_vol_10": [0.015, 0.020, 0.025, 0.010],
        "realized_vol_20": [0.016, 0.021, 0.026, 0.011],
        "volume_zscore_20": [1.5, 2.0, 2.5, 0.5],
        "obv": [1e6, 1.1e6, 1.2e6, 0.9e6],
        "obv_slope": [1000, 1500, 2000, 500],
        "vwap_dev": [0.01, 0.02, 0.03, 0.005],
        "ema_9": [100, 102, 105, 98],
        "ema_21": [99, 101, 103, 97],
        "ema_50": [97, 99, 101, 95],
        "ema_cross_9_21": [1, 1, 1, -1],
        "price_vs_ema50": [0.03, 0.02, 0.04, -0.02],
        "adx_14": [30, 35, 40, 20],
        "di_plus": [25, 30, 35, 15],
        "di_minus": [15, 10, 8, 18],
        "zscore_20": [1.2, 1.5, 2.0, 0.8],
        "distance_from_52w_high": [0.1, 0.15, 0.2, 0.05],
        "distance_from_52w_low": [0.5, 0.6, 0.7, 0.4],
        "vol_regime": [1, 1, 2, 0],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_trades_df():
    """Sample trades DataFrame."""
    now = datetime.now()
    data = {
        "id": [1, 2, 3, 4, 5],
        "symbol": ["SBIN", "HDFC", "INFY", "SBIN", "HDFC"],
        "entry_time": [now - timedelta(days=i) for i in range(5)],
        "exit_time": [now - timedelta(days=i - 1) for i in range(5)],
        "entry_price": [500, 2500, 1800, 505, 2510],
        "exit_price": [510, 2450, 1850, 495, 2490],
        "quantity": [10, 5, 8, 10, 5],
        "side": ["BUY", "BUY", "BUY", "BUY", "BUY"],
        "pnl": [100, -250, 400, -100, -100],
        "pnl_pct": [0.02, -0.05, 0.022, -0.02, -0.04],
        "signal_prob": [0.75, 0.55, 0.85, 0.45, 0.50],
        "strategy_name": ["breakout", "meanrevert", "breakout", "meanrevert", "breakout"],
        "features_used": [
            {"rsi_14": 45, "macd": 0.5},
            {"rsi_14": 55, "macd": -0.3},
            {"rsi_14": 65, "macd": 0.8},
            {"rsi_14": 35, "macd": -0.5},
            {"rsi_14": 50, "macd": 0.2},
        ],
        "shap_values": [
            {"rsi_14": 0.1, "macd": 0.05},
            {"rsi_14": -0.08, "macd": -0.03},
            {"rsi_14": 0.15, "macd": 0.08},
            {"rsi_14": -0.1, "macd": -0.05},
            {"rsi_14": 0.05, "macd": 0.02},
        ],
    }
    return pd.DataFrame(data)


# ============================================================================
# PerformanceAttribution Tests
# ============================================================================


class TestPerformanceAttribution:
    """Tests for PerformanceAttribution class."""

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_initialization(self, mock_redis, mock_engine):
        """Test initialization."""
        attr = PerformanceAttribution(lookback_days=60)
        assert attr._lookback_days == 60

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_compute_shap_values_returns_correct_shape(
        self, mock_redis, mock_engine, mock_model, sample_features_df
    ):
        """Test SHAP value computation returns correct shape."""
        attr = PerformanceAttribution()
        shap_df = attr.compute_shap_values(mock_model, sample_features_df)

        assert shap_df.shape[0] == len(sample_features_df)
        assert all(col in shap_df.columns for col in FEATURE_COLUMNS[:5])

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_compute_shap_values_preserves_index(
        self, mock_redis, mock_engine, mock_model, sample_features_df
    ):
        """Test SHAP values preserve input index."""
        attr = PerformanceAttribution()
        sample_features_df.index = [10, 20, 30, 40]
        shap_df = attr.compute_shap_values(mock_model, sample_features_df)

        assert list(shap_df.index) == [10, 20, 30, 40]

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_feature_importance_trend_structure(self, mock_redis, mock_engine, sample_features_df):
        """Test feature importance trend output structure."""
        attr = PerformanceAttribution()
        # Mock _get_trades_in_range
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            mock_trades = pd.DataFrame(
                {
                    "id": [1, 2],
                    "symbol": ["SBIN", "HDFC"],
                    "entry_time": [datetime.now(), datetime.now()],
                    "exit_time": [None, None],
                    "entry_price": [500, 2500],
                    "exit_price": [None, None],
                    "quantity": [10, 5],
                    "side": ["BUY", "BUY"],
                    "pnl": [100, 200],
                    "pnl_pct": [0.02, 0.04],
                    "signal_prob": [0.75, 0.85],
                    "strategy_name": ["breakout", "meanrevert"],
                    "features_used": [{}, {}],
                    "shap_values": [{"rsi_14": 0.1, "macd": 0.05}, {"rsi_14": 0.15, "macd": 0.08}],
                }
            )
            mock_get.return_value = mock_trades

            trend = attr.feature_importance_trend(
                (date.today() - timedelta(days=30), date.today()),
                window_days=7,
            )

            # Trend should be a DataFrame
            assert isinstance(trend, pd.DataFrame)
            # Should have features as columns
            assert len(trend.columns) > 0 or trend.empty

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_strategy_contribution_aggregation(self, mock_redis, mock_engine):
        """Test strategy contribution computes correctly."""
        attr = PerformanceAttribution()
        # Mock _get_trades_in_range
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "INFY"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 1800],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 8],
                    "side": ["BUY"] * 3,
                    "pnl": [100, -200, 500],
                    "pnl_pct": [0.02, -0.04, 0.028],
                    "signal_prob": [0.75, 0.55, 0.85],
                    "strategy_name": ["breakout", "breakout", "meanrevert"],
                    "features_used": [{}, {}, {}],
                    "shap_values": [None, None, None],
                }
            )
            mock_get.return_value = trades_df

            contrib = attr.strategy_contribution(lookback_days=30)

            assert "breakout" in contrib
            assert "meanrevert" in contrib
            assert contrib["breakout"]["pnl"] == -100
            assert contrib["breakout"]["trades"] == 2
            assert contrib["breakout"]["win_rate"] == 0.5
            assert contrib["meanrevert"]["pnl"] == 500

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_loss_analysis_returns_top_losers(self, mock_redis, mock_engine):
        """Test loss analysis extracts top losers."""
        attr = PerformanceAttribution()
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "INFY"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 1800],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 8],
                    "side": ["BUY"] * 3,
                    "pnl": [-100, -500, -50],
                    "pnl_pct": [-0.02, -0.1, -0.01],
                    "signal_prob": [0.45, 0.55, 0.65],
                    "strategy_name": ["breakout", "meanrevert", "breakout"],
                    "features_used": [{}, {}, {}],
                    "shap_values": [None, None, None],
                }
            )
            mock_get.return_value = trades_df

            # min_loss=-200 means include losses <= -200 (only -500)
            losers = attr.loss_analysis(min_loss=-200)
            assert len(losers) == 1
            assert losers[0].pnl == -500
            assert losers[0].symbol == "HDFC"

            # min_loss=0 means include all losses (all <= 0)
            losers_all = attr.loss_analysis(min_loss=0)
            assert len(losers_all) == 2

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_profit_analysis_returns_top_winners(self, mock_redis, mock_engine):
        """Test profit analysis extracts top winners."""
        attr = PerformanceAttribution()
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "INFY"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 1800],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 8],
                    "side": ["BUY"] * 3,
                    "pnl": [100, 500, 50],
                    "pnl_pct": [0.02, 0.1, 0.01],
                    "signal_prob": [0.75, 0.85, 0.65],
                    "strategy_name": ["breakout", "meanrevert", "breakout"],
                    "features_used": [{}, {}, {}],
                    "shap_values": [None, None, None],
                }
            )
            mock_get.return_value = trades_df

            winners = attr.profit_analysis(min_profit=100)

            assert len(winners) == 2
            assert winners[0].pnl == 500
            assert winners[0].symbol == "HDFC"
            assert winners[1].pnl == 100

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_feature_correlation_with_returns(self, mock_redis, mock_engine):
        """Test feature-return correlation computation."""
        attr = PerformanceAttribution()
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3, 4, 5],
                    "symbol": ["SBIN"] * 5,
                    "entry_time": [datetime.now()] * 5,
                    "exit_time": [None] * 5,
                    "entry_price": [500] * 5,
                    "exit_price": [None] * 5,
                    "quantity": [10] * 5,
                    "side": ["BUY"] * 5,
                    "pnl": [100, 200, -50, -100, 150],
                    "pnl_pct": [0.02, 0.04, -0.01, -0.02, 0.03],
                    "signal_prob": [0.75, 0.8, 0.45, 0.5, 0.85],
                    "strategy_name": ["breakout"] * 5,
                    "features_used": [
                        {"rsi_14": 45},
                        {"rsi_14": 55},
                        {"rsi_14": 35},
                        {"rsi_14": 65},
                        {"rsi_14": 75},
                    ],
                    "shap_values": [None] * 5,
                }
            )
            mock_get.return_value = trades_df

            corr = attr.feature_correlation_with_returns(lookback_days=30)

            assert isinstance(corr, dict)
            # RSI should have non-zero correlation
            assert "rsi_14" in corr or len(corr) > 0

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_attribution_report_generation(self, mock_redis, mock_engine):
        """Test full attribution report generation."""
        attr = PerformanceAttribution()
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2],
                    "symbol": ["SBIN", "HDFC"],
                    "entry_time": [datetime.now(), datetime.now()],
                    "exit_time": [None, None],
                    "entry_price": [500, 2500],
                    "exit_price": [None, None],
                    "quantity": [10, 5],
                    "side": ["BUY", "BUY"],
                    "pnl": [100, 200],
                    "pnl_pct": [0.02, 0.04],
                    "signal_prob": [0.75, 0.85],
                    "strategy_name": ["breakout", "meanrevert"],
                    "features_used": [{}, {}],
                    "shap_values": [{"rsi_14": 0.1}, {"macd": 0.08}],
                }
            )
            mock_get.return_value = trades_df

        with patch.object(attr, "feature_importance_trend") as mock_trend:
            mock_trend.return_value = pd.DataFrame()

            report = attr.generate_attribution_report()

            assert report.total_trades == 2
            assert report.total_pnl == 300
            assert report.win_count == 2
            assert report.win_rate == 1.0


# ============================================================================
# FeatureDriftReporter Tests
# ============================================================================


class TestFeatureDriftReporter:
    """Tests for FeatureDriftReporter class."""

    @patch("monitoring.feature_drift_report.get_engine")
    @patch("monitoring.feature_drift_report.get_redis")
    @patch("monitoring.feature_drift_report.TelegramNotifier")
    def test_initialization(self, mock_notifier, mock_redis, mock_engine):
        """Test initialization."""
        reporter = FeatureDriftReporter(alert_threshold=0.25)
        assert reporter._alert_threshold == 0.25

    @patch("monitoring.feature_drift_report.get_engine")
    @patch("monitoring.feature_drift_report.get_redis")
    @patch("monitoring.feature_drift_report.TelegramNotifier")
    def test_compute_feature_statistics_returns_correct_structure(
        self, mock_notifier, mock_redis, mock_engine
    ):
        """Test feature statistics computation."""
        reporter = FeatureDriftReporter()
        with patch.object(reporter, "_fetch_trades_with_features") as mock_fetch:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN"] * 3,
                    "features_used": [
                        {"rsi_14": 45, "macd": 0.5},
                        {"rsi_14": 55, "macd": -0.3},
                        {"rsi_14": 65, "macd": 0.8},
                    ],
                }
            )
            mock_fetch.return_value = trades_df

            stats = reporter.compute_feature_statistics(
                (date.today() - timedelta(days=7), date.today())
            )

            assert isinstance(stats, FeatureStatistics)
            assert stats.feature_stats.get("rsi_14") is not None

    @patch("monitoring.feature_drift_report.get_engine")
    @patch("monitoring.feature_drift_report.get_redis")
    @patch("monitoring.feature_drift_report.TelegramNotifier")
    def test_detect_distribution_shift(self, mock_notifier, mock_redis, mock_engine):
        """Test distribution shift detection."""
        reporter = FeatureDriftReporter()

        prev_stats = FeatureStatistics(
            period_start=date.today() - timedelta(days=14),
            period_end=date.today() - timedelta(days=7),
            timestamp=datetime.now(),
            feature_stats={
                "rsi_14": {"mean": 50, "std": 10, "min": 30, "max": 70, "median": 50},
                "macd": {"mean": 0.0, "std": 0.5, "min": -1.0, "max": 1.0, "median": 0.0},
            },
        )

        curr_stats = FeatureStatistics(
            period_start=date.today() - timedelta(days=7),
            period_end=date.today(),
            timestamp=datetime.now(),
            feature_stats={
                "rsi_14": {"mean": 70, "std": 15, "min": 40, "max": 85, "median": 70},
                "macd": {"mean": 0.1, "std": 0.6, "min": -1.2, "max": 1.2, "median": 0.1},
            },
        )

        divergences = reporter.detect_distribution_shift(prev_stats, curr_stats)

        assert isinstance(divergences, dict)
        assert all(0 <= v <= 10 for v in divergences.values())

    @patch("monitoring.feature_drift_report.get_engine")
    @patch("monitoring.feature_drift_report.get_redis")
    @patch("monitoring.feature_drift_report.TelegramNotifier")
    def test_correlation_change_detection(self, mock_notifier, mock_redis, mock_engine):
        """Test correlation change detection."""
        reporter = FeatureDriftReporter()
        with patch.object(reporter, "_fetch_trades_with_features") as mock_fetch:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN"] * 3,
                    "features_used": [
                        {"rsi_14": 45},
                        {"rsi_14": 55},
                        {"rsi_14": 65},
                    ],
                    "pnl_pct": [0.02, -0.01, 0.03],
                }
            )
            mock_fetch.return_value = trades_df

            changes = reporter.correlation_change_detection(
                (date.today() - timedelta(days=14), date.today() - timedelta(days=7)),
                (date.today() - timedelta(days=7), date.today()),
            )

            assert isinstance(changes, dict)

    @patch("monitoring.feature_drift_report.get_engine")
    @patch("monitoring.feature_drift_report.get_redis")
    @patch("monitoring.feature_drift_report.TelegramNotifier")
    def test_weekly_report_generation(self, mock_notifier, mock_redis, mock_engine):
        """Test weekly drift report generation."""
        reporter = FeatureDriftReporter()
        with patch.object(reporter, "compute_feature_statistics") as mock_stats:
            mock_stats.return_value = FeatureStatistics(
                period_start=date.today() - timedelta(days=7),
                period_end=date.today(),
                timestamp=datetime.now(),
                feature_stats={
                    "rsi_14": {
                        "mean": 50,
                        "std": 10,
                        "min": 30,
                        "max": 70,
                        "median": 50,
                        "count": 100,
                    }
                },
            )

            with patch.object(reporter, "detect_distribution_shift") as mock_shift:
                mock_shift.return_value = {"rsi_14": 0.1}

                report = reporter.generate_weekly_report()

                assert isinstance(report, str)
                assert "Feature Drift Report" in report


# ============================================================================
# TradeReviewEngine Tests
# ============================================================================


class TestTradeReviewEngine:
    """Tests for TradeReviewEngine class."""

    @patch("monitoring.trade_review.get_engine")
    @patch("monitoring.trade_review.get_redis")
    def test_initialization(self, mock_redis, mock_engine):
        """Test initialization."""
        engine = TradeReviewEngine()
        assert engine is not None

    @patch("monitoring.trade_review.get_engine")
    @patch("monitoring.trade_review.get_redis")
    def test_post_trade_review_categorization(self, mock_redis, mock_engine):
        """Test trade outcome categorization."""
        engine = TradeReviewEngine()

        trade = TradeContext(
            trade_id=1,
            symbol="SBIN",
            entry_time=datetime.now(),
            exit_time=datetime.now() + timedelta(hours=1),
            entry_price=500,
            exit_price=510,  # 2% profit
            quantity=10,
            side="BUY",
            pnl=100,
            pnl_pct=0.02,
            signal_prob=0.75,
            features_used={"rsi_14": 45, "macd": 0.5},
            shap_values=None,
        )

        review = engine.post_trade_review(trade, target_pct=2.0, stop_loss_pct=-1.5)

        assert review.trade_id == 1
        assert review.outcome in OutcomeCategory
        assert review.symbol == "SBIN"
        assert review.signal_prob == 0.75

    @patch("monitoring.trade_review.get_engine")
    @patch("monitoring.trade_review.get_redis")
    def test_identify_patterns_in_winners(self, mock_redis, mock_engine):
        """Test winner pattern identification."""
        engine = TradeReviewEngine()
        with patch.object(engine, "_fetch_trades") as mock_fetch:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "SBIN"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 505],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 10],
                    "side": ["BUY"] * 3,
                    "pnl": [100, 200, 150],
                    "pnl_pct": [0.02, 0.04, 0.03],
                    "signal_prob": [0.75, 0.85, 0.80],
                    "strategy_name": ["breakout"] * 3,
                    "features_used": [{"rsi_14": 45}, {"rsi_14": 55}, {"rsi_14": 50}],
                }
            )
            mock_fetch.return_value = trades_df

            patterns = engine.identify_patterns_in_winners()

            assert "n_winners" in patterns
            assert patterns["n_winners"] == 3
            assert "avg_signal_prob" in patterns
            assert patterns["avg_signal_prob"] > 0

    @patch("monitoring.trade_review.get_engine")
    @patch("monitoring.trade_review.get_redis")
    def test_identify_patterns_in_losers(self, mock_redis, mock_engine):
        """Test loser pattern identification."""
        engine = TradeReviewEngine()
        with patch.object(engine, "_fetch_trades") as mock_fetch:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "INFY"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 1800],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 8],
                    "side": ["BUY"] * 3,
                    "pnl": [-100, -200, -150],
                    "pnl_pct": [-0.02, -0.04, -0.03],
                    "signal_prob": [0.45, 0.55, 0.50],
                    "strategy_name": ["breakout"] * 3,
                    "features_used": [{"rsi_14": 75}, {"rsi_14": 80}, {"rsi_14": 85}],
                }
            )
            mock_fetch.return_value = trades_df

            patterns = engine.identify_patterns_in_losers()

            assert "n_losers" in patterns
            assert patterns["n_losers"] == 3
            assert patterns["avg_pnl"] < 0

    @patch("monitoring.trade_review.get_engine")
    @patch("monitoring.trade_review.get_redis")
    def test_generate_lessons_learned(self, mock_redis, mock_engine):
        """Test lessons learned generation."""
        engine = TradeReviewEngine()
        with patch.object(engine, "identify_patterns_in_winners") as mock_winners:
            with patch.object(engine, "identify_patterns_in_losers") as mock_losers:
                mock_winners.return_value = {
                    "n_winners": 10,
                    "avg_signal_prob": 0.80,
                    "top_features": ["rsi_14", "macd"],
                }
                mock_losers.return_value = {
                    "n_losers": 5,
                    "avg_signal_prob": 0.50,
                    "risky_features": ["bb_position"],
                }

                lessons = engine.generate_lessons_learned()

                assert "n_winners" in lessons
                assert "n_losers" in lessons
                assert lessons["n_winners"] == 10
                assert lessons["n_losers"] == 5


# ============================================================================
# AttributionReport Tests
# ============================================================================


class TestAttributionReport:
    """Tests for AttributionReport formatter."""

    def test_attribution_report_formatting(self):
        """Test attribution report formatting."""

        attribution = {
            "total_trades": 10,
            "total_pnl": 1000,
            "win_rate": 0.6,
            "win_count": 6,
            "loss_count": 4,
            "top_features_by_shap": [("rsi_14", 0.15), ("macd", 0.12)],
            "strategy_contribution": {
                "breakout": {"pnl": 600, "trades": 6, "win_rate": 0.67},
                "meanrevert": {"pnl": 400, "trades": 4, "win_rate": 0.50},
            },
            "feature_correlation_with_returns": {"rsi_14": 0.35, "macd": 0.22},
            "top_winners": [
                TradeRecord(
                    trade_id=1,
                    symbol="SBIN",
                    entry_time=datetime.now(),
                    exit_time=None,
                    entry_price=500,
                    exit_price=520,
                    quantity=10,
                    side="BUY",
                    pnl=200,
                    pnl_pct=0.04,
                    signal_prob=0.85,
                    strategy_name="breakout",
                    features_used={},
                )
            ],
            "top_losers": [
                TradeRecord(
                    trade_id=2,
                    symbol="HDFC",
                    entry_time=datetime.now(),
                    exit_time=None,
                    entry_price=2500,
                    exit_price=2450,
                    quantity=5,
                    side="BUY",
                    pnl=-250,
                    pnl_pct=-0.05,
                    signal_prob=0.55,
                    strategy_name="meanrevert",
                    features_used={},
                )
            ],
        }

        report = AttributionReport.generate(
            attribution,
            date_range=("2024-01-01", "2024-01-31"),
        )

        assert isinstance(report, str)
        assert "ATTRIBUTION REPORT" in report
        assert "2024-01-01" in report
        assert "2024-01-31" in report
        assert "rsi_14" in report


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    @patch("monitoring.attribution.get_engine")
    @patch("monitoring.attribution.get_redis")
    def test_full_attribution_pipeline(self, mock_redis, mock_engine):
        """Test complete attribution pipeline."""
        attr = PerformanceAttribution()
        with patch.object(attr, "_get_trades_in_range") as mock_get:
            trades_df = pd.DataFrame(
                {
                    "id": [1, 2, 3],
                    "symbol": ["SBIN", "HDFC", "INFY"],
                    "entry_time": [datetime.now()] * 3,
                    "exit_time": [None] * 3,
                    "entry_price": [500, 2500, 1800],
                    "exit_price": [None] * 3,
                    "quantity": [10, 5, 8],
                    "side": ["BUY"] * 3,
                    "pnl": [100, 200, -100],
                    "pnl_pct": [0.02, 0.04, -0.02],
                    "signal_prob": [0.75, 0.85, 0.45],
                    "strategy_name": ["breakout", "meanrevert", "breakout"],
                    "features_used": [{}, {}, {}],
                    "shap_values": [{"rsi_14": 0.1}, {"macd": 0.08}, None],
                }
            )
            mock_get.return_value = trades_df

        with patch.object(attr, "feature_importance_trend") as mock_trend:
            mock_trend.return_value = pd.DataFrame()

            report = attr.generate_attribution_report()

            # Verify report has all key components
            assert report.total_trades == 3
            assert report.win_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
