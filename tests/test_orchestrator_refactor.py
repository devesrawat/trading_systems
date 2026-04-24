"""
Unit tests for orchestrator refactoring to Signal contract and registry.

Tests:
- Loading strategies from registry
- Normalizing strategy results to Signal objects
- Mode gating logic
- Signal execution with proper filtering
"""

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.main import TradingSystem, _mode_gate
from signals.contracts import Direction, Signal, SignalType
from signals.registry import StrategyRegistry
from signals.signal_router import normalize_strategy_result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_strategy_result():
    """Sample strategy result dict (as returned by BaseStrategy.scan)."""
    return {
        "symbol": "INFY",
        "strategy": "vcp",
        "current_price": 100.0,
        "pivot_buy": 102.0,
        "stop_price": 98.0,
        "target_price": 106.0,
        "confidence": 0.75,
        "score": 0.75,
        "direction": "long",
        "exchange": "NSE",
        "asset_class": "equity",
        "timeframe": "daily",
        "size_hint_pct": 0.01,
        "liquidity_score": 0.7,
        "volatility_score": 0.5,
    }


@pytest.fixture
def sample_signal(mock_strategy_result):
    """Create a sample Signal object."""
    return normalize_strategy_result(
        strategy_result=mock_strategy_result,
        symbol="INFY",
        strategy_name="vcp",
        confidence=0.75,
        mode="paper",
    )


@pytest.fixture
def trading_system_equity():
    """Create a TradingSystem instance for testing (equity market)."""
    mock_settings = MagicMock()
    mock_settings.paper_trade_mode = True
    mock_settings.market_type = "equity"
    mock_settings.initial_capital = 500_000.0
    mock_settings.signal_threshold = 0.65
    mock_settings.max_position_pct = 0.02
    mock_settings.daily_dd_limit = 0.03
    mock_settings.weekly_dd_limit = 0.07
    mock_settings.mlflow_tracking_uri = "http://localhost:5000"
    mock_settings.ab_test_pct = 0.0

    mock_broker = MagicMock()
    mock_broker.get_balance.return_value = 500_000.0
    mock_broker.refresh_auth.return_value = None

    mock_cb = MagicMock()
    mock_cb.is_halted.return_value = False

    with (
        patch("orchestrator.main.settings", mock_settings),
        patch("orchestrator.main.get_provider", return_value=MagicMock()),
        patch("orchestrator.main.get_broker_adapter", return_value=mock_broker),
        patch("orchestrator.main.CircuitBreaker", return_value=mock_cb),
        patch("orchestrator.main.PositionSizer", return_value=MagicMock()),
        patch("orchestrator.main.OrderExecutor", return_value=MagicMock()),
        patch("orchestrator.main.TradeLogger", return_value=MagicMock()),
        patch("orchestrator.main.PortfolioMonitor", return_value=MagicMock()),
        patch("orchestrator.main.ModelRegistry", return_value=MagicMock()),
        patch("signals.regime.RegimeDetector", return_value=MagicMock()),
    ):
        system = TradingSystem(market_type="equity")
        system._circuit_breaker = mock_cb
        return system


# ---------------------------------------------------------------------------
# Tests: Mode Gating
# ---------------------------------------------------------------------------


class TestModeGate:
    """Test the _mode_gate utility function."""

    def test_research_mode_always_rejected(self):
        """Research mode signals are never executed."""
        allowed, reason = _mode_gate(
            signal_mode="research",
            paper_trade_mode=True,
            circuit_breaker_halted=False,
        )
        assert allowed is False
        assert "research mode" in reason

    def test_watchlist_mode_always_rejected(self):
        """Watchlist mode signals are never executed."""
        allowed, reason = _mode_gate(
            signal_mode="watchlist",
            paper_trade_mode=True,
            circuit_breaker_halted=False,
        )
        assert allowed is False
        assert "watchlist mode" in reason

    def test_paper_mode_allowed_in_paper_trading(self):
        """Paper mode signals are allowed when paper_trade_mode=True."""
        allowed, reason = _mode_gate(
            signal_mode="paper",
            paper_trade_mode=True,
            circuit_breaker_halted=False,
        )
        assert allowed is True
        assert reason is None

    def test_paper_mode_rejected_in_live_trading(self):
        """Paper mode signals are rejected when paper_trade_mode=False."""
        allowed, reason = _mode_gate(
            signal_mode="paper",
            paper_trade_mode=False,
            circuit_breaker_halted=False,
        )
        assert allowed is False
        assert "live trading" in reason

    def test_paper_mode_rejected_when_circuit_broken(self):
        """Paper mode signals are rejected when circuit breaker is halted."""
        allowed, reason = _mode_gate(
            signal_mode="paper",
            paper_trade_mode=True,
            circuit_breaker_halted=True,
        )
        assert allowed is False
        assert "circuit breaker" in reason

    def test_live_mode_allowed_in_live_trading(self):
        """Live mode signals are allowed when paper_trade_mode=False."""
        allowed, reason = _mode_gate(
            signal_mode="live",
            paper_trade_mode=False,
            circuit_breaker_halted=False,
        )
        assert allowed is True
        assert reason is None

    def test_live_mode_rejected_in_paper_trading(self):
        """Live mode signals are rejected when paper_trade_mode=True."""
        allowed, reason = _mode_gate(
            signal_mode="live",
            paper_trade_mode=True,
            circuit_breaker_halted=False,
        )
        assert allowed is False
        assert "paper trading" in reason

    def test_live_mode_rejected_when_circuit_broken(self):
        """Live mode signals are rejected when circuit breaker is halted."""
        allowed, reason = _mode_gate(
            signal_mode="live",
            paper_trade_mode=False,
            circuit_breaker_halted=True,
        )
        assert allowed is False
        assert "circuit breaker" in reason

    def test_unknown_mode_rejected(self):
        """Unknown modes are rejected."""
        allowed, reason = _mode_gate(
            signal_mode="invalid_mode",
            paper_trade_mode=True,
            circuit_breaker_halted=False,
        )
        assert allowed is False
        assert "unknown signal mode" in reason


# ---------------------------------------------------------------------------
# Tests: Signal Normalization
# ---------------------------------------------------------------------------


class TestSignalNormalization:
    """Test normalizing strategy results to Signal objects."""

    def test_normalize_strategy_result_creates_valid_signal(self, mock_strategy_result):
        """normalize_strategy_result creates a valid Signal object."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            confidence=0.75,
            mode="paper",
        )

        assert isinstance(signal, Signal)
        assert signal.symbol == "INFY"
        assert signal.strategy_name == "vcp"
        assert signal.confidence == 0.75
        assert signal.mode == "paper"
        assert signal.signal_type == SignalType.scanner_hit
        assert signal.direction == Direction.long

    def test_normalize_extracts_entry_spec(self, mock_strategy_result):
        """normalize_strategy_result extracts entry/stop/target prices."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="research",
        )

        assert signal.entry is not None
        assert signal.entry.entry_price == 102.0
        assert signal.entry.stop_price == 98.0
        assert signal.entry.target_price == 106.0

    def test_normalize_extracts_risk_spec(self, mock_strategy_result):
        """normalize_strategy_result extracts risk specifications."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="research",
        )

        assert signal.risk is not None
        assert signal.risk.size_hint_pct == 0.01
        assert signal.risk.liquidity_score == 0.7
        assert signal.risk.volatility_score == 0.5

    def test_normalize_defaults_mode_to_research(self, mock_strategy_result):
        """normalize_strategy_result defaults mode to 'research'."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
        )

        assert signal.mode == "research"


# ---------------------------------------------------------------------------
# Tests: Signal Execution with Mode Gating
# ---------------------------------------------------------------------------


class TestSignalExecution:
    """Test Signal-based execution with mode gating."""

    def test_research_signal_not_executed(self, trading_system_equity, mock_strategy_result):
        """Research mode signals are logged but never executed."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="research",
        )

        trading_system_equity._executor.place_market_order.reset_mock()
        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Order should never be placed
        trading_system_equity._executor.place_market_order.assert_not_called()

    def test_watchlist_signal_not_executed(self, trading_system_equity, mock_strategy_result):
        """Watchlist mode signals are logged but never executed."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="watchlist",
        )

        trading_system_equity._executor.place_market_order.reset_mock()
        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Order should never be placed
        trading_system_equity._executor.place_market_order.assert_not_called()

    def test_paper_signal_executed_in_paper_mode(self, trading_system_equity, mock_strategy_result):
        """Paper mode signals execute when paper_trade_mode=True."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="paper",
        )

        # Set up mocks for execution
        trading_system_equity._executor.place_market_order = MagicMock(return_value="ORDER_001")
        trading_system_equity._sizer.shares = MagicMock(return_value=10)

        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Verify execution was attempted
        trading_system_equity._executor.place_market_order.assert_called_once()

    def test_paper_signal_blocked_in_live_mode(self, trading_system_equity, mock_strategy_result):
        """Paper mode signals are blocked when paper_trade_mode=False."""
        # Switch to live mode
        with patch("orchestrator.main.settings") as mock_settings:
            mock_settings.paper_trade_mode = False
            trading_system_equity._circuit_breaker.is_halted = MagicMock(return_value=False)

            signal = normalize_strategy_result(
                strategy_result=mock_strategy_result,
                symbol="INFY",
                strategy_name="vcp",
                mode="paper",
            )

            trading_system_equity._executor.place_market_order = MagicMock()
            trading_system_equity._execute_signal(signal, asset_class="equity")

            # Order should not be placed
            trading_system_equity._executor.place_market_order.assert_not_called()

    def test_live_signal_blocked_in_paper_mode(self, trading_system_equity, mock_strategy_result):
        """Live mode signals are blocked when paper_trade_mode=True."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="live",
        )

        trading_system_equity._executor.place_market_order = MagicMock()
        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Order should not be placed
        trading_system_equity._executor.place_market_order.assert_not_called()

    def test_signal_blocked_when_circuit_breaker_halted(
        self, trading_system_equity, mock_strategy_result
    ):
        """Signals are blocked when circuit breaker is halted."""
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            mode="paper",
        )

        # Halt circuit breaker
        trading_system_equity._circuit_breaker.is_halted = MagicMock(return_value=True)
        trading_system_equity._executor.place_market_order = MagicMock()

        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Order should not be placed
        trading_system_equity._executor.place_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Strategy Registry Integration
# ---------------------------------------------------------------------------


class TestStrategyRegistry:
    """Test loading strategies from registry."""

    def test_registry_loads_enabled_strategies(self):
        """StrategyRegistry loads enabled strategies from config."""
        registry = StrategyRegistry()
        enabled = registry.enabled_strategies()

        # Should have at least one enabled strategy
        assert len(enabled) > 0

        # Each strategy should have required fields
        for _name, cfg in enabled.items():
            assert "class_path" in cfg
            assert "enabled" in cfg
            assert cfg["enabled"] is True

    def test_registry_can_instantiate_strategies(self):
        """StrategyRegistry can instantiate strategy classes."""
        registry = StrategyRegistry()
        enabled = registry.enabled_strategies()

        for name in enabled:
            strategy = registry.get_strategy(name)
            assert strategy is not None
            assert hasattr(strategy, "scan")
            assert hasattr(strategy, "prepare")

    def test_registry_groups_by_interval_asset_class(self):
        """StrategyRegistry groups strategies by interval and asset class."""
        registry = StrategyRegistry()
        groups = registry.group_by_interval_asset_class()

        # Should have at least one group
        assert len(groups) > 0

        # Each group should have a valid key structure
        for (asset_class, interval), strategies in groups.items():
            assert asset_class in ["equity", "crypto", "futures"]
            assert isinstance(interval, str)
            assert isinstance(strategies, list)
            assert len(strategies) > 0


# ---------------------------------------------------------------------------
# Tests: Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining registry, normalization, and execution."""

    def test_strategy_result_to_signal_to_execution_flow(
        self, trading_system_equity, mock_strategy_result
    ):
        """Full flow: strategy result → Signal → execution."""
        # 1. Normalize strategy result to Signal
        signal = normalize_strategy_result(
            strategy_result=mock_strategy_result,
            symbol="INFY",
            strategy_name="vcp",
            confidence=0.8,
            mode="paper",
        )

        # 2. Verify Signal has all required fields
        # Note: confidence from dict (0.75) takes precedence over parameter
        assert signal.symbol == "INFY"
        assert signal.strategy_name == "vcp"
        assert signal.confidence == 0.75  # From mock_strategy_result["confidence"]
        assert signal.entry is not None
        assert signal.risk is not None

        # 3. Execute Signal (should be allowed in paper mode)
        trading_system_equity._executor.place_market_order = MagicMock(return_value="ORDER_001")
        trading_system_equity._sizer.shares = MagicMock(return_value=10)

        trading_system_equity._execute_signal(signal, asset_class="equity")

        # 4. Verify order was placed
        trading_system_equity._executor.place_market_order.assert_called_once()

    def test_low_confidence_signal_not_executed(self, trading_system_equity, mock_strategy_result):
        """Signals with confidence below threshold are not executed."""
        signal = normalize_strategy_result(
            strategy_result={**mock_strategy_result, "confidence": 0.5},
            symbol="INFY",
            strategy_name="vcp",
            confidence=0.5,  # Below default 0.65 threshold
            mode="paper",
        )

        trading_system_equity._executor.place_market_order = MagicMock()
        trading_system_equity._execute_signal(signal, asset_class="equity")

        # Order should not be placed (below threshold)
        trading_system_equity._executor.place_market_order.assert_not_called()
