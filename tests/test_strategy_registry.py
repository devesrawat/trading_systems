"""
Tests for the strategy registry system.

Covers
------
- Loading strategies from config
- Validation (duplicates, invalid class paths, non-BaseStrategy classes)
- Grouping by asset class and interval
- Disabled strategies are excluded
- Parameter injection
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from signals.base_strategy import BaseStrategy
from signals.registry import StrategyRegistry, load_enabled_strategies
from signals.strategies.rs_breakout import RSBreakoutStrategy
from signals.strategies.tight_closes import TightClosesStrategy
from signals.strategies.vcp import VCPStrategy


class TestStrategyRegistryLoading:
    """Test loading strategies from config."""

    def test_load_strategies_from_default_config(self):
        """Strategies load from the default config file."""
        registry = StrategyRegistry()
        assert len(registry._strategies) > 0

    def test_load_enabled_strategies_convenience_function(self):
        """load_enabled_strategies() returns a dict grouped by (asset_class, interval)."""
        result = load_enabled_strategies()
        assert isinstance(result, dict)
        assert len(result) > 0
        # All keys should be tuples of (asset_class, interval)
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)  # asset_class
            assert isinstance(key[1], str)  # interval

    def test_enabled_strategies_filter(self):
        """Only enabled strategies are returned."""
        registry = StrategyRegistry()
        enabled = registry.enabled_strategies()
        for cfg in enabled.values():
            assert cfg.get("enabled") is True

    def test_get_strategy_by_name(self):
        """get_strategy() returns an instantiated strategy by name."""
        registry = StrategyRegistry()
        # Assuming 'vcp' is enabled in the config
        strategy = registry.get_strategy("vcp")
        assert strategy is not None
        assert isinstance(strategy, VCPStrategy)

    def test_get_strategy_nonexistent_returns_none(self):
        """get_strategy() returns None for non-existent strategy."""
        registry = StrategyRegistry()
        strategy = registry.get_strategy("nonexistent_strategy")
        assert strategy is None

    def test_get_strategy_disabled_returns_none(self):
        """get_strategy() returns None for disabled strategies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": False,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    }
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            registry = StrategyRegistry(config_path)
            strategy = registry.get_strategy("vcp")
            assert strategy is None


class TestStrategyRegistryValidation:
    """Test validation of strategy configurations."""

    def test_missing_strategies_section_raises_error(self):
        """Missing 'strategies' section raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {}  # No strategies section
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="No 'strategies' section"):
                StrategyRegistry(config_path)

    def test_missing_config_file_raises_error(self):
        """Missing config file raises ValueError."""
        with pytest.raises(ValueError, match="Strategy config not found"):
            StrategyRegistry("/nonexistent/path/config.yaml")

    def test_duplicate_strategy_names_raise_error(self):
        """Duplicate strategy names raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                    {
                        "name": "vcp",  # Duplicate!
                        "class_path": "signals.strategies.rs_breakout.RSBreakoutStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="Duplicate strategy name"):
                StrategyRegistry(config_path)

    def test_invalid_class_path_raises_error(self):
        """Invalid class path raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "bad_strategy",
                        "class_path": "nonexistent.module.NonexistentStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    }
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="invalid class_path"):
                StrategyRegistry(config_path)

    def test_non_base_strategy_class_raises_error(self):
        """Non-BaseStrategy class raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "bad_class",
                        "class_path": "signals.filters.EarningsFilter",  # Not a BaseStrategy
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    }
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="not a BaseStrategy"):
                StrategyRegistry(config_path)

    def test_missing_required_fields_raises_error(self):
        """Missing required fields raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp",
                        # Missing class_path
                        "enabled": True,
                    }
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="missing required fields"):
                StrategyRegistry(config_path)

    def test_strategies_not_list_raises_error(self):
        """strategies section not being a list raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {"strategies": "not_a_list"}
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            with pytest.raises(ValueError, match="strategies must be a list"):
                StrategyRegistry(config_path)


class TestStrategyGrouping:
    """Test strategy grouping by asset class and interval."""

    def test_group_by_interval_asset_class(self):
        """Strategies are grouped by (asset_class, interval)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                    {
                        "name": "rs_breakout",
                        "class_path": "signals.strategies.rs_breakout.RSBreakoutStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            registry = StrategyRegistry(config_path)
            grouped = registry.group_by_interval_asset_class()

            assert len(grouped) == 1
            key = ("equity", "day")
            assert key in grouped
            assert len(grouped[key]) == 2
            assert all(isinstance(s, BaseStrategy) for s in grouped[key])

    def test_group_by_multiple_intervals(self):
        """Strategies with different intervals are grouped separately."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp_day",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                    {
                        "name": "vcp_5min",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "5minute",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            registry = StrategyRegistry(config_path)
            grouped = registry.group_by_interval_asset_class()

            assert len(grouped) == 2
            assert ("equity", "day") in grouped
            assert ("equity", "5minute") in grouped

    def test_group_by_multiple_asset_classes(self):
        """Strategies with multiple asset classes appear in all groups."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "multi_asset",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity", "crypto"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            registry = StrategyRegistry(config_path)
            grouped = registry.group_by_interval_asset_class()

            assert ("equity", "day") in grouped
            assert ("crypto", "day") in grouped

    def test_disabled_strategies_excluded_from_grouping(self):
        """Disabled strategies do not appear in grouped results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = {
                "strategies": [
                    {
                        "name": "vcp",
                        "class_path": "signals.strategies.vcp.VCPStrategy",
                        "enabled": True,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                    {
                        "name": "rs_breakout",
                        "class_path": "signals.strategies.rs_breakout.RSBreakoutStrategy",
                        "enabled": False,
                        "asset_classes": ["equity"],
                        "interval": "day",
                        "params": {},
                        "risk_profile": "default",
                        "mode": "research",
                        "schedule": None,
                        "min_backtest_status": None,
                    },
                ]
            }
            with open(config_path, "w") as f:
                yaml.dump(config, f)

            registry = StrategyRegistry(config_path)
            grouped = registry.group_by_interval_asset_class()

            key = ("equity", "day")
            assert len(grouped[key]) == 1
            assert grouped[key][0].name == "vcp"


class TestStrategyInstantiation:
    """Test dynamic strategy instantiation."""

    def test_instantiate_vcp_strategy(self):
        """VCPStrategy can be instantiated."""
        strategy = StrategyRegistry.instantiate_strategy("signals.strategies.vcp.VCPStrategy", {})
        assert isinstance(strategy, VCPStrategy)

    def test_instantiate_rs_breakout_strategy(self):
        """RSBreakoutStrategy can be instantiated."""
        strategy = StrategyRegistry.instantiate_strategy(
            "signals.strategies.rs_breakout.RSBreakoutStrategy", {}
        )
        assert isinstance(strategy, RSBreakoutStrategy)

    def test_instantiate_tight_closes_strategy(self):
        """TightClosesStrategy can be instantiated."""
        strategy = StrategyRegistry.instantiate_strategy(
            "signals.strategies.tight_closes.TightClosesStrategy", {}
        )
        assert isinstance(strategy, TightClosesStrategy)

    def test_instantiate_with_invalid_module_raises_error(self):
        """Invalid module name raises ImportError."""
        with pytest.raises(ImportError, match="Cannot import module"):
            StrategyRegistry.instantiate_strategy("nonexistent.module.Strategy", {})

    def test_instantiate_with_invalid_class_raises_error(self):
        """Invalid class name raises AttributeError."""
        with pytest.raises(AttributeError, match="Class 'NonexistentClass' not found"):
            StrategyRegistry.instantiate_strategy("signals.strategies.vcp.NonexistentClass", {})

    def test_instantiate_with_malformed_class_path_raises_error(self):
        """Malformed class path raises ValueError."""
        with pytest.raises(ValueError, match="Invalid class path format"):
            StrategyRegistry.instantiate_strategy("no_dot_in_path", {})

    def test_instantiate_non_base_strategy_raises_error(self):
        """Non-BaseStrategy class raises TypeError."""
        with pytest.raises(TypeError, match="is not a BaseStrategy subclass"):
            StrategyRegistry.instantiate_strategy("signals.filters.EarningsFilter", {})


class TestDefaultConfig:
    """Test the default config file has valid strategies."""

    def test_default_config_has_three_strategies(self):
        """The default config contains vcp, rs_breakout, and tight_closes."""
        registry = StrategyRegistry()
        strategies = registry.enabled_strategies()
        assert "vcp" in strategies
        assert "rs_breakout" in strategies
        assert "tight_closes" in strategies

    def test_all_default_strategies_are_base_strategy_instances(self):
        """All default strategies are BaseStrategy subclasses."""
        registry = StrategyRegistry()
        for name in registry.enabled_strategies():
            strategy = registry.get_strategy(name)
            assert isinstance(strategy, BaseStrategy)

    def test_default_strategies_have_correct_asset_classes(self):
        """All default strategies have equity asset class."""
        registry = StrategyRegistry()
        for cfg in registry.enabled_strategies().values():
            assert "equity" in cfg.get("asset_classes", [])

    def test_default_strategies_have_day_interval(self):
        """All default strategies use day interval."""
        registry = StrategyRegistry()
        for cfg in registry.enabled_strategies().values():
            assert cfg.get("interval") == "day"
