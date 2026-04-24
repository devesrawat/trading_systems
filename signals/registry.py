"""
Strategy registry — load, validate, and instantiate scan strategies from config.

Design
------
1. Load strategies from config/strategy_params.yaml
2. Validate: duplicate names, invalid class paths, non-BaseStrategy classes
3. Group by (asset_class, interval)
4. Fail fast with clear error messages
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog
import yaml

from signals.base_strategy import BaseStrategy

log = structlog.get_logger(__name__)


class StrategyRegistry:
    """Load, validate, and manage enabled scan strategies."""

    def __init__(self, config_path: Path | str = "config/strategy_params.yaml"):
        """
        Initialize the registry by loading the config file.

        Parameters
        ----------
        config_path : Path or str
            Path to strategy_params.yaml

        Raises
        ------
        ValueError
            If config is invalid (missing strategies, invalid structure, etc.)
        """
        self.config_path = Path(config_path)
        self._strategies: dict[str, dict[str, Any]] = {}
        self._instances: dict[str, BaseStrategy] = {}

        if not self.config_path.exists():
            msg = f"Strategy config not found: {self.config_path}"
            log.error("config_missing", path=str(self.config_path))
            raise ValueError(msg)

        with open(self.config_path) as f:
            config = yaml.safe_load(f) or {}

        if "strategies" not in config:
            msg = "No 'strategies' section in config"
            log.error("strategies_missing")
            raise ValueError(msg)

        self._validate_strategy_config(config["strategies"])
        self._strategies = {s["name"]: s for s in config["strategies"]}

        log.info(
            "registry_loaded",
            total_strategies=len(self._strategies),
            enabled_strategies=len(self.enabled_strategies()),
        )

    def enabled_strategies(self) -> dict[str, dict[str, Any]]:
        """Return only enabled strategies."""
        return {name: cfg for name, cfg in self._strategies.items() if cfg.get("enabled", False)}

    def group_by_interval_asset_class(self) -> dict[tuple[str, str], list[BaseStrategy]]:
        """
        Group enabled strategies by (asset_class, interval).

        Returns
        -------
        dict with keys (asset_class, interval) and values as lists of strategy instances
        """
        groups: dict[tuple[str, str], list[BaseStrategy]] = defaultdict(list)

        for name, cfg in self.enabled_strategies().items():
            strategy = self.instantiate_strategy(cfg["class_path"], cfg.get("params", {}))
            self._instances[name] = strategy

            for asset_class in cfg.get("asset_classes", []):
                interval = cfg.get("interval", "day")
                key = (asset_class, interval)
                groups[key].append(strategy)

        return dict(groups)

    def get_strategy(self, name: str) -> BaseStrategy | None:
        """
        Get a strategy instance by name.

        Returns None if not found or not enabled.
        """
        if name not in self.enabled_strategies():
            return None
        if name not in self._instances:
            cfg = self._strategies[name]
            self._instances[name] = self.instantiate_strategy(
                cfg["class_path"], cfg.get("params", {})
            )
        return self._instances.get(name)

    @staticmethod
    def _validate_strategy_config(strategies: list[dict[str, Any]]) -> None:
        """
        Validate strategy configurations.

        Checks
        ------
        - All strategies have required fields: name, class_path, enabled
        - No duplicate strategy names
        - All class_paths are valid and importable
        - All classes are subclasses of BaseStrategy

        Raises
        ------
        ValueError
            If any validation check fails
        """
        if not isinstance(strategies, list):
            msg = f"strategies must be a list, got {type(strategies)}"
            log.error("invalid_strategies_type", type=str(type(strategies)))
            raise ValueError(msg)

        required_fields = {"name", "class_path", "enabled"}
        seen_names = set()

        for i, strategy in enumerate(strategies):
            # Check required fields
            missing = required_fields - set(strategy.keys())
            if missing:
                msg = f"Strategy {i}: missing required fields: {missing}"
                log.error("strategy_missing_fields", index=i, missing=list(missing))
                raise ValueError(msg)

            name = strategy.get("name")

            # Check for duplicates
            if name in seen_names:
                msg = f"Duplicate strategy name: '{name}'"
                log.error("duplicate_strategy_name", name=name)
                raise ValueError(msg)
            seen_names.add(name)

            # Validate class path
            class_path = strategy.get("class_path", "")
            if not class_path:
                msg = f"Strategy {i}: class_path is empty or missing"
                log.error("empty_class_path", index=i)
                raise ValueError(msg)
            try:
                cls = StrategyRegistry.instantiate_strategy(class_path, {})
            except (ImportError, AttributeError, TypeError) as e:
                msg = f"Strategy '{name}': invalid class_path '{class_path}': {e}"
                log.error("invalid_class_path", name=name, class_path=class_path, error=str(e))
                raise ValueError(msg) from e

            # Check it's a BaseStrategy
            if not isinstance(cls, BaseStrategy):
                msg = f"Strategy '{name}': {class_path} is not a BaseStrategy subclass"
                log.error("not_base_strategy", name=name, class_path=class_path)
                raise ValueError(msg)

    @staticmethod
    def instantiate_strategy(class_path: str, params: dict[str, Any]) -> BaseStrategy:
        """
        Dynamically load and instantiate a strategy class.

        Parameters
        ----------
        class_path : str
            Fully-qualified class name (e.g. 'signals.strategies.vcp.VCPStrategy')
        params : dict
            Parameters to pass to the strategy constructor

        Returns
        -------
        BaseStrategy instance

        Raises
        ------
        ImportError
            If module cannot be imported
        AttributeError
            If class not found in module
        TypeError
            If instantiation fails
        """
        try:
            module_name, class_name = class_path.rsplit(".", 1)
        except ValueError as e:
            msg = f"Invalid class path format: {class_path}"
            log.error("invalid_class_path_format", class_path=class_path)
            raise ValueError(msg) from e

        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            msg = f"Cannot import module '{module_name}': {e}"
            log.error("import_error", module=module_name, error=str(e))
            raise ImportError(msg) from e

        try:
            cls = getattr(module, class_name)
        except AttributeError as e:
            msg = f"Class '{class_name}' not found in module '{module_name}'"
            log.error("class_not_found", module=module_name, class_name=class_name)
            raise AttributeError(msg) from e

        try:
            # Instantiate with no-arg constructor
            # Strategy params are applied at scan time if needed
            instance = cls()
        except TypeError as e:
            msg = f"Cannot instantiate {class_path}: {e}"
            log.error("instantiation_error", class_path=class_path, error=str(e))
            raise TypeError(msg) from e

        if not isinstance(instance, BaseStrategy):
            msg = f"{class_path} is not a BaseStrategy subclass"
            log.error("not_base_strategy", class_path=class_path)
            raise TypeError(msg)

        return instance


def load_enabled_strategies() -> dict[tuple[str, str], list[BaseStrategy]]:
    """
    Convenience function to load and group enabled strategies from default config.

    Returns
    -------
    dict with keys (asset_class, interval) and values as lists of strategy instances
    """
    registry = StrategyRegistry()
    return registry.group_by_interval_asset_class()
