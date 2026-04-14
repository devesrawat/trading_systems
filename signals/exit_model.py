"""
XGBoost exit model for partial profit-taking.

Strategy:
  - Partial exit (50%) at R2 (2x ATR from entry)
  - Full exit at trailing stop breach (1x ATR below highest close since entry)
  - Model-based exit: XGBoost trained on exit outcomes (Phase 4 retrain)

Status: STUB — model-based exits not yet trained. Rule-based exits are
implemented and functional; model predict() always returns 0.0 until
a model is loaded from MLflow model name "nse_exit_signal_eq".

Usage:
    exit_model = ExitModel()
    score = exit_model.predict(feature_row)      # 0.0 = hold, 1.0 = exit
    if exit_model.should_exit_rule_based(entry_price, current_price, atr):
        executor.place_market_order(symbol, "SELL", quantity, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

_EXIT_MODEL_NAME = "nse_exit_signal_eq"
_PROFIT_TARGET_R = 2.0  # take partial profit at 2× ATR
_TRAILING_STOP_R = 1.0  # trailing stop at 1× ATR below high-water mark


@dataclass
class PositionContext:
    """Runtime context for a single open position."""

    symbol: str
    entry_price: float
    atr_at_entry: float = 0.0
    quantity: int = 0
    current_price: float = 0.0
    atr: float = 0.0  # alias for atr_at_entry for convenience
    held_days: int = 0
    high_water_mark: float = field(init=False)

    def __post_init__(self) -> None:
        self.high_water_mark = self.entry_price
        # Allow either atr or atr_at_entry; prefer the non-zero one
        if self.atr_at_entry == 0.0 and self.atr != 0.0:
            self.atr_at_entry = self.atr
        elif self.atr == 0.0 and self.atr_at_entry != 0.0:
            self.atr = self.atr_at_entry


class ExitModel:
    """
    Hybrid exit model: rule-based exits with optional model-based overlay.

    The model overlay is disabled until an exit model is trained and registered
    in MLflow. Rule-based exits (profit target + trailing stop) are always active.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        self._model = None
        self._tracking_uri = tracking_uri

    # ------------------------------------------------------------------
    # Rule-based exits
    # ------------------------------------------------------------------

    def should_exit(self, ctx: PositionContext, current_price: float | None = None) -> bool:
        """
        Convenience wrapper: return True if any rule-based exit condition is met.

        Uses `ctx.current_price` if `current_price` is not provided.
        """
        price = current_price if current_price is not None else ctx.current_price
        return self.should_partial_exit(ctx, price) or self.should_trailing_stop(ctx, price)

    def should_partial_exit(self, ctx: PositionContext, current_price: float) -> bool:
        """Return True when price has risen 2× ATR from entry (partial profit target)."""
        if ctx.atr_at_entry <= 0:
            return False
        profit_target = ctx.entry_price + _PROFIT_TARGET_R * ctx.atr_at_entry
        return current_price >= profit_target

    def update_high_water_mark(self, ctx: PositionContext, current_price: float) -> None:
        """Update trailing stop reference price."""
        if current_price > ctx.high_water_mark:
            ctx.high_water_mark = current_price

    def should_trailing_stop(self, ctx: PositionContext, current_price: float) -> bool:
        """Return True when price has fallen 1× ATR below the high-water mark."""
        if ctx.atr_at_entry <= 0:
            return False
        stop_level = ctx.high_water_mark - _TRAILING_STOP_R * ctx.atr_at_entry
        return current_price <= stop_level

    # ------------------------------------------------------------------
    # Model-based overlay (stub until retrained)
    # ------------------------------------------------------------------

    def predict(self, feature_row: dict[str, Any]) -> float:
        """
        Return P(exit=1) from the trained exit model.

        Returns 0.0 (hold) until an exit model is loaded.
        """
        if self._model is None:
            self._try_load_model()

        if self._model is None:
            return 0.0

        try:
            import numpy as np
            import pandas as pd

            df = pd.DataFrame([feature_row])[self._model.feature_names]
            probs = self._model._model.predict_proba(df.values.astype(np.float32))
            return float(probs[0, 1])
        except Exception as exc:
            log.debug("exit_model_predict_failed", error=str(exc))
            return 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        try:
            import mlflow

            if self._tracking_uri:
                mlflow.set_tracking_uri(self._tracking_uri)
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(_EXIT_MODEL_NAME, stages=["Production"])
            if versions:
                from signals.model import SignalModel

                self._model = SignalModel(versions[0].source)
                log.info("exit_model_loaded", version=versions[0].version)
        except Exception as exc:
            log.debug("exit_model_load_failed", error=str(exc))
