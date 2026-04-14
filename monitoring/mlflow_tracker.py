"""
MLflow experiment tracking and model drift detection.

Experiments:
  nse_equity_signals    — live trade signals + outcomes
  nse_options_signals   — F&O signals
  backtest_results      — walk-forward backtest runs

ModelDriftMonitor compares the rolling live win rate against
the backtest baseline. Drift score > 0.3 triggers a Telegram alert
and flags the model for retraining.
"""

from __future__ import annotations

import mlflow
import structlog

log = structlog.get_logger(__name__)

_BASELINE_WIN_RATE = 0.58  # expected from backtest


class ModelDriftMonitor:
    """
    Detects degradation in live model performance vs backtest baseline.

    Drift score = abs(baseline_win_rate - live_win_rate)
    Score > 0.3 → alert + retrain flag
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.MlflowClient()

    def compare_live_vs_backtest(self, window_trades: int = 20) -> float:
        """
        Compute drift score from the last *window_trades* completed runs
        in the nse_equity_signals experiment.

        Returns 0.0 if no completed runs exist.
        """
        try:
            runs = self._client.search_runs(
                experiment_names=["nse_equity_signals"],
                filter_string="tags.outcome != 'pending'",
                order_by=["start_time DESC"],
                max_results=window_trades,
            )
        except Exception as exc:
            log.error("mlflow_search_failed", error=str(exc))
            return 0.0

        if not runs:
            return 0.0

        outcomes = [r.data.tags.get("outcome", "pending") for r in runs]
        completed = [o for o in outcomes if o in ("win", "loss")]
        if not completed:
            return 0.0

        live_win_rate = sum(1 for o in completed if o == "win") / len(completed)
        drift_score = abs(_BASELINE_WIN_RATE - live_win_rate)

        log.info(
            "drift_computed",
            live_win_rate=round(live_win_rate, 3),
            baseline=_BASELINE_WIN_RATE,
            drift=round(drift_score, 3),
            n_trades=len(completed),
        )
        return float(drift_score)

    def register_model(
        self,
        run_id: str,
        segment: str = "EQ",
        stage: str = "Staging",
    ) -> str:
        """Register a trained model artifact and set its stage."""
        from signals.model import _MODEL_NAME_PREFIX

        model_name = f"{_MODEL_NAME_PREFIX}_{segment.lower()}"
        model_uri = f"runs:/{run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        self._client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage,
        )
        log.info("model_registered", name=model_name, version=mv.version, stage=stage)
        return str(mv.version)

    def promote_to_production(self, segment: str, version: str) -> None:
        """Promote a Staging model version to Production."""
        from signals.model import _MODEL_NAME_PREFIX

        model_name = f"{_MODEL_NAME_PREFIX}_{segment.lower()}"
        self._client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
        )
        log.info("model_promoted", name=model_name, version=version)
