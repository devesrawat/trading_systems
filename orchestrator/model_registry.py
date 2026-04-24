"""
Model registry for champion/challenger tracking.

Manages the lifecycle of champion and challenger models, including:
- Setting/getting champion and challenger models
- Tracking promotion history
- Persisting state to Redis and MLflow
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import mlflow
import structlog

from data.store import get_redis

log = structlog.get_logger(__name__)

_CHAMPION_MODEL_KEY = "trading:ml:champion_model"
_CHALLENGER_MODEL_KEY = "trading:ml:challenger_model"
_PROMOTION_HISTORY_KEY = "trading:ml:promotion_history"
_MLFLOW_CHAMPION_TAG = "champion"
_MLFLOW_CHALLENGER_TAG = "challenger"
_MLFLOW_PROMOTION_TAG = "promotion_version"


class ModelRegistry:
    """
    Tracks champion and challenger models for A/B testing.

    Persists state in Redis with optional MLflow integration for metadata.
    """

    def __init__(self) -> None:
        self._redis = get_redis()

    # ------------------------------------------------------------------
    # Champion/Challenger setters
    # ------------------------------------------------------------------

    def set_champion(
        self, model_path: str, version: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Mark a model as the current champion (production).

        Parameters
        ----------
        model_path : str
            Path to the champion model file
        version : str
            Version string (e.g., "v1.0", "20250101_v1")
        metadata : dict[str, Any] | None
            Optional metadata (e.g., sharpe, win_rate, num_trades)
        """
        champion_info = {
            "model_path": model_path,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._redis.set(_CHAMPION_MODEL_KEY, json.dumps(champion_info), ex=7776000)  # 90 days

        log.info("champion_set", model_path=model_path, version=version, metadata=metadata or {})

    def set_challenger(
        self, model_path: str, version: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Mark a model as the current challenger (staging).

        Parameters
        ----------
        model_path : str
            Path to the challenger model file
        version : str
            Version string
        metadata : dict[str, Any] | None
            Optional metadata
        """
        challenger_info = {
            "model_path": model_path,
            "version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self._redis.set(_CHALLENGER_MODEL_KEY, json.dumps(challenger_info), ex=7776000)  # 90 days

        log.info("challenger_set", model_path=model_path, version=version, metadata=metadata or {})

    # ------------------------------------------------------------------
    # Champion/Challenger getters
    # ------------------------------------------------------------------

    def get_champion(self) -> dict[str, Any] | None:
        """
        Get current champion model info.

        Returns
        -------
        dict or None
            {"model_path": str, "version": str, "timestamp": str, "metadata": dict}
            or None if no champion set
        """
        data = self._redis.get(_CHAMPION_MODEL_KEY)
        if data is None:
            log.warning("champion_not_found")
            return None
        return json.loads(data)

    def get_challenger(self) -> dict[str, Any] | None:
        """
        Get current challenger model info.

        Returns
        -------
        dict or None
            {"model_path": str, "version": str, "timestamp": str, "metadata": dict}
            or None if no challenger set
        """
        data = self._redis.get(_CHALLENGER_MODEL_KEY)
        if data is None:
            log.warning("challenger_not_found")
            return None
        return json.loads(data)

    # ------------------------------------------------------------------
    # Promotion history
    # ------------------------------------------------------------------

    def record_promotion(
        self,
        old_champion_version: str,
        new_champion_version: str,
        reason: str,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """
        Record a promotion event in history.

        Parameters
        ----------
        old_champion_version : str
            Version of the model being replaced
        new_champion_version : str
            Version of the new champion
        reason : str
            Reason for promotion (e.g., "beat_on_sharpe", "statistical_win")
        metrics : dict[str, float] | None
            Comparison metrics (e.g., old_sharpe, new_sharpe, p_value)
        """
        promotion_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "old_champion": old_champion_version,
            "new_champion": new_champion_version,
            "reason": reason,
            "metrics": metrics or {},
        }

        # Append to history list
        history_json = self._redis.get(_PROMOTION_HISTORY_KEY)
        history = json.loads(history_json) if history_json else []
        history.append(promotion_event)

        # Keep last 100 promotions
        history = history[-100:]

        self._redis.set(_PROMOTION_HISTORY_KEY, json.dumps(history), ex=7776000)  # 90 days

        log.info(
            "promotion_recorded",
            old_champion=old_champion_version,
            new_champion=new_champion_version,
            reason=reason,
            metrics=metrics or {},
        )

    def list_promotion_history(self) -> list[dict[str, Any]]:
        """
        Get full promotion history (newest first).

        Returns
        -------
        list[dict]
            List of promotion events, ordered newest first
        """
        history_json = self._redis.get(_PROMOTION_HISTORY_KEY)
        history = json.loads(history_json) if history_json else []
        return list(reversed(history))  # newest first

    # ------------------------------------------------------------------
    # Rollback
    # ------------------------------------------------------------------

    def get_previous_champion(self) -> dict[str, Any] | None:
        """
        Get the champion version that was in place before the current one.

        Returns
        -------
        dict or None
            Champion model info from the previous promotion, or None if no history
        """
        history = self.list_promotion_history()
        if not history:
            log.warning("rollback_no_history")
            return None

        # The most recent promotion event tells us the old_champion
        last_promotion = history[0]
        old_version = last_promotion.get("old_champion")

        # Try to reconstruct from metadata (we'd need to store old champion metadata)
        # For now, we return the event metadata
        return {
            "version": old_version,
            "promotion_event": last_promotion,
        }

    def save_to_redis(self) -> None:
        """Ensure current state is persisted to Redis (idempotent)."""
        champion = self.get_champion()
        challenger = self.get_challenger()

        if champion:
            self._redis.set(_CHAMPION_MODEL_KEY, json.dumps(champion), ex=7776000)
            log.info("champion_persisted_to_redis", version=champion.get("version"))

        if challenger:
            self._redis.set(_CHALLENGER_MODEL_KEY, json.dumps(challenger), ex=7776000)
            log.info("challenger_persisted_to_redis", version=challenger.get("version"))

    def load_from_redis(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """
        Load both champion and challenger from Redis.

        Returns
        -------
        tuple[dict | None, dict | None]
            (champion_info, challenger_info)
        """
        champion = self.get_champion()
        challenger = self.get_challenger()

        if champion:
            log.info("champion_loaded_from_redis", version=champion.get("version"))
        if challenger:
            log.info("challenger_loaded_from_redis", version=challenger.get("version"))

        return champion, challenger

    # ------------------------------------------------------------------
    # MLflow integration (optional)
    # ------------------------------------------------------------------

    def tag_champion_in_mlflow(self, run_id: str, version: str) -> None:
        """
        Tag an MLflow run as the current champion.

        Parameters
        ----------
        run_id : str
            MLflow run ID
        version : str
            Model version string
        """
        try:
            mlflow.set_tag(_MLFLOW_CHAMPION_TAG, version)
            log.info("mlflow_champion_tagged", run_id=run_id, version=version)
        except Exception as e:
            log.warning("mlflow_tag_failed", run_id=run_id, error=str(e))

    def tag_challenger_in_mlflow(self, run_id: str, version: str) -> None:
        """Tag an MLflow run as the current challenger."""
        try:
            mlflow.set_tag(_MLFLOW_CHALLENGER_TAG, version)
            log.info("mlflow_challenger_tagged", run_id=run_id, version=version)
        except Exception as e:
            log.warning("mlflow_tag_failed", run_id=run_id, error=str(e))
