"""
XGBoost signal model — inference wrapper and MLflow registry integration.

SignalModel   — load a trained model, run predictions, explain with SHAP
ModelRegistry — fetch/register models via MLflow model registry
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import shap
import structlog
import xgboost as xgb

from signals.features import FEATURE_COLUMNS

log = structlog.get_logger(__name__)

_MODEL_NAME_PREFIX = "nse_signal"


class SignalModel:
    """
    Wraps a trained XGBoost classifier for inference.

    Validates that the feature schema at load time matches FEATURE_COLUMNS.
    """

    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        self.feature_names: list[str] = list(FEATURE_COLUMNS)
        self._explainer: shap.TreeExplainer | None = None
        log.info("signal_model_loaded", path=model_path)

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """
        Return P(label=1) — probability of +2% in 5 days — for each row.

        Parameters
        ----------
        features : pd.DataFrame
            Must contain exactly FEATURE_COLUMNS. Index is preserved.

        Returns
        -------
        pd.Series of float in [0, 1] with same index as *features*.
        """
        self._validate_features(features)
        X = features[self.feature_names].values.astype(np.float32)
        probs = self._model.predict_proba(X)[:, 1]
        return pd.Series(probs, index=features.index, name="signal_prob")

    def predict_single(self, feature_row: dict[str, Any]) -> float:
        """Return signal probability for a single feature dict."""
        df = pd.DataFrame([feature_row])[self.feature_names]
        return float(self.predict(df).iloc[0])

    # ------------------------------------------------------------------
    # explain (SHAP)
    # ------------------------------------------------------------------

    def explain(self, feature_row: dict[str, Any]) -> dict[str, float]:
        """
        Return SHAP values for the top 10 most influential features.

        Uses TreeExplainer — fast and exact for XGBoost.
        """
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)

        df = pd.DataFrame([feature_row])[self.feature_names]
        shap_values = self._explainer.shap_values(df)

        # shap_values shape: (1, n_features) — take absolute value for ranking
        importances = np.abs(shap_values[0])
        top_indices = np.argsort(importances)[::-1][:10]

        return {self.feature_names[i]: float(shap_values[0][i]) for i in top_indices}

    # ------------------------------------------------------------------
    # health check
    # ------------------------------------------------------------------

    def is_healthy(self) -> bool:
        """Return True if model is loaded and feature schema is intact."""
        return self._model is not None and len(self.feature_names) > 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _validate_features(self, features: pd.DataFrame) -> None:
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(
                f"Missing feature columns for inference: {missing}. Expected: {self.feature_names}"
            )


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Thin wrapper around MLflow model registry.

    Supports registering new models and fetching the current Production model.
    """

    def __init__(self, tracking_uri: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        self._client = mlflow.MlflowClient()

    def get_latest_model(self, segment: str = "EQ", stage: str = "Production") -> SignalModel:
        """
        Fetch the *stage*-stage model for *segment* from MLflow.

        Parameters
        ----------
        segment : Registry segment name, e.g. "EQ" or "CRYPTO".
        stage   : MLflow model stage — "Production" (default) or "Staging".

        Raises RuntimeError if no model is found in the requested stage.
        """
        model_name = f"{_MODEL_NAME_PREFIX}_{segment.lower()}"
        versions = self._client.get_latest_versions(model_name, stages=[stage])

        if not versions:
            raise RuntimeError(
                f"No {stage} model found for '{model_name}'. Train and register a model first."
            )

        latest = versions[0]
        model_path = latest.source
        log.info("registry_model_fetched", model=model_name, version=latest.version, stage=stage)
        return SignalModel(model_path=model_path)

    def register_model(
        self,
        run_id: str,
        segment: str,
        model_path: str,
    ) -> str:
        """
        Register a trained model artifact to MLflow model registry.

        Returns the registered model version string.
        """
        model_name = f"{_MODEL_NAME_PREFIX}_{segment.lower()}"
        model_uri = f"runs:/{run_id}/model" if not model_path.startswith("runs:") else model_path

        mv = mlflow.register_model(model_uri=model_uri, name=model_name)
        log.info("model_registered", name=model_name, version=mv.version, run_id=run_id)
        return str(mv.version)
