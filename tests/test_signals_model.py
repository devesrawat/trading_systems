"""Unit tests for signals/model.py — TDD RED phase. No MLflow server required."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from signals.features import FEATURE_COLUMNS
from signals.model import ModelRegistry, SignalModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_features(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(0, 1, (n, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


def _train_tiny_model(features: pd.DataFrame) -> xgb.XGBClassifier:
    """Train a minimal XGBoost model on random labels."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, len(features))
    model = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(features, labels)
    return model


def _saved_model_path(tmp_path: Path) -> str:
    features = _make_features(100)
    model = _train_tiny_model(features)
    path = str(tmp_path / "model.ubj")
    model.save_model(path)
    return path


# ---------------------------------------------------------------------------
# SignalModel — loading
# ---------------------------------------------------------------------------

class TestSignalModelLoad:
    def test_loads_from_local_path(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        assert sm.is_healthy()

    def test_feature_names_match_expected(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        assert sm.feature_names == FEATURE_COLUMNS

    def test_invalid_path_raises(self):
        with pytest.raises((FileNotFoundError, Exception)):
            SignalModel(model_path="/nonexistent/model.ubj")

    def test_is_healthy_returns_false_before_load(self):
        sm = SignalModel.__new__(SignalModel)
        sm._model = None
        sm.feature_names = []
        assert not sm.is_healthy()


# ---------------------------------------------------------------------------
# SignalModel — predict
# ---------------------------------------------------------------------------

class TestSignalModelPredict:
    def test_predict_returns_series(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        features = _make_features(10)
        result = sm.predict(features)
        assert isinstance(result, pd.Series)

    def test_predict_length_matches_input(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        features = _make_features(20)
        result = sm.predict(features)
        assert len(result) == 20

    def test_predict_probabilities_bounded_0_to_1(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        features = _make_features(30)
        result = sm.predict(features)
        assert result.between(0.0, 1.0).all()

    def test_predict_index_preserved(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        features = _make_features(10)
        result = sm.predict(features)
        pd.testing.assert_index_equal(result.index, features.index)

    def test_predict_raises_on_wrong_columns(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        bad_df = pd.DataFrame({"wrong_col": [1.0, 2.0]})
        with pytest.raises(ValueError, match="feature"):
            sm.predict(bad_df)

    def test_predict_single_returns_float(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        result = sm.predict_single(row)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# SignalModel — SHAP explain
# ---------------------------------------------------------------------------

class TestSignalModelExplain:
    def test_explain_returns_dict(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        result = sm.explain(row)
        assert isinstance(result, dict)

    def test_explain_returns_top_10_features(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        result = sm.explain(row)
        assert len(result) <= 10

    def test_explain_keys_are_feature_names(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        result = sm.explain(row)
        for key in result:
            assert key in FEATURE_COLUMNS

    def test_explain_values_are_floats(self, tmp_path):
        path = _saved_model_path(tmp_path)
        sm = SignalModel(model_path=path)
        row = {col: 0.5 for col in FEATURE_COLUMNS}
        result = sm.explain(row)
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_register_model_calls_mlflow(self, tmp_path):
        path = _saved_model_path(tmp_path)
        with patch("signals.model.mlflow") as mock_mlflow:
            mock_mlflow.register_model.return_value = MagicMock(version="1")
            registry = ModelRegistry(tracking_uri="http://localhost:5000")
            registry.register_model(run_id="abc123", segment="EQ", model_path=path)
            mock_mlflow.register_model.assert_called_once()

    def test_get_latest_model_returns_signal_model(self, tmp_path):
        path = _saved_model_path(tmp_path)
        with patch("signals.model.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client
            mock_version = MagicMock()
            mock_version.source = path
            mock_client.get_latest_versions.return_value = [mock_version]

            registry = ModelRegistry(tracking_uri="http://localhost:5000")
            model = registry.get_latest_model(segment="EQ")
            assert isinstance(model, SignalModel)

    def test_get_latest_model_raises_if_no_production_model(self):
        with patch("signals.model.mlflow") as mock_mlflow:
            mock_client = MagicMock()
            mock_mlflow.MlflowClient.return_value = mock_client
            mock_client.get_latest_versions.return_value = []

            registry = ModelRegistry(tracking_uri="http://localhost:5000")
            with pytest.raises(RuntimeError, match="No Production model"):
                registry.get_latest_model(segment="EQ")
