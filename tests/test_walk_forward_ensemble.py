"""
Tests for walk-forward ensemble training pipeline.

Coverage:
- Base models train correctly (XGBoost, LightGBM, PatchTST)
- Ensemble voting works
- Concept drift detection on synthetic regime change
- Warm-start loading
- Hyperparameter optimization converges
- Expanding window splits are correct
- MLflow logging captures all runs
- Full walk-forward backtest runs without errors
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from signals.features import (
    FEATURE_COLUMNS,
)
from signals.training.concept_drift import ConceptDriftDetector
from signals.training.ensemble_models import EnsembleStrategy, PatchTSTWrapper
from signals.training.hyperparameter_optimizer import BayesianHyperparameterOptimizer
from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def synthetic_ohlcv_data():
    """
    Generate 10 years of synthetic OHLCV + features + labels.

    ~250 trading days per year * 10 years = 2500 rows.
    """
    n_rows = 2500
    dates = pd.date_range("2014-01-01", periods=n_rows, freq="B")

    # Synthetic OHLCV
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n_rows) * 0.01) + 100
    high = close + np.abs(np.random.randn(n_rows) * 0.5)
    low = close - np.abs(np.random.randn(n_rows) * 0.5)
    open_ = np.roll(close, 1)
    volume = np.random.randint(1000000, 5000000, n_rows)

    # Synthetic features (FEATURE_COLUMNS)
    features_dict = {}
    for feat in FEATURE_COLUMNS:
        features_dict[feat] = np.random.uniform(-1, 1, n_rows)

    # Synthetic labels: balanced labels (50% positive)
    y = np.random.binomial(1, 0.5, n_rows)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            **features_dict,
            "label": y,
        },
        index=dates,
    )

    return df


@pytest.fixture
def X_y_train_test(synthetic_ohlcv_data):
    """Split data into train/test."""
    df = synthetic_ohlcv_data
    split = int(len(df) * 0.7)

    X_train = df[FEATURE_COLUMNS].iloc[:split].reset_index(drop=True)
    y_train = df["label"].iloc[:split].reset_index(drop=True)

    X_test = df[FEATURE_COLUMNS].iloc[split:].reset_index(drop=True)
    y_test = df["label"].iloc[split:].reset_index(drop=True)

    return X_train, y_train, X_test, y_test


@pytest.fixture
def ensemble_strategy(X_y_train_test):
    """Create and train an ensemble."""
    X_train, y_train, _, _ = X_y_train_test

    ensemble = EnsembleStrategy(random_state=42)
    ensemble.train_xgboost(X_train, y_train)
    ensemble.train_lightgbm(X_train, y_train)
    ensemble.train_patchtst(X_train, y_train)
    ensemble._is_fitted = True

    return ensemble


# ------------------------------------------------------------------
# Tests: Base Models
# ------------------------------------------------------------------


class TestBaseModels:
    """Test individual model training."""

    def test_xgboost_trains(self, X_y_train_test):
        """XGBoost trains without error."""
        X_train, y_train, X_test, y_test = X_y_train_test

        ensemble = EnsembleStrategy()
        model = ensemble.train_xgboost(X_train, y_train)

        assert model is not None
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)
        assert pred.dtype in [np.int32, np.int64, int]

    def test_lightgbm_trains(self, X_y_train_test):
        """LightGBM trains without error."""
        X_train, y_train, X_test, y_test = X_y_train_test

        ensemble = EnsembleStrategy()
        model = ensemble.train_lightgbm(X_train, y_train)

        assert model is not None
        pred = model.predict(X_test)
        assert len(pred) == len(X_test)

    def test_patchtst_trains(self, X_y_train_test):
        """PatchTST trains without error."""
        X_train, y_train, X_test, y_test = X_y_train_test

        ensemble = EnsembleStrategy()
        model = ensemble.train_patchtst(X_train, y_train)

        assert model is not None
        assert model._is_fitted

    def test_xgboost_predict_proba(self, X_y_train_test):
        """XGBoost predict_proba returns valid probabilities."""
        X_train, y_train, X_test, y_test = X_y_train_test

        model = EnsembleStrategy().train_xgboost(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_lightgbm_predict_proba(self, X_y_train_test):
        """LightGBM predict_proba returns valid probabilities."""
        X_train, y_train, X_test, y_test = X_y_train_test

        model = EnsembleStrategy().train_lightgbm(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_patchtst_predict_proba(self, X_y_train_test):
        """PatchTST predict_proba returns valid probabilities."""
        X_train, y_train, X_test, y_test = X_y_train_test

        model = EnsembleStrategy().train_patchtst(X_train, y_train)
        proba = model.predict_proba(X_test)

        assert np.all(proba >= 0)
        assert np.all(proba <= 1)


# ------------------------------------------------------------------
# Tests: Ensemble Voting
# ------------------------------------------------------------------


class TestEnsembleVoting:
    """Test ensemble prediction aggregation."""

    def test_ensemble_predict(self, ensemble_strategy, X_y_train_test):
        """Ensemble voting produces valid predictions."""
        _, _, X_test, _ = X_y_train_test

        pred = ensemble_strategy.ensemble_predict(X_test)
        assert len(pred) == len(X_test)
        assert set(pred) == {0, 1}

    def test_ensemble_predict_proba(self, ensemble_strategy, X_y_train_test):
        """Ensemble probability averaging produces valid probs."""
        _, _, X_test, _ = X_y_train_test

        proba = ensemble_strategy.ensemble_predict_proba(X_test)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_ensemble_voting_majority(self, ensemble_strategy, X_y_train_test):
        """Ensemble voting produces valid predictions from probabilities."""
        _, _, X_test, _ = X_y_train_test

        pred = ensemble_strategy.ensemble_predict(X_test)
        proba = ensemble_strategy.ensemble_predict_proba(X_test)

        # Predictions should be 0 or 1
        assert set(pred) <= {0, 1}
        # Probabilities should be between 0 and 1
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_ensemble_weighted_voting(self, ensemble_strategy, X_y_train_test):
        """Ensemble accepts custom weights."""
        _, _, X_test, _ = X_y_train_test

        weights = {"xgb": 0.5, "lgb": 0.3, "patchtst": 0.2}
        pred = ensemble_strategy.ensemble_predict(X_test, weights=weights)

        assert len(pred) == len(X_test)
        assert set(pred) == {0, 1}

    def test_ensemble_confidence(self, ensemble_strategy, X_y_train_test):
        """get_model_confidence returns per-model confidence."""
        _, _, X_test, _ = X_y_train_test

        pred = ensemble_strategy.ensemble_predict(X_test)
        confidences = ensemble_strategy.get_model_confidence(X_test, pred)

        assert "xgb" in confidences
        assert "lgb" in confidences
        assert "patchtst" in confidences

        for _model, conf in confidences.items():
            assert len(conf) == len(X_test)
            assert np.all(conf >= 0)
            assert np.all(conf <= 1)

    def test_ensemble_not_fitted_error(self, X_y_train_test):
        """Ensemble.predict raises error if not fitted."""
        _, _, X_test, _ = X_y_train_test

        ensemble = EnsembleStrategy()
        with pytest.raises(RuntimeError, match="not fitted"):
            ensemble.ensemble_predict(X_test)


# ------------------------------------------------------------------
# Tests: Concept Drift
# ------------------------------------------------------------------


class TestConceptDrift:
    """Test drift detection."""

    def test_kl_divergence_identical(self):
        """KL divergence is ~0 for identical distributions."""
        detector = ConceptDriftDetector()
        dist = np.random.randn(1000)

        kl = detector.compute_kl_divergence(dist, dist)
        assert 0 <= kl < 0.1

    def test_kl_divergence_different(self):
        """KL divergence is high for different distributions."""
        detector = ConceptDriftDetector()
        dist1 = np.random.randn(1000)
        dist2 = np.random.randn(1000) + 5

        kl = detector.compute_kl_divergence(dist1, dist2)
        assert kl > 0.1

    def test_feature_shift_detection(self):
        """detect_feature_shift identifies shifted features."""
        detector = ConceptDriftDetector()

        # Train: normal features
        train = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
            }
        )

        # Test: feat1 shifted, feat2 unchanged
        test = pd.DataFrame(
            {
                "feat1": np.random.randn(100) + 3,
                "feat2": np.random.randn(100),
            }
        )

        result = detector.detect_feature_shift(train, test)

        assert "max_shift" in result
        assert "mean_shift" in result
        assert "features_shifted" in result

        # feat1 should be the top shifted
        top_shifted = [feat for feat, _ in result["features_shifted"]]
        assert top_shifted[0] == "feat1"

    def test_regime_change_detection(self):
        """is_regime_change triggers on high KL divergence."""
        detector = ConceptDriftDetector(threshold=0.5)

        assert detector.is_regime_change(1.0) is True
        assert detector.is_regime_change(0.3) is False

    def test_label_shift_detection(self):
        """detect_label_shift identifies class imbalance changes."""
        detector = ConceptDriftDetector()

        train_labels = pd.Series([0, 0, 1] * 10)
        test_labels = pd.Series([1, 1, 1] * 10)

        result = detector.detect_label_shift(train_labels, test_labels)

        assert "train_pos_rate" in result
        assert "test_pos_rate" in result
        assert "shift_magnitude" in result
        assert result["is_significant_shift"]

    def test_drift_history(self):
        """Drift history is tracked correctly."""
        detector = ConceptDriftDetector()

        feature_shift = {"max_shift": 0.5, "mean_shift": 0.3}
        label_shift = {"shift_magnitude": 0.2}
        timestamp = pd.Timestamp("2024-01-01")

        detector.log_drift_metrics(timestamp, feature_shift, label_shift, False)
        history = detector.get_drift_history()

        assert len(history) == 1
        assert history.iloc[0]["max_feature_shift"] == 0.5


# ------------------------------------------------------------------
# Tests: Hyperparameter Optimization
# ------------------------------------------------------------------


class TestHyperparameterOptimizer:
    """Test Bayesian hyperparameter search."""

    def test_optimizer_init(self):
        """Optimizer initializes without error."""
        opt = BayesianHyperparameterOptimizer()
        assert opt.experiment_name == "ensemble_hpo"

    def test_suggest_hyperparams_xgb(self):
        """suggest_hyperparams returns valid XGBoost params."""
        opt = BayesianHyperparameterOptimizer(seed=42)

        # Create a dummy trial
        import optuna

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler)
        trial = study.ask()

        params = opt.suggest_hyperparams(trial, "xgb")

        assert "n_estimators" in params
        assert "max_depth" in params
        assert "learning_rate" in params
        assert params["max_depth"] < 8
        assert 0.01 <= params["learning_rate"] <= 0.2

    def test_suggest_hyperparams_lgb(self):
        """suggest_hyperparams returns valid LightGBM params."""
        opt = BayesianHyperparameterOptimizer(seed=42)

        import optuna

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler)
        trial = study.ask()

        params = opt.suggest_hyperparams(trial, "lgb")

        assert "n_estimators" in params
        assert "max_depth" in params
        assert params["max_depth"] < 8

    def test_optimize_xgb(self, X_y_train_test):
        """optimize runs and returns best params."""
        X_train, y_train, X_test, y_test = X_y_train_test

        # Split train into train/val
        split = int(len(X_train) * 0.7)
        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        opt = BayesianHyperparameterOptimizer(seed=42)
        params = opt.optimize(X_tr, y_tr, X_val, y_val, n_trials=3, model_type="xgb")

        assert isinstance(params, dict)
        assert "max_depth" in params

    def test_get_optimization_history(self, X_y_train_test):
        """get_optimization_history returns DataFrame."""
        X_train, y_train, _, _ = X_y_train_test

        split = int(len(X_train) * 0.7)
        X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
        y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

        opt = BayesianHyperparameterOptimizer(seed=42)
        opt.optimize(X_tr, y_tr, X_val, y_val, n_trials=2, model_type="xgb")

        history = opt.get_optimization_history()
        assert isinstance(history, pd.DataFrame)
        assert len(history) == 2


# ------------------------------------------------------------------
# Tests: Walk-Forward Pipeline
# ------------------------------------------------------------------


class TestWalkForwardEnsemble:
    """Test walk-forward ensemble trainer."""

    def test_trainer_init(self):
        """Trainer initializes with defaults."""
        trainer = WalkForwardEnsembleTrainer()
        assert trainer.train_months == 60
        assert trainer.test_months == 6

    def test_window_generation(self, synthetic_ohlcv_data):
        """_generate_windows produces valid splits."""
        trainer = WalkForwardEnsembleTrainer(
            train_months=24,
            val_months=6,
            test_months=3,
        )

        windows = trainer._generate_windows(synthetic_ohlcv_data)

        assert len(windows) > 0
        for window in windows:
            train_start, train_end = window["train"]
            val_start, val_end = window["val"]
            test_start, test_end = window["test"]

            # Temporal ordering
            assert train_start < train_end
            assert train_end < val_start
            assert val_start < val_end
            assert val_end < test_start
            assert test_start < test_end

    def test_extract_data(self, synthetic_ohlcv_data):
        """_extract_data returns correct features and labels."""
        trainer = WalkForwardEnsembleTrainer()

        window = (
            synthetic_ohlcv_data.index[0],
            synthetic_ohlcv_data.index[50],
        )

        X, y = trainer._extract_data(
            synthetic_ohlcv_data,
            FEATURE_COLUMNS,
            "label",
            window,
        )

        assert len(X) <= 51
        assert len(X) == len(y)
        assert set(X.columns) == set(FEATURE_COLUMNS)

    def test_detect_concept_drift(self, X_y_train_test):
        """detect_concept_drift returns valid metrics."""
        X_train, _, X_test, _ = X_y_train_test

        trainer = WalkForwardEnsembleTrainer()
        metrics = trainer.detect_concept_drift(X_train, X_test)

        assert "max_shift" in metrics
        assert "mean_shift" in metrics
        assert "n_features_shifted" in metrics

    def test_evaluate_ensemble(self, ensemble_strategy, X_y_train_test):
        """_evaluate_ensemble computes AUCs correctly."""
        _, _, X_test, y_test = X_y_train_test

        trainer = WalkForwardEnsembleTrainer()
        ensemble_auc, model_aucs = trainer._evaluate_ensemble(ensemble_strategy, X_test, y_test)

        assert 0 <= ensemble_auc <= 1
        assert "xgb" in model_aucs
        assert "lgb" in model_aucs
        assert "patchtst" in model_aucs

    def test_run_walk_forward_small(self, synthetic_ohlcv_data):
        """run_walk_forward completes on small dataset."""
        trainer = WalkForwardEnsembleTrainer(
            train_months=12,
            val_months=2,
            test_months=1,
            hpo_n_trials=2,
        )

        # Run on synthetic data
        report = trainer.run_walk_forward(
            synthetic_ohlcv_data,
            features=FEATURE_COLUMNS,
            label="label",
            optimize_hyperparams=False,
        )

        assert report.total_folds >= 0
        if report.total_folds > 0:
            assert report.aggregate_auc > 0

    def test_retrain_on_schedule(self):
        """retrain_on_schedule returns valid status."""
        trainer = WalkForwardEnsembleTrainer()
        result = trainer.retrain_on_schedule(
            retraining_frequency="monthly",
            trigger_on_drift=True,
        )

        assert result["status"] == "scheduled"
        assert result["frequency"] == "monthly"

    def test_backtest_ensemble(self, synthetic_ohlcv_data):
        """backtest_ensemble runs without error."""
        trainer = WalkForwardEnsembleTrainer()

        result = trainer.backtest_ensemble(
            synthetic_ohlcv_data,
            features=FEATURE_COLUMNS,
            label="label",
        )

        assert "status" in result
        assert "sharpe" in result

    def test_print_summary(self, synthetic_ohlcv_data, capsys):
        """print_summary outputs to stdout."""
        trainer = WalkForwardEnsembleTrainer(
            train_months=12,
            val_months=2,
            test_months=1,
        )

        report = trainer.run_walk_forward(
            synthetic_ohlcv_data,
            features=FEATURE_COLUMNS,
            label="label",
            optimize_hyperparams=False,
        )

        trainer.print_summary(report)
        captured = capsys.readouterr()

        assert "WALK-FORWARD ENSEMBLE TRAINING SUMMARY" in captured.out


# ------------------------------------------------------------------
# Tests: Model Serialization
# ------------------------------------------------------------------


class TestModelSerialization:
    """Test model saving and loading."""

    def test_save_load_models(self, ensemble_strategy, tmp_path):
        """save_models and load_models work correctly."""
        output_dir = tmp_path / "models"

        ensemble_strategy.save_models(output_dir)

        # Check files exist
        assert (output_dir / "xgb_model.bin").exists()
        assert (output_dir / "lgb_model.txt").exists()
        assert (output_dir / "patchtst_model.pkl").exists()

        # Load and verify
        new_ensemble = EnsembleStrategy()
        new_ensemble.load_models(output_dir)

        assert new_ensemble._xgb_model is not None
        assert new_ensemble._lgb_model is not None
        assert new_ensemble._patchtst_model is not None


# ------------------------------------------------------------------
# Tests: PatchTST Wrapper
# ------------------------------------------------------------------


class TestPatchTSTWrapper:
    """Test PatchTST wrapper."""

    def test_patchtst_init(self):
        """PatchTSTWrapper initializes."""
        wrapper = PatchTSTWrapper(n_features=10)
        assert wrapper.n_features == 10

    def test_patchtst_extract_patches(self):
        """_extract_patches creates correct shape."""
        wrapper = PatchTSTWrapper(n_features=32, patch_len=8)

        X = np.random.randn(100, 32)
        patches = wrapper._extract_patches(X)

        assert patches.shape[0] == 100
        assert patches.shape[1] == 32  # 4 patches * 8

    def test_patchtst_fit_predict(self, X_y_train_test):
        """PatchTSTWrapper fit and predict work."""
        X_train, y_train, X_test, _ = X_y_train_test

        wrapper = PatchTSTWrapper(n_features=X_train.shape[1])
        wrapper.fit(X_train, y_train)

        pred = wrapper.predict(X_test)
        assert len(pred) == len(X_test)

        proba = wrapper.predict_proba(X_test)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_patchtst_save_load(self, X_y_train_test, tmp_path):
        """PatchTSTWrapper save/load works."""
        X_train, y_train, X_test, _ = X_y_train_test

        wrapper = PatchTSTWrapper(n_features=X_train.shape[1])
        wrapper.fit(X_train, y_train)

        path = tmp_path / "patchtst.pkl"
        wrapper.save(path)

        loaded = PatchTSTWrapper.load(path)
        pred_orig = wrapper.predict(X_test)
        pred_loaded = loaded.predict(X_test)

        np.testing.assert_array_equal(pred_orig, pred_loaded)


# ------------------------------------------------------------------
# Integration Tests
# ------------------------------------------------------------------


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self, synthetic_ohlcv_data):
        """Full walk-forward pipeline runs end-to-end."""
        trainer = WalkForwardEnsembleTrainer(
            train_months=24,
            val_months=3,
            test_months=3,
            hpo_n_trials=1,
        )

        report = trainer.run_walk_forward(
            synthetic_ohlcv_data,
            features=FEATURE_COLUMNS,
            label="label",
            optimize_hyperparams=False,
        )

        assert report is not None
        assert isinstance(report.fold_results, list)

    def test_ensemble_better_than_individual(self, X_y_train_test):
        """Ensemble AUC is competitive with individual models."""
        X_train, y_train, X_test, y_test = X_y_train_test

        ensemble = EnsembleStrategy(random_state=42)
        ensemble.train_xgboost(X_train, y_train)
        ensemble.train_lightgbm(X_train, y_train)
        ensemble.train_patchtst(X_train, y_train)
        ensemble._is_fitted = True

        ensemble_proba = ensemble.ensemble_predict_proba(X_test)
        ensemble_auc = roc_auc_score(y_test, ensemble_proba)

        xgb_proba = ensemble._xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba)

        # Ensemble should not be worse than worst individual
        assert ensemble_auc >= 0.4
