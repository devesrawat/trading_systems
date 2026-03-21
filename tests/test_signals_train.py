"""Unit tests for signals/train.py — TDD RED phase. Uses synthetic data, mocks MLflow."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from signals.features import FEATURE_COLUMNS

try:
    from signals.train import WalkForwardTrainer
except Exception:
    pytest.skip("xgboost/libomp not available on this system", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_training_df(n: int = 600) -> pd.DataFrame:
    """Synthetic feature + label DataFrame spanning ~2.5 years of business days."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    data = {col: rng.uniform(-1, 1, n) for col in FEATURE_COLUMNS}
    data["label"] = rng.integers(0, 2, n).astype(float)
    data["forward_return_5d"] = rng.uniform(-0.05, 0.08, n)
    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# WalkForwardTrainer — constructor / config
# ---------------------------------------------------------------------------

class TestWalkForwardTrainerConfig:
    def test_default_params(self):
        wft = WalkForwardTrainer()
        assert wft.train_months == 24
        assert wft.test_months == 3
        assert wft.purge_days == 5

    def test_custom_params(self):
        wft = WalkForwardTrainer(train_months=12, test_months=2, purge_days=10)
        assert wft.train_months == 12
        assert wft.test_months == 2
        assert wft.purge_days == 10

    def test_purge_days_cannot_be_negative(self):
        with pytest.raises(ValueError):
            WalkForwardTrainer(purge_days=-1)


# ---------------------------------------------------------------------------
# WalkForwardTrainer — fold splitting
# ---------------------------------------------------------------------------

class TestFoldSplitting:
    def test_generates_at_least_one_fold(self):
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        folds = wft._generate_folds(df)
        assert len(folds) >= 1

    def test_fold_train_ends_before_test_starts(self):
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        for train_df, test_df in wft._generate_folds(df):
            assert train_df.index.max() < test_df.index.min()

    def test_purge_gap_enforced(self):
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        for train_df, test_df in wft._generate_folds(df):
            gap = (test_df.index.min() - train_df.index.max()).days
            assert gap >= wft.purge_days

    def test_no_data_leakage_between_folds(self):
        """Train indices must never appear in test indices."""
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        for train_df, test_df in wft._generate_folds(df):
            overlap = set(train_df.index) & set(test_df.index)
            assert len(overlap) == 0

    def test_no_random_shuffle(self):
        """Fold splits must be purely temporal — same df always produces same folds."""
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        folds_a = wft._generate_folds(df)
        folds_b = wft._generate_folds(df)
        for (ta, _), (tb, _) in zip(folds_a, folds_b):
            pd.testing.assert_index_equal(ta.index, tb.index)


# ---------------------------------------------------------------------------
# WalkForwardTrainer — run()
# ---------------------------------------------------------------------------

class TestWalkForwardRun:
    @patch("signals.train.mlflow")
    def test_run_returns_fold_results(self, mock_mlflow):
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        results = wft.run(df, features=FEATURE_COLUMNS, label="label")
        assert isinstance(results, dict)
        assert "folds" in results

    @patch("signals.train.mlflow")
    def test_each_fold_has_auc_metric(self, mock_mlflow):
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        results = wft.run(df, features=FEATURE_COLUMNS, label="label")
        for fold in results["folds"]:
            assert hasattr(fold, "auc")
            assert 0.0 <= fold.auc <= 1.0

    @patch("signals.train.mlflow")
    def test_run_logs_to_mlflow(self, mock_mlflow):
        mock_run = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        wft.run(df, features=FEATURE_COLUMNS, label="label")
        assert mock_mlflow.log_metrics.called or mock_mlflow.log_metric.called

    @patch("signals.train.mlflow")
    def test_missing_label_column_raises(self, mock_mlflow):
        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500).drop(columns=["label"])
        with pytest.raises(ValueError, match="label"):
            wft.run(df, features=FEATURE_COLUMNS, label="label")

    @patch("signals.train.mlflow")
    def test_insufficient_data_raises(self, mock_mlflow):
        wft = WalkForwardTrainer(train_months=24, test_months=3, purge_days=5)
        df = _make_training_df(50)   # way too short
        with pytest.raises(ValueError, match="insufficient"):
            wft.run(df, features=FEATURE_COLUMNS, label="label")


# ---------------------------------------------------------------------------
# WalkForwardTrainer — best_model()
# ---------------------------------------------------------------------------

class TestBestModel:
    @patch("signals.train.mlflow")
    def test_best_model_returns_xgboost(self, mock_mlflow):
        import xgboost as xgb
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        wft.run(df, features=FEATURE_COLUMNS, label="label")
        model = wft.best_model()
        assert isinstance(model, xgb.XGBClassifier)

    @patch("signals.train.mlflow")
    def test_best_model_raises_before_run(self, mock_mlflow):
        wft = WalkForwardTrainer()
        with pytest.raises(RuntimeError, match="run()"):
            wft.best_model()


# ---------------------------------------------------------------------------
# WalkForwardTrainer — print_summary()
# ---------------------------------------------------------------------------

class TestPrintSummary:
    @patch("signals.train.mlflow")
    def test_print_summary_runs_without_error(self, mock_mlflow, capsys):
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        wft = WalkForwardTrainer(train_months=12, test_months=3, purge_days=5)
        df = _make_training_df(500)
        wft.run(df, features=FEATURE_COLUMNS, label="label")
        wft.print_summary()
        captured = capsys.readouterr()
        assert "AUC" in captured.out or "Fold" in captured.out
