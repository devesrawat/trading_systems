"""
Walk-forward training loop for the XGBoost signal model.

Rules (non-negotiable):
  - Splits are always temporal — never random
  - Purge gap between train end and test start prevents leakage at boundaries
  - Minimum 5 folds before trusting aggregate metrics
  - Train window: rolling 24 months (not expanding)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import structlog
import xgboost as xgb
from sklearn.metrics import roc_auc_score

log = structlog.get_logger(__name__)

# Minimum rows needed to form at least one valid fold
_MIN_ROWS_REQUIRED = 200

# XGBoost base params (overridden by Optuna if tune=True)
_BASE_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,
    "use_label_encoder": False,
}


@dataclass
class FoldResult:
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    auc: float
    precision: float
    recall: float
    n_train: int
    n_test: int
    model: xgb.XGBClassifier = field(repr=False)
    mlflow_run_id: str = ""


class WalkForwardTrainer:
    """
    Temporally-split walk-forward training and evaluation.

    Usage
    -----
    trainer = WalkForwardTrainer(train_months=24, test_months=3)
    results = trainer.run(df, features=FEATURE_COLUMNS, label="label")
    model   = trainer.best_model()
    trainer.print_summary()
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 3,
        purge_days: int = 5,
    ) -> None:
        if purge_days < 0:
            raise ValueError("purge_days must be >= 0")
        self.train_months = train_months
        self.test_months = test_months
        self.purge_days = purge_days
        self._fold_results: list[FoldResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        features: list[str],
        label: str,
        tune: bool = False,
        experiment_name: str = "nse_equity_signals",
    ) -> dict[str, Any]:
        """
        Execute the walk-forward loop.

        Parameters
        ----------
        df      : DataFrame with feature + label columns and DatetimeIndex
        features: list of feature column names
        label   : name of the binary target column
        tune    : if True, run Optuna hyperparameter search on first fold

        Returns a dict with keys: folds (list[FoldResult]), aggregate metrics
        """
        self._validate_inputs(df, features, label)

        mlflow.set_experiment(experiment_name)
        self._fold_results = []

        folds = self._generate_folds(df)
        base_params = _BASE_PARAMS.copy()

        if tune and folds:
            train_df, test_df = folds[0]
            base_params = self._tune_hyperparams(train_df, test_df, features, label)
            log.info("hyperparams_tuned", params=base_params)

        for i, (train_df, test_df) in enumerate(folds):
            fold_result = self._train_fold(
                fold_index=i,
                train_df=train_df,
                test_df=test_df,
                features=features,
                label=label,
                params=base_params,
            )
            self._fold_results.append(fold_result)
            log.info(
                "fold_complete",
                fold=i,
                auc=round(fold_result.auc, 4),
                n_test=fold_result.n_test,
            )

        return self._aggregate_results()

    def save_drift_reference(self, df: pd.DataFrame, features: list[str]) -> None:
        """
        Save a random sample from *df* as the drift detection reference.

        Call after run() to persist the training distribution. The reference
        is used by ConceptDriftDetector.check() during post-market drift checks.
        """
        try:
            from monitoring.drift_detector import ConceptDriftDetector
            ConceptDriftDetector().fit(df, features)
            log.info("drift_reference_saved_after_training")
        except Exception as exc:
            log.warning("drift_reference_save_failed", error=str(exc))

    def best_model(self) -> xgb.XGBClassifier:
        """Return the XGBoost model from the fold with highest out-of-sample AUC."""
        if not self._fold_results:
            raise RuntimeError("No results yet. Call run() first.")
        best = max(self._fold_results, key=lambda r: r.auc)
        return best.model

    def print_summary(self) -> None:
        """Print a formatted table of fold results + aggregate metrics."""
        if not self._fold_results:
            print("No results. Call run() first.")
            return

        header = f"{'Fold':>4}  {'Train Start':>12}  {'Test Start':>12}  {'AUC':>6}  {'Prec':>6}  {'Rec':>6}  {'N-Test':>6}"
        print(header)
        print("-" * len(header))
        for r in self._fold_results:
            print(
                f"{r.fold_index:>4}  "
                f"{str(r.train_start.date()):>12}  "
                f"{str(r.test_start.date()):>12}  "
                f"{r.auc:>6.4f}  "
                f"{r.precision:>6.4f}  "
                f"{r.recall:>6.4f}  "
                f"{r.n_test:>6}"
            )
        agg = self._aggregate_results()
        print("-" * len(header))
        print(
            f"{'AGG':>4}  {'':>12}  {'':>12}  "
            f"{agg['mean_auc']:>6.4f}  "
            f"{agg['mean_precision']:>6.4f}  "
            f"{agg['mean_recall']:>6.4f}"
        )

    # ------------------------------------------------------------------
    # Fold generation
    # ------------------------------------------------------------------

    def _generate_folds(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split *df* into (train, test) pairs using a rolling window.

        Train window = train_months months.
        Gap          = purge_days business days.
        Test window  = test_months months.
        Rolling: each fold advances by test_months months.
        """
        folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        start = df.index.min()
        end = df.index.max()

        train_delta = pd.DateOffset(months=self.train_months)
        test_delta = pd.DateOffset(months=self.test_months)
        purge_delta = timedelta(days=self.purge_days)

        fold_start = start
        while True:
            train_end = fold_start + train_delta
            test_start = train_end + purge_delta
            test_end = test_start + test_delta

            if test_end > end:
                break

            train_df = df.loc[(df.index >= fold_start) & (df.index < train_end)]
            test_df = df.loc[(df.index >= test_start) & (df.index < test_end)]

            if len(train_df) < 50 or len(test_df) < 10:
                fold_start += test_delta
                continue

            folds.append((train_df, test_df))
            fold_start += test_delta

        return folds

    # ------------------------------------------------------------------
    # Single fold training
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        fold_index: int,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        features: list[str],
        label: str,
        params: dict[str, Any],
    ) -> FoldResult:
        X_train = train_df[features].values.astype(np.float32)
        y_train = train_df[label].values
        X_test = test_df[features].values.astype(np.float32)
        y_test = test_df[label].values

        early_stop = params.get("early_stopping_rounds", 50)
        model_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        model_params["early_stopping_rounds"] = early_stop

        model = xgb.XGBClassifier(**model_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        probs = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, probs))

        preds = (probs >= 0.5).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        run_id = ""
        with mlflow.start_run(run_name=f"fold_{fold_index}", nested=True):
            mlflow.log_params({
                "fold": fold_index,
                "train_start": str(train_df.index.min().date()),
                "train_end": str(train_df.index.max().date()),
                "test_start": str(test_df.index.min().date()),
                "n_train": len(train_df),
                "n_test": len(test_df),
            })
            mlflow.log_metrics({"auc": auc, "precision": precision, "recall": recall})
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""

        return FoldResult(
            fold_index=fold_index,
            train_start=train_df.index.min(),
            train_end=train_df.index.max(),
            test_start=test_df.index.min(),
            test_end=test_df.index.max(),
            auc=auc,
            precision=precision,
            recall=recall,
            n_train=len(train_df),
            n_test=len(test_df),
            model=model,
            mlflow_run_id=run_id,
        )

    # ------------------------------------------------------------------
    # Optuna hyperparameter tuning
    # ------------------------------------------------------------------

    def _tune_hyperparams(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        features: list[str],
        label: str,
        n_trials: int = 20,
    ) -> dict[str, Any]:
        X_train = train_df[features].values.astype(np.float32)
        y_train = train_df[label].values
        X_val = val_df[features].values.astype(np.float32)
        y_val = val_df[label].values

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 20, 100),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "auc",
            }
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            probs = model.predict_proba(X_val)[:, 1]
            return float(roc_auc_score(y_val, probs))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params.copy()
        best["eval_metric"] = "auc"
        best["early_stopping_rounds"] = 50
        best["random_state"] = 42
        best["n_jobs"] = -1
        return best

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        features: list[str],
        label: str,
    ) -> None:
        if label not in df.columns:
            raise ValueError(f"Label column '{label}' not found in DataFrame.")
        missing_features = set(features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
        if len(df) < _MIN_ROWS_REQUIRED:
            raise ValueError(
                f"DataFrame has {len(df)} rows — insufficient for walk-forward "
                f"training (minimum {_MIN_ROWS_REQUIRED} required)."
            )

    def _aggregate_results(self) -> dict[str, Any]:
        if not self._fold_results:
            return {"folds": []}
        aucs = [r.auc for r in self._fold_results]
        precs = [r.precision for r in self._fold_results]
        recs = [r.recall for r in self._fold_results]
        return {
            "folds": self._fold_results,
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_precision": float(np.mean(precs)),
            "mean_recall": float(np.mean(recs)),
            "n_folds": len(self._fold_results),
        }
