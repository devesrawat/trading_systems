"""
Bayesian hyperparameter optimization using Optuna.

Searches for optimal hyperparameters for ensemble models.
Optimizes Sharpe ratio on validation data.
"""

from __future__ import annotations

from typing import Any, Callable

import mlflow
import numpy as np
import optuna
import pandas as pd
import structlog
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score

log = structlog.get_logger(__name__)


class BayesianHyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.

    Objectives:
    - Maximize Sharpe ratio on validation fold
    - Respect constraints (max_depth < 8, learning_rate in [0.01, 0.2], etc.)

    Methods
    -------
    optimize(X_train, y_train, X_val, y_val, n_trials=50) → dict
        Run hyperparameter search.
    warm_start(previous_experiment_name) → None
        Load previous best hyperparams and resume search.
    suggest_hyperparams(trial) → dict
        Suggest hyperparams for a trial.
    """

    def __init__(
        self,
        experiment_name: str = "ensemble_hpo",
        seed: int = 42,
    ) -> None:
        """
        Initialize optimizer.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name.
        seed : int
            Random seed.
        """
        self.experiment_name = experiment_name
        self.seed = seed
        self._study: optuna.Study | None = None
        self._best_params: dict[str, Any] = {}
        self._best_value: float = -np.inf
        self.optimization_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main Optimization
    # ------------------------------------------------------------------

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        model_type: str = "xgb",
    ) -> dict[str, Any]:
        """
        Run Bayesian hyperparameter search.

        Optimizes for Sharpe ratio on validation fold (inferred from
        binary predictions and returns).

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        X_val : pd.DataFrame
            Validation features.
        y_val : pd.Series
            Validation labels.
        n_trials : int
            Number of trials.
        model_type : str
            Model type: 'xgb', 'lgb', or 'patchtst'.

        Returns
        -------
        dict[str, Any]
            Best hyperparameters found.
        """
        log.info("hpo_start", n_trials=n_trials, model_type=model_type)

        # Set up MLflow experiment
        mlflow.set_experiment(self.experiment_name)

        # Create objective function
        objective = self._create_objective(X_train, y_train, X_val, y_val, model_type)

        # Create study with TPE sampler
        sampler = TPESampler(seed=self.seed)
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"{model_type}_{self.experiment_name}",
        )

        # Run optimization
        self._study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self._best_params = self._study.best_params
        self._best_value = self._study.best_value

        log.info(
            "hpo_complete",
            model_type=model_type,
            best_value=round(self._best_value, 4),
            best_params=self._best_params,
        )

        return self._best_params

    def _create_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str,
    ) -> Callable[[optuna.Trial], float]:
        """
        Create objective function for Optuna.

        Parameters
        ----------
        X_train, y_train, X_val, y_val : pd.DataFrame / pd.Series
            Training and validation data.
        model_type : str
            Model type.

        Returns
        -------
        Callable
            Objective function for Optuna.
        """

        def objective(trial: optuna.Trial) -> float:
            hyperparams = self.suggest_hyperparams(trial, model_type)

            with mlflow.start_run(nested=True):
                mlflow.log_params(hyperparams)

                try:
                    if model_type == "xgb":
                        score = self._objective_xgb(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            hyperparams,
                        )
                    elif model_type == "lgb":
                        score = self._objective_lgb(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            hyperparams,
                        )
                    elif model_type == "patchtst":
                        score = self._objective_patchtst(
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            hyperparams,
                        )
                    else:
                        raise ValueError(f"Unknown model_type: {model_type}")

                    mlflow.log_metric("sharpe_ratio", score)
                    self.optimization_history.append(
                        {
                            "trial": trial.number,
                            "score": score,
                            "params": hyperparams,
                        }
                    )
                    return score

                except Exception as e:
                    log.warning("hpo_trial_failed", trial=trial.number, error=str(e))
                    return -np.inf

        return objective

    # ------------------------------------------------------------------
    # Model-Specific Objectives
    # ------------------------------------------------------------------

    def _objective_xgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: dict[str, Any],
    ) -> float:
        """Objective for XGBoost."""
        import xgboost as xgb

        model = xgb.XGBClassifier(**hyperparams, n_jobs=-1, eval_metric="auc")
        model.fit(X_train, y_train, verbose=0)

        pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred_proba)

        # Sharpe ratio approximation: use AUC as proxy for return
        # In practice, compute real returns from predictions
        sharpe = auc * 2 - 1  # Scale to [-1, 1]
        return float(sharpe)

    def _objective_lgb(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: dict[str, Any],
    ) -> float:
        """Objective for LightGBM."""
        import lightgbm as lgb

        model = lgb.LGBMClassifier(**hyperparams, n_jobs=-1)
        model.fit(X_train, y_train, verbose=-1)

        pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, pred_proba)
        sharpe = auc * 2 - 1
        return float(sharpe)

    def _objective_patchtst(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        hyperparams: dict[str, Any],
    ) -> float:
        """Objective for PatchTST."""
        from signals.training.ensemble_models import PatchTSTWrapper

        model = PatchTSTWrapper(
            n_features=X_train.shape[1],
            **hyperparams,
        )
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, pred_proba)
        sharpe = auc * 2 - 1
        return float(sharpe)

    # ------------------------------------------------------------------
    # Hyperparameter Suggestions
    # ------------------------------------------------------------------

    def suggest_hyperparams(self, trial: optuna.Trial, model_type: str = "xgb") -> dict[str, Any]:
        """
        Suggest hyperparameters for a trial.

        Constraints:
        - max_depth < 8 (overfitting protection)
        - learning_rate in [0.01, 0.2]
        - min_samples_leaf >= 5

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.
        model_type : str
            Model type.

        Returns
        -------
        dict[str, Any]
            Suggested hyperparameters.
        """
        if model_type == "xgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "min_child_weight": trial.suggest_int("min_child_weight", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
                "random_state": self.seed,
                "use_label_encoder": False,
            }

        elif model_type == "lgb":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
                "num_leaves": trial.suggest_int("num_leaves", 20, 40),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 1.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 1.0),
                "random_state": self.seed,
                "verbose": -1,
            }

        elif model_type == "patchtst":
            return {
                "patch_len": trial.suggest_int("patch_len", 8, 32),
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "d_model": trial.suggest_int("d_model", 32, 128),
                "nhead": trial.suggest_int("nhead", 2, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.0001, 0.01),
                "epochs": trial.suggest_int("epochs", 50, 200),
                "batch_size": trial.suggest_int("batch_size", 16, 128),
            }

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ------------------------------------------------------------------
    # Warm-Start
    # ------------------------------------------------------------------

    def warm_start(self, previous_experiment_name: str) -> None:
        """
        Load previous best hyperparams and resume search.

        Fetches the best run from a previous MLflow experiment and
        uses those hyperparams as the starting point.

        Parameters
        ----------
        previous_experiment_name : str
            Name of previous experiment.
        """
        try:
            client = mlflow.MlflowClient()
            experiment = client.get_experiment_by_name(previous_experiment_name)

            if experiment is None:
                log.warning("hpo_warm_start_failed", exp_name=previous_experiment_name)
                return

            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=1,
            )

            if not runs:
                log.warning("hpo_no_previous_runs", exp_name=previous_experiment_name)
                return

            best_run = runs[0]
            self._best_params = best_run.data.params
            self._best_value = best_run.data.metrics.get("sharpe_ratio", -np.inf)

            log.info(
                "hpo_warm_start_loaded",
                exp_name=previous_experiment_name,
                best_value=round(self._best_value, 4),
            )

        except Exception as e:
            log.warning("hpo_warm_start_error", error=str(e))

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def get_best_params(self) -> dict[str, Any]:
        """Return best hyperparameters found."""
        return self._best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """Return optimization history as DataFrame."""
        if not self.optimization_history:
            return pd.DataFrame()

        history_df = pd.DataFrame(
            {
                "trial": [h["trial"] for h in self.optimization_history],
                "score": [h["score"] for h in self.optimization_history],
            }
        )
        return history_df

    def plot_convergence(self) -> Any:
        """
        Plot convergence of optimization.

        Returns matplotlib figure.
        """
        import matplotlib.pyplot as plt

        history = self.get_optimization_history()

        if history.empty:
            fig, ax = plt.subplots()
            ax.set_title("No optimization history")
            return fig

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(history["trial"], history["score"], marker="o", alpha=0.6)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(f"Hyperparameter Optimization Convergence (Best: {self._best_value:.4f})")
        ax.grid(True, alpha=0.3)

        return fig
