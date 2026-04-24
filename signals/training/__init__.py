"""
Ensemble training pipeline for walk-forward backtesting.

Modules:
  - ensemble_models.py — XGBoost, LightGBM, PatchTST ensemble
  - walk_forward_ensemble.py — Main walk-forward trainer
  - concept_drift.py — Drift detection and monitoring
  - hyperparameter_optimizer.py — Bayesian hyperparameter search
"""

from __future__ import annotations
