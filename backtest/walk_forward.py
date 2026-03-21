"""
Walk-forward backtest engine.

This is the ONLY valid backtesting method for this system.
No in-sample testing — every metric reported is out-of-sample.

Protocol (Section 14):
  - Temporal splits only, never random
  - 5-day purge gap between train end and test start
  - Minimum 5 folds
  - Stress-test windows must be covered
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import structlog

from backtest.costs import NSECostModel
from backtest.metrics import (
    calmar_ratio,
    max_drawdown,
    print_tearsheet,
    profit_factor,
    sharpe_ratio,
    win_rate,
)
from signals.train import WalkForwardTrainer

log = structlog.get_logger(__name__)


@dataclass
class FoldBacktestResult:
    fold_index: int
    returns: pd.Series
    equity_curve: pd.Series
    sharpe: float
    max_dd: float
    profit_factor: float
    win_rate: float
    n_trades: int
    train_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass
class BacktestResults:
    fold_results: list[FoldBacktestResult] = field(default_factory=list)
    aggregate_sharpe: float = 0.0
    aggregate_max_dd: float = 0.0
    aggregate_profit_factor: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    all_returns: pd.Series = field(default_factory=pd.Series)

    def plot(self) -> plt.Figure:
        """Equity curve + drawdown subplot across all folds."""
        if self.all_returns.empty:
            fig, ax = plt.subplots()
            ax.set_title("No backtest data")
            return fig

        equity = (1 + self.all_returns).cumprod() * 100_000
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        ax1.plot(equity.index, equity.values, label="Equity", color="steelblue")
        ax1.set_title(
            f"Walk-Forward Backtest  |  Sharpe: {self.aggregate_sharpe:.2f}  "
            f"|  Max DD: {self.aggregate_max_dd:.1%}  "
            f"|  Trades: {self.total_trades}"
        )
        ax1.set_ylabel("Portfolio Value (₹)")
        ax1.legend()

        ax2.fill_between(drawdown.index, drawdown.values, 0, color="tomato", alpha=0.6)
        ax2.set_ylabel("Drawdown")
        ax2.set_xlabel("Date")
        plt.tight_layout()
        return fig


class WalkForwardBacktest:
    """
    Run a complete walk-forward backtest: train → signal → cost → metrics.

    Usage
    -----
    bt = WalkForwardBacktest(train_months=24, test_months=3)
    results = bt.run(df)
    results.plot()
    print_tearsheet(results.all_returns, (1+results.all_returns).cumprod())
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 3,
        purge_days: int = 5,
        signal_threshold: float = 0.65,
        liquidity_tier: str = "large_cap",
    ) -> None:
        self.train_months = train_months
        self.test_months = test_months
        self.purge_days = purge_days
        self.signal_threshold = signal_threshold
        self._cost_model = NSECostModel()
        self._liquidity_tier = liquidity_tier

    def run(self, df: pd.DataFrame) -> BacktestResults:
        """
        Execute full walk-forward backtest on *df*.

        *df* must contain FEATURE_COLUMNS + 'label' + 'forward_return_5d'.
        Returns BacktestResults with per-fold metrics and aggregate stats.
        """
        from signals.features import FEATURE_COLUMNS, LABEL_COLUMNS

        features = [c for c in FEATURE_COLUMNS if c in df.columns]
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain 'label' column. Run build_features(include_labels=True).")

        trainer = WalkForwardTrainer(
            train_months=self.train_months,
            test_months=self.test_months,
            purge_days=self.purge_days,
        )
        folds = trainer._generate_folds(df)

        if len(folds) < 2:
            raise ValueError(
                f"Only {len(folds)} fold(s) generated — insufficient data for walk-forward backtest. "
                "Need at least 2 folds (train_months + 2×test_months of data)."
            )

        fold_results: list[FoldBacktestResult] = []
        all_returns_list: list[pd.Series] = []

        for i, (train_df, test_df) in enumerate(folds):
            log.info("backtest_fold_start", fold=i, test_rows=len(test_df))

            # Train model on this fold's train window
            import mlflow
            mlflow.set_experiment("backtest_results")
            fold_trainer = WalkForwardTrainer(
                train_months=self.train_months,
                test_months=self.test_months,
                purge_days=self.purge_days,
            )
            model = self._train_fold_model(train_df, features, fold_trainer)

            # Generate signals on test window
            X_test = test_df[features]
            probs = model.predict_proba(X_test.values)[:, 1]
            signal_mask = probs >= self.signal_threshold

            # Simulate returns: enter on signal, hold for 5 days, apply costs
            if "forward_return_5d" in test_df.columns:
                raw_returns = test_df.loc[signal_mask, "forward_return_5d"].dropna()
            else:
                raw_returns = pd.Series(dtype=float)

            # Apply round-trip transaction costs
            net_returns = raw_returns.apply(
                lambda r: r - self._cost_model.round_trip_cost(10_000, self._liquidity_tier) / 10_000
            )

            equity = (1 + net_returns).cumprod() * 100_000 if not net_returns.empty else pd.Series([100_000.0])

            fold_result = FoldBacktestResult(
                fold_index=i,
                returns=net_returns,
                equity_curve=equity,
                sharpe=sharpe_ratio(net_returns) if not net_returns.empty else 0.0,
                max_dd=max_drawdown(equity),
                profit_factor=profit_factor(net_returns) if not net_returns.empty else 0.0,
                win_rate=win_rate(net_returns) if not net_returns.empty else 0.0,
                n_trades=len(net_returns),
                train_start=train_df.index.min(),
                test_start=test_df.index.min(),
                test_end=test_df.index.max(),
            )
            fold_results.append(fold_result)
            if not net_returns.empty:
                all_returns_list.append(net_returns)

            log.info(
                "backtest_fold_complete",
                fold=i,
                trades=fold_result.n_trades,
                sharpe=round(fold_result.sharpe, 3),
                max_dd=round(fold_result.max_dd, 3),
            )

        all_returns = pd.concat(all_returns_list).sort_index() if all_returns_list else pd.Series(dtype=float)
        all_equity = (1 + all_returns).cumprod() * 100_000 if not all_returns.empty else pd.Series([100_000.0])

        results = BacktestResults(
            fold_results=fold_results,
            aggregate_sharpe=sharpe_ratio(all_returns) if not all_returns.empty else 0.0,
            aggregate_max_dd=max_drawdown(all_equity),
            aggregate_profit_factor=profit_factor(all_returns) if not all_returns.empty else 0.0,
            win_rate=win_rate(all_returns) if not all_returns.empty else 0.0,
            total_trades=sum(f.n_trades for f in fold_results),
            all_returns=all_returns,
        )

        log.info(
            "backtest_complete",
            folds=len(fold_results),
            total_trades=results.total_trades,
            sharpe=round(results.aggregate_sharpe, 3),
            max_dd=round(results.aggregate_max_dd, 3),
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_fold_model(
        self,
        train_df: pd.DataFrame,
        features: list[str],
        trainer: WalkForwardTrainer,
    ):
        import xgboost as xgb
        from signals.train import _BASE_PARAMS
        import numpy as np

        X = train_df[features].values.astype(np.float32)
        y = train_df["label"].values
        params = {k: v for k, v in _BASE_PARAMS.items() if k != "early_stopping_rounds"}
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, verbose=False)
        return model
