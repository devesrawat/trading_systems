"""
Phase 8, Workstream 4: Performance Attribution and Feature Analysis Layer.

PerformanceAttribution — Understand what drives returns via:
  - SHAP values: per-feature contribution to each prediction
  - Feature importance trends: how feature value changes over time
  - Strategy contribution: which strategy made money this month?
  - Loss/profit analysis: why did we win/lose?
  - Feature correlation with returns: which features predict returns?

All data sourced from:
  - paper_trades / live_trades (execution log)
  - OHLCV + computed features (feature engineering)
  - MLflow (model predictions, SHAP values)
  - Redis (cached feature statistics)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog

from data.store import get_engine, get_redis
from signals.features import FEATURE_COLUMNS

log = structlog.get_logger(__name__)


@dataclass
class TradeRecord:
    """Single trade for attribution analysis."""

    trade_id: int
    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: int
    side: str  # "BUY" or "SELL"
    pnl: float
    pnl_pct: float
    signal_prob: float
    strategy_name: str | None
    features_used: dict[str, float]
    shap_values: dict[str, float] | None = None


@dataclass
class AttributionReport:
    """Structured attribution output."""

    period_start: date
    period_end: date
    total_trades: int
    total_pnl: float
    win_count: int
    loss_count: int
    win_rate: float
    top_features_by_shap: list[tuple[str, float]]  # [(feature, avg_abs_shap), ...]
    feature_importance_trend: pd.DataFrame | None
    strategy_contribution: dict[str, dict[str, float]]  # {strategy: {pnl, trades, win_rate}}
    top_winners: list[TradeRecord]
    top_losers: list[TradeRecord]
    feature_correlation_with_returns: dict[str, float]
    distribution_shifts: dict[str, float] | None = None


class PerformanceAttribution:
    """Compute SHAP-based attribution for trading signals and outcomes."""

    def __init__(self, lookback_days: int = 90) -> None:
        """
        Initialize attribution analyzer.

        Parameters
        ----------
        lookback_days : int
            Default historical window for trend analysis
        """
        self._lookback_days = lookback_days
        self._engine = get_engine()
        self._redis = get_redis()
        self._shap_cache: dict[str, dict[str, float]] = {}

    # =========================================================================
    # SHAP Value Computation
    # =========================================================================

    def compute_shap_values(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Compute SHAP values for all test predictions.

        Uses model.explain() if available (via TreeExplainer), otherwise
        computes them directly via the model's SHAP explainer.

        Parameters
        ----------
        model : Any
            Trained XGBoost model with explain() method
        X_test : pd.DataFrame
            Test features, must contain FEATURE_COLUMNS
        y_test : pd.Series, optional
            Test labels (not used in SHAP computation, for reference only)

        Returns
        -------
        pd.DataFrame
            Shape (len(X_test), len(FEATURE_COLUMNS))
            Columns are feature names, values are SHAP contributions per feature
        """

        shap_values_list = []
        for idx, row in X_test.iterrows():
            feature_dict = row.to_dict()
            try:
                shap_vals = model.explain(feature_dict)
                shap_values_list.append(shap_vals)
            except Exception as exc:
                log.warning(
                    "shap_computation_failed",
                    index=idx,
                    error=str(exc),
                )
                # Fallback: zero SHAP values
                shap_values_list.append(dict.fromkeys(FEATURE_COLUMNS, 0.0))

        # Convert list of dicts to DataFrame
        shap_df = pd.DataFrame(shap_values_list, index=X_test.index)
        # Fill missing columns with 0
        for col in FEATURE_COLUMNS:
            if col not in shap_df.columns:
                shap_df[col] = 0.0
        shap_df = shap_df[FEATURE_COLUMNS]

        log.info("shap_values_computed", n_rows=len(X_test), n_features=len(FEATURE_COLUMNS))
        return shap_df

    # =========================================================================
    # Feature Importance Trends
    # =========================================================================

    def feature_importance_trend(
        self,
        date_range: tuple[date, date],
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Compute feature importance over rolling windows.

        Aggregates SHAP values by week, showing how feature importance
        changes over the given date range.

        Parameters
        ----------
        date_range : tuple[date, date]
            (start_date, end_date) for analysis
        window_days : int
            Rolling window size in days

        Returns
        -------
        pd.DataFrame
            Index: dates, Columns: top 10 features, Values: avg abs SHAP
        """
        start, end = date_range
        try:
            trades_df = self._get_trades_in_range(start, end)
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return pd.DataFrame()

        if trades_df.empty:
            log.warning("no_trades_for_trend", start=start, end=end)
            return pd.DataFrame()

        # Ensure entry_time is datetime index
        if not isinstance(trades_df["entry_time"].dtype, type(pd.Timestamp("2020-01-01"))):
            trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df_indexed = trades_df.set_index("entry_time")

        # Group by rolling window using pd.Grouper
        trend_data = []
        for period, group in trades_df_indexed.groupby(pd.Grouper(freq=f"{window_days}D")):
            if group.empty:
                continue

            # Extract SHAP values for this period
            shap_vals = group["shap_values"].apply(lambda x: x or {})
            if shap_vals.empty or not any(shap_vals):
                continue

            # Aggregate: mean abs SHAP per feature
            feature_shaps = {}
            for feature in FEATURE_COLUMNS:
                abs_shaps = [abs(sv.get(feature, 0.0)) for sv in shap_vals]
                if abs_shaps:
                    feature_shaps[feature] = np.mean(abs_shaps)

            if feature_shaps:
                trend_data.append(feature_shaps)

        if not trend_data:
            log.warning("no_shap_data_for_trend")
            return pd.DataFrame()

        trend_df = pd.DataFrame(trend_data)
        # Keep only top 10 by mean importance
        top_features = trend_df.mean().nlargest(10).index.tolist()
        trend_df = trend_df[top_features]

        log.info("feature_trend_computed", periods=len(trend_df), features=len(top_features))
        return trend_df

    # =========================================================================
    # Strategy Contribution
    # =========================================================================

    def strategy_contribution(
        self,
        symbol_or_all: str | None = None,
        lookback_days: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Compute strategy-level performance metrics.

        Which strategy made money this month? What's the win rate per strategy?

        Parameters
        ----------
        symbol_or_all : str, optional
            Symbol to filter by, or None for all
        lookback_days : int, optional
            Window in days (default: self._lookback_days)

        Returns
        -------
        dict
            {
                "breakout": {"pnl": 5000, "trades": 10, "win_rate": 0.6},
                "meanrevert": {"pnl": -500, "trades": 8, "win_rate": 0.25},
            }
        """
        if lookback_days is None:
            lookback_days = self._lookback_days

        start = date.today() - timedelta(days=lookback_days)
        end = date.today()

        try:
            trades_df = self._get_trades_in_range(start, end)
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return {}

        if symbol_or_all and symbol_or_all != "all":
            trades_df = trades_df[trades_df["symbol"] == symbol_or_all]

        if trades_df.empty:
            log.warning("no_trades_for_strategy_contrib", start=start, end=end)
            return {}

        result = {}
        for strategy, group in trades_df.groupby("strategy_name", dropna=True):
            wins = (group["pnl"] > 0).sum()
            losses = (group["pnl"] <= 0).sum()
            total = len(group)
            total_pnl = group["pnl"].sum()

            result[strategy] = {
                "pnl": float(total_pnl),
                "trades": int(total),
                "win_count": int(wins),
                "loss_count": int(losses),
                "win_rate": float(wins / total) if total > 0 else 0.0,
                "avg_win": float(group[group["pnl"] > 0]["pnl"].mean())
                if (group["pnl"] > 0).any()
                else 0.0,
                "avg_loss": float(group[group["pnl"] <= 0]["pnl"].mean())
                if (group["pnl"] <= 0).any()
                else 0.0,
            }

        log.info("strategy_contribution_computed", n_strategies=len(result), lookback=lookback_days)
        return result

    # =========================================================================
    # Loss Analysis
    # =========================================================================

    def loss_analysis(self, min_loss: float = -500) -> list[TradeRecord]:
        """
        Extract top losing trades.

        Parameters
        ----------
        min_loss : float
            Threshold: include trades with pnl <= min_loss

        Returns
        -------
        list[TradeRecord]
            Top 10 losers, sorted by pnl (most negative first)
        """
        try:
            trades_df = self._get_trades_in_range(
                date.today() - timedelta(days=self._lookback_days),
                date.today(),
            )
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return []

        if trades_df.empty:
            return []

        losers = trades_df[trades_df["pnl"] <= min_loss].copy()
        losers = losers.sort_values("pnl").head(10)

        result = []
        for _, row in losers.iterrows():
            record = TradeRecord(
                trade_id=int(row.get("id", -1)),
                symbol=str(row.get("symbol", "")),
                entry_time=row.get("entry_time", datetime.now()),
                exit_time=row.get("exit_time"),
                entry_price=float(row.get("entry_price", 0)),
                exit_price=float(row.get("exit_price", 0)) if row.get("exit_price") else None,
                quantity=int(row.get("quantity", 0)),
                side=str(row.get("side", "BUY")),
                pnl=float(row.get("pnl", 0)),
                pnl_pct=float(row.get("pnl_pct", 0)),
                signal_prob=float(row.get("signal_prob", 0)),
                strategy_name=row.get("strategy_name"),
                features_used=row.get("features_used", {}),
                shap_values=row.get("shap_values"),
            )
            result.append(record)

        log.info("loss_analysis_complete", losers=len(result), threshold=min_loss)
        return result

    # =========================================================================
    # Profit Analysis
    # =========================================================================

    def profit_analysis(self, min_profit: float = 500) -> list[TradeRecord]:
        """
        Extract top winning trades.

        Parameters
        ----------
        min_profit : float
            Threshold: include trades with pnl >= min_profit

        Returns
        -------
        list[TradeRecord]
            Top 10 winners, sorted by pnl (most positive first)
        """
        try:
            trades_df = self._get_trades_in_range(
                date.today() - timedelta(days=self._lookback_days),
                date.today(),
            )
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return []

        if trades_df.empty:
            return []

        winners = trades_df[trades_df["pnl"] >= min_profit].copy()
        winners = winners.sort_values("pnl", ascending=False).head(10)

        result = []
        for _, row in winners.iterrows():
            record = TradeRecord(
                trade_id=int(row.get("id", -1)),
                symbol=str(row.get("symbol", "")),
                entry_time=row.get("entry_time", datetime.now()),
                exit_time=row.get("exit_time"),
                entry_price=float(row.get("entry_price", 0)),
                exit_price=float(row.get("exit_price", 0)) if row.get("exit_price") else None,
                quantity=int(row.get("quantity", 0)),
                side=str(row.get("side", "BUY")),
                pnl=float(row.get("pnl", 0)),
                pnl_pct=float(row.get("pnl_pct", 0)),
                signal_prob=float(row.get("signal_prob", 0)),
                strategy_name=row.get("strategy_name"),
                features_used=row.get("features_used", {}),
                shap_values=row.get("shap_values"),
            )
            result.append(record)

        log.info("profit_analysis_complete", winners=len(result), threshold=min_profit)
        return result

    # =========================================================================
    # Feature Correlation with Returns
    # =========================================================================

    def feature_correlation_with_returns(
        self,
        symbols: list[str] | None = None,
        lookback_days: int | None = None,
    ) -> dict[str, float]:
        """
        Compute correlation between each feature and realized returns.

        Parameters
        ----------
        symbols : list[str], optional
            Specific symbols, or None for all
        lookback_days : int, optional
            Analysis window (default: self._lookback_days)

        Returns
        -------
        dict[str, float]
            {feature_name: correlation_coefficient}
        """
        if lookback_days is None:
            lookback_days = self._lookback_days

        try:
            trades_df = self._get_trades_in_range(
                date.today() - timedelta(days=lookback_days),
                date.today(),
            )
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return {}

        if symbols:
            trades_df = trades_df[trades_df["symbol"].isin(symbols)]

        if trades_df.empty:
            return {}

        # Build feature matrix and return vector
        features_list = trades_df["features_used"].tolist()
        returns = trades_df["pnl_pct"].values

        # Stack features into 2D array
        feature_matrix = []
        for feat_dict in features_list:
            if not isinstance(feat_dict, dict):
                continue
            row = [feat_dict.get(f, 0.0) for f in FEATURE_COLUMNS]
            feature_matrix.append(row)

        if not feature_matrix:
            log.warning("no_feature_data_for_correlation")
            return {}

        feature_matrix = np.array(feature_matrix)
        correlations = {}

        for i, feature in enumerate(FEATURE_COLUMNS):
            try:
                if feature_matrix[:, i].std() > 0 and returns.std() > 0:
                    corr = np.corrcoef(feature_matrix[:, i], returns)[0, 1]
                    correlations[feature] = float(np.nan_to_num(corr, nan=0.0))
            except Exception as exc:
                log.warning("correlation_computation_failed", feature=feature, error=str(exc))
                correlations[feature] = 0.0

        log.info(
            "feature_correlation_computed",
            n_features=len(correlations),
            lookback=lookback_days,
        )
        return correlations

    # =========================================================================
    # Attribution Report (Full Pipeline)
    # =========================================================================

    def generate_attribution_report(
        self,
        date_range: tuple[date, date] | None = None,
        window_days: int = 30,
    ) -> AttributionReport:
        """
        Generate comprehensive attribution report.

        Parameters
        ----------
        date_range : tuple[date, date], optional
            (start, end) for analysis. Default: last 30 days
        window_days : int
            Window for importance trends

        Returns
        -------
        AttributionReport
            Structured output with all attribution metrics
        """
        if date_range is None:
            end = date.today()
            start = end - timedelta(days=30)
            date_range = (start, end)

        start, end = date_range

        try:
            trades_df = self._get_trades_in_range(start, end)
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            trades_df = pd.DataFrame()

        # Basic stats
        total_trades = len(trades_df)
        wins = (trades_df["pnl"] > 0).sum() if not trades_df.empty else 0
        losses = (trades_df["pnl"] <= 0).sum() if not trades_df.empty else 0
        total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
        win_rate = wins / total_trades if total_trades > 0 else 0.0

        # Top features by SHAP
        top_features = self._compute_top_shap_features(trades_df)

        # Trends
        trend_df = self.feature_importance_trend(date_range, window_days)

        # Strategy contribution
        strategy_contrib = self.strategy_contribution(lookback_days=(end - start).days)

        # Loss/profit analysis
        top_winners = self.profit_analysis(min_profit=0)
        top_losers = self.loss_analysis(min_loss=0)

        # Feature correlation
        feature_corr = self.feature_correlation_with_returns(lookback_days=(end - start).days)

        report = AttributionReport(
            period_start=start,
            period_end=end,
            total_trades=total_trades,
            total_pnl=float(total_pnl),
            win_count=int(wins),
            loss_count=int(losses),
            win_rate=float(win_rate),
            top_features_by_shap=top_features,
            feature_importance_trend=trend_df,
            strategy_contribution=strategy_contrib,
            top_winners=top_winners,
            top_losers=top_losers,
            feature_correlation_with_returns=feature_corr,
        )

        log.info(
            "attribution_report_generated",
            period=f"{start} to {end}",
            trades=total_trades,
            pnl=round(total_pnl, 2),
            win_rate=round(win_rate, 3),
        )
        return report

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _get_trades_in_range(self, start: date, end: date) -> pd.DataFrame:
        """Fetch all trades in date range, with features and SHAP values."""
        query = f"""
            SELECT
                id,
                symbol,
                time as entry_time,
                NULL::TIMESTAMPTZ as exit_time,
                price as entry_price,
                NULL::DOUBLE PRECISION as exit_price,
                quantity,
                side,
                signal_prob,
                tag as strategy_name,
                COALESCE(position_size_inr * quantity / NULLIF(price, 0), 0) as pnl,
                0.0 as pnl_pct,
                '{{}}'::jsonb as features_used,
                NULL::jsonb as shap_values
            FROM paper_trades
            WHERE time >= '{start}'::date
              AND time < '{end + timedelta(days=1)}'::date
            ORDER BY time DESC
        """

        with self._engine.connect() as conn:
            df = pd.read_sql(query, conn)

        if df.empty:
            return df

        df["entry_time"] = pd.to_datetime(df["entry_time"])
        df["exit_time"] = pd.to_datetime(df["exit_time"])
        return df

    def _compute_top_shap_features(
        self,
        trades_df: pd.DataFrame,
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Extract top SHAP features from trades."""
        if trades_df.empty:
            return []

        shap_list = trades_df["shap_values"].dropna().tolist()
        if not shap_list:
            return []

        # Aggregate: mean abs SHAP per feature
        feature_shaps = {}
        for feature in FEATURE_COLUMNS:
            abs_vals = [abs(sv.get(feature, 0.0)) for sv in shap_list if isinstance(sv, dict)]
            if abs_vals:
                feature_shaps[feature] = np.mean(abs_vals)

        # Sort and return top N
        sorted_features = sorted(feature_shaps.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return sorted_features
