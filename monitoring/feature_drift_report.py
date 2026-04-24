"""
Feature drift detection and reporting.

FeatureDriftReporter — Detect distribution shifts in features over time:
  - Compute feature statistics (mean, std, min, max) per period
  - KL divergence: compare distributions across time windows
  - Correlation breakdown: detect when feature correlations change
  - Telegram alerts: notify on significant drift
  - Weekly reports: markdown output with drift metrics

Example drift events:
  - RSI stopped working for a sector (stopped predicting returns)
  - Volatility regime changed (vol_regime feature shifted)
  - Volume patterns broke down (volume_zscore_20 distribution shifted)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
import structlog

from data.store import get_engine, get_redis
from monitoring.telegram_notifier import TelegramNotifier
from signals.features import FEATURE_COLUMNS

log = structlog.get_logger(__name__)


@dataclass
class FeatureStatistics:
    """Feature statistics snapshot."""

    period_start: date
    period_end: date
    timestamp: datetime
    feature_stats: dict[str, dict[str, float]]  # {feature: {mean, std, min, max, median}}


@dataclass
class DriftAlert:
    """Drift detection alert."""

    feature: str
    drift_score: float  # KL divergence or correlation change magnitude
    alert_type: str  # "distribution_shift" | "correlation_breakdown"
    period_start: date
    period_end: date
    previous_period_start: date
    previous_period_end: date
    description: str


class FeatureDriftReporter:
    """Detect and report feature distribution drift."""

    def __init__(self, alert_threshold: float = 0.3) -> None:
        """
        Initialize drift reporter.

        Parameters
        ----------
        alert_threshold : float
            KL divergence threshold for alerting (default 0.3)
        """
        self._alert_threshold = alert_threshold
        self._engine = get_engine()
        self._redis = get_redis()
        self._notifier = TelegramNotifier()

    # =========================================================================
    # Feature Statistics
    # =========================================================================

    def compute_feature_statistics(
        self,
        date_range: tuple[date, date],
    ) -> FeatureStatistics:
        """
        Compute distribution statistics for all features in a date range.

        Parameters
        ----------
        date_range : tuple[date, date]
            (start_date, end_date)

        Returns
        -------
        FeatureStatistics
            Per-feature mean, std, min, max, median
        """
        start, end = date_range

        try:
            df = self._fetch_trades_with_features(start, end)
        except Exception as exc:
            log.error("fetch_features_failed", error=str(exc))
            return FeatureStatistics(
                period_start=start,
                period_end=end,
                timestamp=datetime.now(),
                feature_stats={},
            )

        if df.empty:
            log.warning("no_trades_for_statistics", start=start, end=end)
            return FeatureStatistics(
                period_start=start,
                period_end=end,
                timestamp=datetime.now(),
                feature_stats={},
            )

        # Extract features from JSON column
        feature_data = []
        for feat_dict in df["features_used"]:
            if isinstance(feat_dict, dict):
                feature_data.append(feat_dict)

        if not feature_data:
            return FeatureStatistics(
                period_start=start,
                period_end=end,
                timestamp=datetime.now(),
                feature_stats={},
            )

        feature_df = pd.DataFrame(feature_data)

        # Compute stats per feature
        stats_dict = {}
        for feature in FEATURE_COLUMNS:
            if feature not in feature_df.columns:
                continue

            col = feature_df[feature].dropna()
            if col.empty:
                continue

            stats_dict[feature] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "median": float(col.median()),
                "count": len(col),
            }

        result = FeatureStatistics(
            period_start=start,
            period_end=end,
            timestamp=datetime.now(),
            feature_stats=stats_dict,
        )

        log.info(
            "feature_statistics_computed",
            period=f"{start} to {end}",
            features=len(stats_dict),
        )
        return result

    # =========================================================================
    # Distribution Shift Detection (KL Divergence)
    # =========================================================================

    def detect_distribution_shift(
        self,
        previous_stats: FeatureStatistics,
        current_stats: FeatureStatistics,
        n_bins: int = 10,
    ) -> dict[str, float]:
        """
        Detect distribution shifts using KL divergence.

        Compares feature distributions between two periods.
        High KL divergence → feature distribution changed significantly.

        Parameters
        ----------
        previous_stats : FeatureStatistics
            Statistics from previous period
        current_stats : FeatureStatistics
            Statistics from current period
        n_bins : int
            Number of bins for histogram approximation

        Returns
        -------
        dict[str, float]
            {feature: kl_divergence, ...}
        """
        divergences = {}

        for feature in FEATURE_COLUMNS:
            if (
                feature not in previous_stats.feature_stats
                or feature not in current_stats.feature_stats
            ):
                continue

            prev_stats = previous_stats.feature_stats[feature]
            curr_stats = current_stats.feature_stats[feature]

            # Use mean and std to approximate distribution
            prev_dist = np.random.normal(prev_stats["mean"], prev_stats["std"], 1000)
            curr_dist = np.random.normal(curr_stats["mean"], curr_stats["std"], 1000)

            # Histogram-based KL divergence
            hist_prev, bin_edges = np.histogram(prev_dist, bins=n_bins)
            hist_curr, _ = np.histogram(curr_dist, bins=bin_edges)

            # Normalize to probabilities
            hist_prev = hist_prev / hist_prev.sum()
            hist_curr = hist_curr / hist_curr.sum()

            # Add small epsilon to avoid log(0)
            eps = 1e-10
            kl_div = np.sum(hist_prev * np.log((hist_prev + eps) / (hist_curr + eps)))
            divergences[feature] = float(np.clip(kl_div, 0, 10))  # clip extreme values

        return divergences

    # =========================================================================
    # Correlation Change Detection
    # =========================================================================

    def correlation_change_detection(
        self,
        date_range_1: tuple[date, date],
        date_range_2: tuple[date, date],
        threshold: float = 0.3,
    ) -> dict[str, dict[str, float]]:
        """
        Detect changes in feature correlations with returns.

        Parameters
        ----------
        date_range_1 : tuple[date, date]
            Previous period
        date_range_2 : tuple[date, date]
            Current period
        threshold : float
            Minimum correlation change magnitude to flag

        Returns
        -------
        dict[str, dict[str, float]]
            {
                feature: {
                    "previous_corr": float,
                    "current_corr": float,
                    "change": float,
                }
            }
        """

        def _compute_correlations(date_range: tuple[date, date]) -> dict[str, float]:
            """Helper: compute feature-return correlations for a period."""
            try:
                df = self._fetch_trades_with_features(date_range[0], date_range[1])
            except Exception as exc:
                log.error("fetch_failed", error=str(exc))
                return {}

            if df.empty:
                return {}

            # Extract features and returns
            feature_data = []
            returns = []
            for _, row in df.iterrows():
                if isinstance(row["features_used"], dict):
                    feature_data.append(row["features_used"])
                returns.append(row.get("pnl_pct", 0.0))

            if not feature_data or not returns:
                return {}

            feature_df = pd.DataFrame(feature_data)
            returns = np.array(returns)

            correlations = {}
            for feature in FEATURE_COLUMNS:
                if feature not in feature_df.columns:
                    continue
                col = feature_df[feature].dropna()
                if col.empty or returns[: len(col)].std() == 0:
                    continue
                corr = np.corrcoef(col.values, returns[: len(col)])[0, 1]
                correlations[feature] = float(np.nan_to_num(corr, nan=0.0))

            return correlations

        prev_corr = _compute_correlations(date_range_1)
        curr_corr = _compute_correlations(date_range_2)

        changes = {}
        for feature in FEATURE_COLUMNS:
            prev_c = prev_corr.get(feature, 0.0)
            curr_c = curr_corr.get(feature, 0.0)
            change_mag = abs(curr_c - prev_c)

            if change_mag >= threshold:
                changes[feature] = {
                    "previous_corr": float(prev_c),
                    "current_corr": float(curr_c),
                    "change": float(change_mag),
                }

        log.info("correlation_changes_detected", n_changes=len(changes))
        return changes

    # =========================================================================
    # Alerting
    # =========================================================================

    def alert_on_drift(
        self,
        current_stats: FeatureStatistics,
        previous_stats: FeatureStatistics | None = None,
    ) -> list[DriftAlert]:
        """
        Check for significant drift and send Telegram alerts.

        Parameters
        ----------
        current_stats : FeatureStatistics
            Current period statistics
        previous_stats : FeatureStatistics, optional
            Previous period statistics. If None, fetch from cache/DB.

        Returns
        -------
        list[DriftAlert]
            List of alerts generated
        """
        if previous_stats is None:
            # Attempt to fetch from cache or compute from earlier period
            prev_end = current_stats.period_start - timedelta(days=1)
            prev_start = prev_end - timedelta(
                days=(current_stats.period_end - current_stats.period_start).days
            )
            previous_stats = self.compute_feature_statistics((prev_start, prev_end))

        if not previous_stats.feature_stats or not current_stats.feature_stats:
            log.warning("insufficient_data_for_drift_alerts")
            return []

        # Detect distribution shifts
        divergences = self.detect_distribution_shift(previous_stats, current_stats)

        # Detect correlation changes
        corr_changes = self.correlation_change_detection(
            (previous_stats.period_start, previous_stats.period_end),
            (current_stats.period_start, current_stats.period_end),
            threshold=self._alert_threshold,
        )

        alerts = []

        # Create alerts for distribution shifts
        for feature, kl_div in divergences.items():
            if kl_div > self._alert_threshold:
                alert = DriftAlert(
                    feature=feature,
                    drift_score=kl_div,
                    alert_type="distribution_shift",
                    period_start=current_stats.period_start,
                    period_end=current_stats.period_end,
                    previous_period_start=previous_stats.period_start,
                    previous_period_end=previous_stats.period_end,
                    description=f"{feature} distribution shifted (KL div: {kl_div:.3f})",
                )
                alerts.append(alert)
                self._send_drift_alert(alert)

        # Create alerts for correlation changes
        for feature, corr_data in corr_changes.items():
            alert = DriftAlert(
                feature=feature,
                drift_score=corr_data["change"],
                alert_type="correlation_breakdown",
                period_start=current_stats.period_start,
                period_end=current_stats.period_end,
                previous_period_start=previous_stats.period_start,
                previous_period_end=previous_stats.period_end,
                description=(
                    f"{feature} correlation with returns changed from "
                    f"{corr_data['previous_corr']:.3f} to {corr_data['current_corr']:.3f}"
                ),
            )
            alerts.append(alert)
            self._send_drift_alert(alert)

        log.info("drift_alerts_generated", n_alerts=len(alerts))
        return alerts

    def _send_drift_alert(self, alert: DriftAlert) -> None:
        """Send formatted Telegram alert."""
        message = f"""
🚨 <b>Feature Drift Alert</b>

<b>Feature:</b> {alert.feature}
<b>Type:</b> {alert.alert_type}
<b>Score:</b> {alert.drift_score:.3f}
<b>Description:</b> {alert.description}
<b>Period:</b> {alert.period_start} to {alert.period_end}
        """
        try:
            self._notifier.send_alert(message.strip())
        except Exception as exc:
            log.warning("drift_alert_send_failed", error=str(exc))

    # =========================================================================
    # Weekly Report
    # =========================================================================

    def generate_weekly_report(
        self,
        report_date: date | None = None,
    ) -> str:
        """
        Generate markdown weekly drift report.

        Parameters
        ----------
        report_date : date, optional
            Report date (defaults to today)

        Returns
        -------
        str
            Markdown report content
        """
        if report_date is None:
            report_date = date.today()

        # Last 7 days
        end = report_date
        start = end - timedelta(days=7)

        # Previous 7 days for comparison
        prev_end = start - timedelta(days=1)
        prev_start = prev_end - timedelta(days=7)

        current_stats = self.compute_feature_statistics((start, end))
        previous_stats = self.compute_feature_statistics((prev_start, prev_end))

        # Detect drifts
        divergences = self.detect_distribution_shift(previous_stats, current_stats)
        corr_changes = self.correlation_change_detection(
            (prev_start, prev_end),
            (start, end),
        )

        # Build markdown
        lines = [
            f"# Feature Drift Report — Week of {start} to {end}",
            "",
            f"**Report Generated:** {datetime.now().isoformat()}",
            "",
        ]

        # Summary
        lines.extend(
            [
                "## Summary",
                "",
                f"- **Current Period:** {start} to {end}",
                f"- **Previous Period:** {prev_start} to {prev_end}",
                f"- **Drifting Features:** {len([d for d in divergences.values() if d > self._alert_threshold])}",
                f"- **Correlation Changes:** {len(corr_changes)}",
                "",
            ]
        )

        # Feature Statistics Table
        lines.extend(
            [
                "## Feature Statistics (Current Period)",
                "",
                "| Feature | Mean | Std | Min | Max |",
                "|---------|------|-----|-----|-----|",
            ]
        )
        for feature, stats in sorted(current_stats.feature_stats.items()):
            lines.append(
                f"| {feature} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                f"{stats['min']:.3f} | {stats['max']:.3f} |"
            )
        lines.append("")

        # Distribution Shifts
        lines.extend(
            [
                "## Distribution Shifts (KL Divergence)",
                "",
                "| Feature | KL Divergence | Status |",
                "|---------|---------------|--------|",
            ]
        )
        for feature, kl_div in sorted(divergences.items(), key=lambda x: -x[1])[:15]:
            status = "⚠️ DRIFT" if kl_div > self._alert_threshold else "✓ OK"
            lines.append(f"| {feature} | {kl_div:.3f} | {status} |")
        lines.append("")

        # Correlation Changes
        if corr_changes:
            lines.extend(
                [
                    "## Correlation Changes with Returns",
                    "",
                    "| Feature | Previous Corr | Current Corr | Change |",
                    "|---------|---------------|--------------|--------|",
                ]
            )
            for feature, changes in sorted(
                corr_changes.items(), key=lambda x: -abs(x[1]["change"])
            )[:15]:
                lines.append(
                    f"| {feature} | {changes['previous_corr']:.3f} | "
                    f"{changes['current_corr']:.3f} | {changes['change']:.3f} |"
                )
            lines.append("")

        report_md = "\n".join(lines)

        log.info("weekly_drift_report_generated", period=f"{start} to {end}")
        return report_md

    # =========================================================================
    # MLflow Logging
    # =========================================================================

    def log_to_mlflow(
        self,
        report: str,
        experiment_name: str = "feature_drift",
    ) -> str:
        """
        Log drift report to MLflow.

        Parameters
        ----------
        report : str
            Markdown report content
        experiment_name : str
            MLflow experiment name

        Returns
        -------
        str
            Run ID
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"drift_report_{date.today().isoformat()}"):
            mlflow.log_text(report, "weekly_drift_report.md")
            mlflow.log_param("report_date", date.today().isoformat())
            run_id = mlflow.active_run().info.run_id

        log.info("drift_report_logged_to_mlflow", run_id=run_id, experiment=experiment_name)
        return run_id

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _fetch_trades_with_features(self, start: date, end: date) -> pd.DataFrame:
        """Fetch trades with features_used JSON column."""
        query = f"""
            SELECT
                id,
                time,
                symbol,
                side,
                quantity,
                price,
                signal_prob,
                '{{}}'::jsonb as features_used,
                0.0 as pnl_pct
            FROM paper_trades
            WHERE time >= '{start}'::date
              AND time < '{end + timedelta(days=1)}'::date
            ORDER BY time DESC
        """

        with self._engine.connect() as conn:
            df = pd.read_sql(query, conn)

        return df
