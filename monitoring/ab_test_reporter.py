"""
A/B test reporter for weekly model comparison and promotion decisions.

Generates markdown reports with statistical tests and sends Telegram alerts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import structlog

from data.store import get_redis
from monitoring.telegram_notifier import TelegramNotifier
from orchestrator.ab_tester import ABTestOrchestrator

log = structlog.get_logger(__name__)


class ABTestReporter:
    """
    Generates weekly A/B test comparison reports.

    Performs statistical testing and formats output for Telegram.
    """

    def __init__(
        self,
        ab_tester: ABTestOrchestrator | None = None,
        telegram_notifier: TelegramNotifier | None = None,
    ) -> None:
        self._ab_tester = ab_tester or ABTestOrchestrator()
        self._telegram = telegram_notifier
        self._redis = get_redis()

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_weekly_report(self, lookback_days: int = 7) -> str:
        """
        Generate a markdown report comparing champion vs. challenger.

        Parameters
        ----------
        lookback_days : int
            Number of days to include (default 7 for weekly)

        Returns
        -------
        str
            Markdown-formatted report
        """
        comparison = self._ab_tester.compare_models(lookback_days=lookback_days)

        champion_stats = comparison.get("champion", {})
        challenger_stats = comparison.get("challenger", {})
        p_value = comparison.get("p_value")
        challenger_wins = comparison.get("challenger_wins", False)

        # Build markdown report
        report = self._format_report(champion_stats, challenger_stats, p_value, challenger_wins)

        log.info(
            "weekly_report_generated",
            lookback_days=lookback_days,
            challenger_wins=challenger_wins,
        )

        return report

    def _format_report(
        self,
        champion_stats: dict[str, float],
        challenger_stats: dict[str, float],
        p_value: float | None,
        challenger_wins: bool,
    ) -> str:
        """Format comparison into markdown."""
        report_lines = [
            "# 🎯 A/B Test Weekly Report",
            "",
            f"📅 **Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            "",
        ]

        # Metrics table
        report_lines.extend(
            [
                "## 📊 Performance Metrics",
                "",
                "| Metric | Champion | Challenger | Δ |",
                "|--------|----------|------------|---|",
            ]
        )

        metrics_to_show = [
            ("# Trades", "num_trades"),
            ("Total P&L", "total_pnl"),
            ("Avg P&L", "avg_pnl"),
            ("Win Rate", "win_rate"),
            ("Sharpe Ratio", "avg_sharpe"),
            ("Max DD", "max_dd"),
            ("Profit Factor", "profit_factor"),
        ]

        for label, key in metrics_to_show:
            champ_val = champion_stats.get(key, "—")
            chall_val = challenger_stats.get(key, "—")

            # Format value
            if isinstance(champ_val, float):
                champ_str = f"{champ_val:.4f}" if key != "num_trades" else str(int(champ_val))
            else:
                champ_str = str(champ_val)

            if isinstance(chall_val, float):
                chall_str = f"{chall_val:.4f}" if key != "num_trades" else str(int(chall_val))
            else:
                chall_str = str(chall_val)

            # Calculate delta
            try:
                if key == "num_trades":
                    delta_str = ""
                elif isinstance(champ_val, (int, float)) and isinstance(chall_val, (int, float)):
                    delta = chall_val - champ_val
                    sign = "📈" if delta > 0 else "📉" if delta < 0 else "→"
                    delta_str = f"{sign} {delta:+.4f}"
                else:
                    delta_str = ""
            except:
                delta_str = ""

            report_lines.append(f"| {label} | {champ_str} | {chall_str} | {delta_str} |")

        report_lines.extend(["", ""])

        # Statistical test
        if p_value is not None:
            report_lines.extend(
                [
                    "## 📈 Statistical Test (t-test on Sharpe Ratio)",
                    "",
                    f"- **p-value**: {p_value:.6f}",
                    "- **Significance Level**: α = 0.05",
                    f"- **Test Result**: {'✅ PASS (significant difference)' if p_value < 0.05 else '❌ FAIL (no significant difference)'}",
                    "",
                ]
            )

        # Recommendation
        report_lines.extend(["## 🎯 Recommendation", ""])

        if challenger_wins:
            report_lines.extend(
                [
                    "**✅ PROMOTE CHALLENGER**",
                    "",
                    f"- Challenger Sharpe ({challenger_stats.get('avg_sharpe', 'N/A'):.4f}) > "
                    f"Champion ({champion_stats.get('avg_sharpe', 'N/A'):.4f})",
                    f"- Statistical significance: p = {p_value:.6f} < 0.05",
                    f"- Win Rate: {(challenger_stats.get('win_rate', 0) * 100):.1f}%",
                    "",
                ]
            )
        else:
            reason = "- Insufficient data for comparison"
            if p_value is not None and p_value >= 0.05:
                reason = f"- No statistically significant difference (p = {p_value:.6f})"
            elif p_value is not None:
                reason = f"- Challenger did not outperform (p = {p_value:.6f})"

            report_lines.extend(
                [
                    "**❌ KEEP CHAMPION**",
                    "",
                    reason,
                    "",
                ]
            )

        # Footer
        report_lines.extend(
            [
                "---",
                "*This report is auto-generated. Review before making production changes.*",
            ]
        )

        report = "\n".join(report_lines)
        return report

    # ------------------------------------------------------------------
    # Telegram alerting
    # ------------------------------------------------------------------

    def send_weekly_report_to_telegram(self, lookback_days: int = 7) -> bool:
        """
        Generate and send weekly report to Telegram.

        Parameters
        ----------
        lookback_days : int
            Days to include in report

        Returns
        -------
        bool
            True if message sent successfully
        """
        if not self._telegram:
            log.warning("telegram_notifier_not_configured")
            return False

        report = self.generate_weekly_report(lookback_days=lookback_days)

        try:
            self._telegram.send_message(report)
            log.info("weekly_report_sent_to_telegram")
            return True
        except Exception as e:
            log.error("telegram_send_failed", error=str(e))
            return False

    # ------------------------------------------------------------------
    # Statistical testing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def statistical_significance_test(
        challenger_results: list[dict[str, Any]],
        champion_results: list[dict[str, Any]],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """
        Run comprehensive statistical tests between two result sets.

        Parameters
        ----------
        challenger_results : list[dict]
            Challenger model results
        champion_results : list[dict]
            Champion model results
        alpha : float
            Significance level (default 0.05)

        Returns
        -------
        dict
            {
                "sharpe_p_value": float,
                "win_rate_p_value": float,
                "pnl_p_value": float,
                "challenger_wins": bool,
                "details": str
            }
        """
        if not challenger_results or not champion_results:
            return {
                "sharpe_p_value": None,
                "win_rate_p_value": None,
                "pnl_p_value": None,
                "challenger_wins": False,
                "details": "Insufficient data",
            }

        # Import stats here to avoid top-level import issues
        from scipy import stats as sp_stats

        # Sharpe ratio test
        chall_sharpes = np.array([r["sharpe"] for r in challenger_results])
        champ_sharpes = np.array([r["sharpe"] for r in champion_results])
        sharpe_t, sharpe_p = sp_stats.ttest_ind(chall_sharpes, champ_sharpes)

        # Win rate test (binomial)
        chall_wins = np.array([r["win"] for r in challenger_results])
        champ_wins = np.array([r["win"] for r in champion_results])
        chall_win_rate = np.mean(chall_wins)
        champ_win_rate = np.mean(champ_wins)

        # Use chi-square for win rate comparison
        contingency = [
            [np.sum(chall_wins), len(chall_wins) - np.sum(chall_wins)],
            [np.sum(champ_wins), len(champ_wins) - np.sum(champ_wins)],
        ]
        chi2, win_rate_p, _, _ = sp_stats.chi2_contingency(contingency)

        # PnL test
        chall_pnls = np.array([r["pnl"] for r in challenger_results])
        champ_pnls = np.array([r["pnl"] for r in champion_results])
        pnl_t, pnl_p = sp_stats.ttest_ind(chall_pnls, champ_pnls)

        # Determine if challenger wins
        challenger_wins = (
            sharpe_p < alpha
            and np.mean(chall_sharpes) > np.mean(champ_sharpes)
            and chall_win_rate > champ_win_rate
        )

        details = (
            f"Sharpe: p={sharpe_p:.4f} (chall={np.mean(chall_sharpes):.4f}, "
            f"champ={np.mean(champ_sharpes):.4f}), "
            f"Win Rate: p={win_rate_p:.4f} (chall={chall_win_rate:.2%}, "
            f"champ={champ_win_rate:.2%}), "
            f"PnL: p={pnl_p:.4f} (chall={np.mean(chall_pnls):.2f}, "
            f"champ={np.mean(champ_pnls):.2f})"
        )

        return {
            "sharpe_p_value": float(sharpe_p),
            "win_rate_p_value": float(win_rate_p),
            "pnl_p_value": float(pnl_p),
            "challenger_wins": challenger_wins,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Feature importance (future enhancement)
    # ------------------------------------------------------------------

    def compare_feature_importance(self) -> str:
        """
        Compare feature importance between champion and challenger models.

        Returns
        -------
        str
            Markdown table comparing top features
        """
        # This would require loading SHAP values from both models
        # For now, return a placeholder
        return "Feature importance comparison coming soon..."
