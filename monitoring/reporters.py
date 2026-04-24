"""
Phase 6: Daily, weekly, and monthly reporting.

Reporters aggregate trading metrics, portfolio state, and system health
to produce human-readable summaries for review and dashboarding.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: datetime
    total_value: float
    cash: float
    invested: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holdings: dict[str, dict[str, Any]]  # {symbol: {qty, avg_price, current_price, pnl_pct}}
    sector_exposure: dict[str, float]  # {sector: pct_of_portfolio}
    correlation_matrix: dict[str, dict[str, float]]
    liquidity_score: float  # 0-1


@dataclass
class SystemHealthSnapshot:
    """System health at a point in time."""

    timestamp: datetime
    uptime_hours: float
    api_latency_ms: float
    cache_hit_rate: float  # 0-1
    error_rate: float  # 0-1
    broker_connection_status: str  # "connected", "degraded", "offline"
    db_connection_status: str


@dataclass
class DailyMetrics:
    """Daily trading metrics."""

    date: datetime
    scans_completed: int
    signals_generated: int
    signals_executed: int
    trades_entered: int
    trades_closed: int
    total_pnl: float
    total_pnl_pct: float
    win_count: int
    loss_count: int
    win_rate: float  # 0-1
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_intraday_dd: float
    daily_dd: float
    portfolio_snapshot: PortfolioSnapshot | None = None
    system_health: SystemHealthSnapshot | None = None


@dataclass
class WeeklyMetrics:
    """Weekly aggregated metrics."""

    week_start: datetime
    week_end: datetime
    days_traded: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    strategy_performance: dict[str, dict[str, float]]  # {strategy: {pnl, win_rate, trades}}
    best_performer: str
    worst_performer: str
    best_trade: float
    worst_trade: float
    average_trade: float
    portfolio_snapshot: PortfolioSnapshot | None = None


@dataclass
class MonthlyMetrics:
    """Monthly aggregated metrics."""

    month_start: datetime
    month_end: datetime
    days_traded: int
    total_pnl: float
    total_pnl_pct: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    strategy_rankings: list[tuple[str, float]]  # [(strategy, pnl)]
    multibagger_count: int
    multibagger_candidates: list[str]
    portfolio_snapshot: PortfolioSnapshot | None = None


class DailyReport:
    """Produces a daily trading report."""

    @staticmethod
    def generate(metrics: DailyMetrics) -> str:
        """
        Generate a formatted daily report.

        Parameters
        ----------
        metrics : DailyMetrics
            Daily trading metrics

        Returns
        -------
        str
            Formatted report text
        """
        lines = [
            f"📊 DAILY REPORT — {metrics.date.strftime('%Y-%m-%d')}",
            "",
            "Scans & Signals:",
            f"  Scans: {metrics.scans_completed}",
            f"  Signals: {metrics.signals_generated}",
            f"  Executed: {metrics.signals_executed}",
            "",
            "Execution:",
            f"  Trades entered: {metrics.trades_entered}",
            f"  Trades closed: {metrics.trades_closed}",
            "",
            "Performance:",
            f"  P&L: {metrics.total_pnl:+.2f} ({metrics.total_pnl_pct:+.2%})",
            f"  Win rate: {metrics.win_rate:.1%} ({metrics.win_count}W / {metrics.loss_count}L)",
            f"  Profit factor: {metrics.profit_factor:.2f}",
            f"  Avg win/loss: {metrics.avg_win:+.2f} / {metrics.avg_loss:+.2f}",
            "",
            "Risk:",
            f"  Daily drawdown: {metrics.daily_dd:.2%}",
            f"  Max intraday DD: {metrics.max_intraday_dd:.2%}",
        ]

        if metrics.portfolio_snapshot:
            ps = metrics.portfolio_snapshot
            lines.extend(
                [
                    "",
                    "Portfolio:",
                    f"  Total: ₹{ps.total_value:,.0f}",
                    f"  Cash: ₹{ps.cash:,.0f}",
                    f"  Invested: ₹{ps.invested:,.0f}",
                    f"  Unrealized P&L: {ps.unrealized_pnl_pct:+.2%}",
                ]
            )

        if metrics.system_health:
            sh = metrics.system_health
            lines.extend(
                [
                    "",
                    "System:",
                    f"  Uptime: {sh.uptime_hours:.1f}h",
                    f"  API latency: {sh.api_latency_ms:.0f}ms",
                    f"  Cache hit rate: {sh.cache_hit_rate:.1%}",
                    f"  Broker: {sh.broker_connection_status}",
                ]
            )

        return "\n".join(lines)


class WeeklyReport:
    """Produces a weekly trading report."""

    @staticmethod
    def generate(metrics: WeeklyMetrics) -> str:
        """
        Generate a formatted weekly report.

        Parameters
        ----------
        metrics : WeeklyMetrics
            Weekly trading metrics

        Returns
        -------
        str
            Formatted report text
        """
        lines = [
            f"📈 WEEKLY REPORT — {metrics.week_start.strftime('%Y-%m-%d')} to {metrics.week_end.strftime('%Y-%m-%d')}",
            "",
            "Summary:",
            f"  Days traded: {metrics.days_traded}",
            f"  P&L: {metrics.total_pnl:+.2f} ({metrics.total_pnl_pct:+.2%})",
            f"  Win rate: {metrics.win_rate:.1%}",
            f"  Profit factor: {metrics.profit_factor:.2f}",
            f"  Sharpe: {metrics.sharpe_ratio:.2f}",
            f"  Max DD: {metrics.max_drawdown:.2%}",
            "",
            "Trade Statistics:",
            f"  Best: {metrics.best_trade:+.2f}",
            f"  Worst: {metrics.worst_trade:+.2f}",
            f"  Average: {metrics.average_trade:+.2f}",
            "",
            "Strategy Performance:",
        ]

        # Add top performers
        for strategy, perf in metrics.strategy_performance.items():
            pnl = perf.get("pnl", 0)
            wr = perf.get("win_rate", 0)
            trades = perf.get("trades", 0)
            lines.append(f"  {strategy}: {pnl:+.2f} ({wr:.1%} WR, {trades} trades)")

        lines.extend(
            [
                "",
                f"Best performer: {metrics.best_performer}",
                f"Worst performer: {metrics.worst_performer}",
            ]
        )

        if metrics.portfolio_snapshot:
            ps = metrics.portfolio_snapshot
            lines.extend(
                [
                    "",
                    "Portfolio:",
                    f"  Total: ₹{ps.total_value:,.0f}",
                    f"  Unrealized P&L: {ps.unrealized_pnl_pct:+.2%}",
                ]
            )

        return "\n".join(lines)


class MonthlyReport:
    """Produces a monthly trading report."""

    @staticmethod
    def generate(metrics: MonthlyMetrics) -> str:
        """
        Generate a formatted monthly report.

        Parameters
        ----------
        metrics : MonthlyMetrics
            Monthly trading metrics

        Returns
        -------
        str
            Formatted report text
        """
        lines = [
            f"📅 MONTHLY REPORT — {metrics.month_start.strftime('%B %Y')}",
            "",
            "Summary:",
            f"  Days traded: {metrics.days_traded}",
            f"  P&L: {metrics.total_pnl:+.2f} ({metrics.total_pnl_pct:+.2%})",
            f"  Win rate: {metrics.win_rate:.1%}",
            f"  Profit factor: {metrics.profit_factor:.2f}",
            f"  Sharpe: {metrics.sharpe_ratio:.2f}",
            f"  Max DD: {metrics.max_drawdown:.2%}",
            f"  Calmar: {metrics.calmar_ratio:.2f}",
            "",
            "Strategy Rankings:",
        ]

        for i, (strategy, pnl) in enumerate(metrics.strategy_rankings, 1):
            lines.append(f"  {i}. {strategy}: {pnl:+.2f}")

        lines.extend(
            [
                "",
                f"Multibaggers: {metrics.multibagger_count}",
            ]
        )

        if metrics.multibagger_candidates:
            lines.append("  Candidates:")
            for candidate in metrics.multibagger_candidates:
                lines.append(f"    - {candidate}")

        if metrics.portfolio_snapshot:
            ps = metrics.portfolio_snapshot
            lines.extend(
                [
                    "",
                    "Portfolio:",
                    f"  Total: ₹{ps.total_value:,.0f}",
                    f"  Invested: ₹{ps.invested:,.0f}",
                ]
            )

        return "\n".join(lines)


class PortfolioStatusReport:
    """Real-time portfolio status snapshot."""

    @staticmethod
    def generate(snapshot: PortfolioSnapshot) -> str:
        """
        Generate a formatted portfolio status report.

        Parameters
        ----------
        snapshot : PortfolioSnapshot
            Portfolio snapshot

        Returns
        -------
        str
            Formatted report text
        """
        lines = [
            f"💼 PORTFOLIO STATUS — {snapshot.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Summary:",
            f"  Total: ₹{snapshot.total_value:,.0f}",
            f"  Cash: ₹{snapshot.cash:,.0f}",
            f"  Invested: ₹{snapshot.invested:,.0f}",
            f"  Unrealized P&L: {snapshot.unrealized_pnl_pct:+.2%}",
            "",
            "Holdings:",
        ]

        if snapshot.holdings:
            for symbol, details in snapshot.holdings.items():
                qty = details.get("qty", 0)
                avg = details.get("avg_price", 0)
                current = details.get("current_price", 0)
                pnl_pct = details.get("pnl_pct", 0)
                lines.append(f"  {symbol}: {qty} @ ₹{avg:.2f} → ₹{current:.2f} ({pnl_pct:+.2%})")
        else:
            lines.append("  (no open positions)")

        if snapshot.sector_exposure:
            lines.extend(["", "Sector Exposure:"])
            for sector, pct in sorted(
                snapshot.sector_exposure.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {sector}: {pct:.1%}")

        lines.append(f"\nLiquidity score: {snapshot.liquidity_score:.2f}")

        return "\n".join(lines)


class SystemHealthReport:
    """System health status report."""

    @staticmethod
    def generate(snapshot: SystemHealthSnapshot) -> str:
        """
        Generate a formatted system health report.

        Parameters
        ----------
        snapshot : SystemHealthSnapshot
            System health snapshot

        Returns
        -------
        str
            Formatted report text
        """
        broker_emoji = "✅" if snapshot.broker_connection_status == "connected" else "⚠️"
        db_emoji = "✅" if snapshot.db_connection_status == "connected" else "⚠️"

        lines = [
            f"🔧 SYSTEM HEALTH — {snapshot.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "Status:",
            f"  {broker_emoji} Broker: {snapshot.broker_connection_status}",
            f"  {db_emoji} Database: {snapshot.db_connection_status}",
            "",
            "Performance:",
            f"  Uptime: {snapshot.uptime_hours:.1f} hours",
            f"  API latency: {snapshot.api_latency_ms:.0f}ms",
            f"  Cache hit rate: {snapshot.cache_hit_rate:.1%}",
            f"  Error rate: {snapshot.error_rate:.2%}",
        ]

        return "\n".join(lines)
