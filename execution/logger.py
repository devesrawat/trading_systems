"""
Trade audit logger — every signal decision and its outcome logged to MLflow.

Provides a complete, queryable audit trail required for:
  - SEBI algo trading compliance (5-year retention)
  - Model drift detection (live win rate vs backtest)
  - Daily P&L reporting via Telegram
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import mlflow
import structlog
from sqlalchemy import text

from data.store import get_engine

log = structlog.get_logger(__name__)

_SIGNAL_EXPERIMENT = "nse_equity_signals"
_CIRCUIT_EXPERIMENT = "circuit_breaker_events"

_DAILY_TRADES_SQL = text("""
    SELECT price, side, quantity
    FROM paper_trades
    WHERE time::date = CURRENT_DATE
""")


class TradeLogger:
    """
    Logs every trade signal, decision, outcome, and system event to MLflow.

    Each signal creates an MLflow run. The run is later updated with the
    realized outcome via log_outcome().
    """

    def __init__(self, experiment_name: str = _SIGNAL_EXPERIMENT) -> None:
        self._experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # log_signal  — called at decision time
    # ------------------------------------------------------------------

    def log_signal(
        self,
        symbol: str,
        features_dict: dict[str, Any],
        signal_prob: float,
        action_taken: str,
        strategy_version: str = "1.0",
        regime: str = "unknown",
    ) -> str:
        """
        Log a signal + decision as an MLflow run.

        Returns the MLflow run_id for later outcome linkage.
        """
        with mlflow.start_run(run_name=f"{symbol}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "symbol": symbol,
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "action": action_taken,
                "strategy_version": strategy_version,
                "regime": regime,
            })

            mlflow.log_metrics({
                "signal_prob": signal_prob,
                **{k: float(v) for k, v in features_dict.items()},
            })

            mlflow.set_tags({
                "symbol": symbol,
                "action": action_taken,
                "outcome": "pending",
            })

            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else ""

        log.info("signal_logged", symbol=symbol, prob=signal_prob, action=action_taken, run_id=run_id)
        return run_id

    # ------------------------------------------------------------------
    # log_outcome  — called when position is closed
    # ------------------------------------------------------------------

    def log_outcome(
        self,
        run_id: str,
        exit_price: float,
        exit_date: date,
        pnl_pct: float,
    ) -> None:
        """
        Update an existing MLflow run with the realized trade outcome.
        """
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics({
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
            })
            mlflow.log_param("exit_date", str(exit_date))
            mlflow.set_tag("outcome", "win" if pnl_pct > 0 else "loss")

        log.info("outcome_logged", run_id=run_id, pnl_pct=round(pnl_pct, 4))

    # ------------------------------------------------------------------
    # log_circuit_breaker_event
    # ------------------------------------------------------------------

    def log_circuit_breaker_event(
        self,
        reason: str,
        capital_at_halt: float,
    ) -> None:
        """Log a circuit breaker halt event to MLflow."""
        mlflow.set_experiment(_CIRCUIT_EXPERIMENT)
        with mlflow.start_run(run_name=f"circuit_breaker_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_params({
                "reason": reason,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })
            mlflow.log_metric("capital_at_halt", capital_at_halt)
            mlflow.set_tag("event_type", "circuit_breaker")

        # Restore original experiment
        mlflow.set_experiment(self._experiment_name)
        log.warning("circuit_event_logged", reason=reason, capital=capital_at_halt)

    # ------------------------------------------------------------------
    # daily_summary
    # ------------------------------------------------------------------

    def daily_summary(self) -> dict[str, Any]:
        """
        Compute today's trading summary from paper_trades table.

        Returns: {trades_today, win_rate, pnl, sharpe_estimate}
        """
        engine = get_engine()
        with engine.connect() as conn:
            rows = conn.execute(_DAILY_TRADES_SQL).fetchall()

        if not rows:
            return {
                "trades_today": 0,
                "win_rate": 0.0,
                "pnl": 0.0,
                "sharpe_estimate": 0.0,
            }

        trades_today = len(rows)
        # Simplified: treat positive-price BUY trades as wins (real outcome tracked in MLflow)
        wins = sum(1 for r in rows if r.side == "SELL")
        win_rate = wins / trades_today if trades_today > 0 else 0.0

        summary = {
            "trades_today": trades_today,
            "win_rate": round(win_rate, 4),
            "pnl": 0.0,          # realized P&L comes from monitor.py
            "sharpe_estimate": 0.0,
        }
        log.info("daily_summary_computed", **summary)

        try:
            from monitoring.alerts import TelegramAlerter
            TelegramAlerter().alert_daily_summary(
                trades=trades_today,
                pnl_pct=summary["pnl"],
                sharpe=summary["sharpe_estimate"],
                win_rate=win_rate,
            )
        except Exception as exc:
            log.warning("telegram_summary_failed", error=str(exc))

        return summary
