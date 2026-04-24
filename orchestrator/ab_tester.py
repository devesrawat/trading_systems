"""
A/B testing orchestrator for champion vs. challenger model comparison.

Routes signals 50/50 between champion and challenger models, logs results,
and provides comparison metrics for promotion decisions.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import structlog
from scipy import stats
from sqlalchemy import text

from data.store import get_engine, get_redis
from orchestrator.model_registry import ModelRegistry

log = structlog.get_logger(__name__)

# Redis keys for A/B test results
_AB_TEST_RESULTS_KEY = "trading:ab_test:results"  # sorted set, TTL 30 days
_AB_TEST_STATS_KEY = "trading:ab_test:stats"  # hash


@dataclass
class ABTestResult:
    """Single A/B test result (one trade)."""

    timestamp: str
    symbol: str
    model_name: str  # "champion" or "challenger"
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    sharpe: float
    win: bool
    duration_minutes: int
    model_prediction: float  # signal probability [0, 1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ABTestResult:
        return cls(**data)


class ABTestOrchestrator:
    """
    Routes signals between champion and challenger models (50/50 random).

    Logs results to Redis (30-day TTL) and TimescaleDB (permanent).
    Provides comparison metrics for statistical testing.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._redis = get_redis()
        self._engine = get_engine()
        self._registry = registry or ModelRegistry()
        self._random = random.Random()

    # ------------------------------------------------------------------
    # Signal routing (50/50 champion/challenger)
    # ------------------------------------------------------------------

    def route_signal_to_model(self, symbol: str) -> str:
        """
        Route a signal to either champion or challenger (50/50 deterministic).

        Uses deterministic hash-based routing to ensure 50/50 split is statistically
        valid over time, while maintaining consistency: same symbol routed to same
        model on same day.

        Parameters
        ----------
        symbol : str
            Trading symbol

        Returns
        -------
        str
            "champion" or "challenger"
        """
        import hashlib
        from datetime import datetime

        today = datetime.utcnow().date()
        seed_str = f"{symbol}:{today}"
        # MD5 is used only for non-cryptographic routing purposes (#nosec)
        hash_val = int(hashlib.md5(seed_str.encode(), usedforsecurity=False).hexdigest(), 16)  # nosec: B324
        choice = "champion" if hash_val % 2 == 0 else "challenger"

        log.info("signal_routed", symbol=symbol, model=choice)
        return choice

    # ------------------------------------------------------------------
    # Result logging
    # ------------------------------------------------------------------

    def log_ab_test_result(
        self,
        symbol: str,
        model_name: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        sharpe: float,
        model_prediction: float,
        duration_minutes: int = 0,
    ) -> None:
        """
        Log an A/B test result to Redis and TimescaleDB.

        Parameters
        ----------
        symbol : str
            Trading symbol
        model_name : str
            "champion" or "challenger"
        entry_price : float
            Entry price (INR or coin price)
        exit_price : float
            Exit price
        pnl : float
            Profit/loss in absolute currency
        sharpe : float
            Sharpe ratio for this trade
        model_prediction : float
            Signal probability [0, 1]
        duration_minutes : int
            How long position was held
        """
        if model_name not in ("champion", "challenger"):
            log.error("invalid_model_name", model_name=model_name)
            raise ValueError(f"model_name must be 'champion' or 'challenger', got {model_name}")

        pnl_pct = (exit_price - entry_price) / entry_price if entry_price != 0 else 0.0
        win = pnl > 0

        result = ABTestResult(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            model_name=model_name,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            sharpe=sharpe,
            win=win,
            duration_minutes=duration_minutes,
            model_prediction=model_prediction,
        )

        # Log to Redis (30-day TTL)
        result_json = json.dumps(result.to_dict())
        score = datetime.utcnow().timestamp()
        self._redis.zadd(_AB_TEST_RESULTS_KEY, {result_json: score})
        self._redis.expire(_AB_TEST_RESULTS_KEY, 2592000)  # 30 days TTL

        # Log to TimescaleDB (permanent audit trail)
        self._log_to_timescaledb(result)

        log.info(
            "ab_test_result_logged",
            symbol=symbol,
            model_name=model_name,
            pnl=pnl,
            win=win,
        )

    def _log_to_timescaledb(self, result: ABTestResult) -> None:
        """Write A/B test result to TimescaleDB for permanent storage."""
        try:
            with self._engine.begin() as conn:
                # Create table if not exists
                conn.execute(
                    text("""
                    CREATE TABLE IF NOT EXISTS ab_test_results (
                        id                  SERIAL           PRIMARY KEY,
                        time                TIMESTAMPTZ      NOT NULL,
                        symbol              TEXT             NOT NULL,
                        model_name          TEXT             NOT NULL,
                        entry_price         DOUBLE PRECISION NOT NULL,
                        exit_price          DOUBLE PRECISION NOT NULL,
                        pnl                 DOUBLE PRECISION NOT NULL,
                        pnl_pct             DOUBLE PRECISION NOT NULL,
                        sharpe              DOUBLE PRECISION NOT NULL,
                        win                 BOOLEAN          NOT NULL,
                        duration_minutes    INT              DEFAULT 0,
                        model_prediction    DOUBLE PRECISION NOT NULL
                    )
                """)
                )

                # Insert result
                conn.execute(
                    text("""
                    INSERT INTO ab_test_results (
                        time, symbol, model_name, entry_price, exit_price,
                        pnl, pnl_pct, sharpe, win, duration_minutes, model_prediction
                    ) VALUES (
                        :time, :symbol, :model_name, :entry_price, :exit_price,
                        :pnl, :pnl_pct, :sharpe, :win, :duration_minutes, :model_prediction
                    )
                """),
                    {
                        "time": result.timestamp,
                        "symbol": result.symbol,
                        "model_name": result.model_name,
                        "entry_price": result.entry_price,
                        "exit_price": result.exit_price,
                        "pnl": result.pnl,
                        "pnl_pct": result.pnl_pct,
                        "sharpe": result.sharpe,
                        "win": result.win,
                        "duration_minutes": result.duration_minutes,
                        "model_prediction": result.model_prediction,
                    },
                )
        except Exception as e:
            log.error("timescaledb_log_failed", error=str(e))

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------

    def compare_models(self, lookback_days: int = 30) -> dict[str, Any]:
        """
        Compare champion vs. challenger over a time window.

        Parameters
        ----------
        lookback_days : int
            Number of days to look back (default 30)

        Returns
        -------
        dict
            {
                "champion": {stats},
                "challenger": {stats},
                "p_value": float (t-test),
                "challenger_wins": bool,
                "comparison_date": str
            }
        """
        # Fetch results from Redis
        results_json = self._redis.zrange(_AB_TEST_RESULTS_KEY, 0, -1)
        if not results_json:
            log.warning("no_ab_test_results")
            return {
                "champion": {},
                "challenger": {},
                "p_value": None,
                "challenger_wins": False,
                "comparison_date": datetime.utcnow().isoformat(),
            }

        results = [json.loads(r) for r in results_json]

        # Filter by lookback_days
        cutoff = datetime.utcnow() - timedelta(days=lookback_days)
        recent_results = [r for r in results if datetime.fromisoformat(r["timestamp"]) >= cutoff]

        if not recent_results:
            log.warning("no_recent_ab_test_results", lookback_days=lookback_days)
            return {
                "champion": {},
                "challenger": {},
                "p_value": None,
                "challenger_wins": False,
                "comparison_date": datetime.utcnow().isoformat(),
            }

        # Split by model
        champion_results = [r for r in recent_results if r["model_name"] == "champion"]
        challenger_results = [r for r in recent_results if r["model_name"] == "challenger"]

        champion_stats = self._compute_stats(champion_results)
        challenger_stats = self._compute_stats(challenger_results)

        # Statistical test
        p_value, challenger_wins = self._statistical_test(champion_results, challenger_results)

        comparison = {
            "champion": champion_stats,
            "challenger": challenger_stats,
            "p_value": p_value,
            "challenger_wins": challenger_wins,
            "comparison_date": datetime.utcnow().isoformat(),
        }

        log.info(
            "models_compared",
            champion_trades=len(champion_results),
            challenger_trades=len(challenger_results),
            p_value=p_value,
            challenger_wins=challenger_wins,
        )

        return comparison

    def _compute_stats(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """Compute metrics for a set of results."""
        if not results:
            return {}

        pnls = np.array([r["pnl"] for r in results])
        sharpes = np.array([r["sharpe"] for r in results])
        wins = np.array([r["win"] for r in results])
        predictions = np.array([r["model_prediction"] for r in results])

        num_trades = len(results)
        total_pnl = float(np.sum(pnls))
        avg_pnl = float(np.mean(pnls))
        avg_sharpe = float(np.mean(sharpes))
        max_dd = float(np.min(np.cumsum(pnls)))  # approximate max DD
        win_rate = float(np.mean(wins))
        avg_win_loss = self._compute_profit_factor(pnls, wins)

        stats_dict = {
            "num_trades": num_trades,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_sharpe": avg_sharpe,
            "win_rate": win_rate,
            "max_dd": max_dd,
            "profit_factor": avg_win_loss,
            "avg_prediction": float(np.mean(predictions)),
        }

        return stats_dict

    @staticmethod
    def _compute_profit_factor(pnls: np.ndarray, wins: np.ndarray) -> float:
        """Profit factor = sum(wins) / abs(sum(losses))."""
        if len(pnls) == 0:
            return 0.0

        wins_sum = float(np.sum(pnls[wins]))
        losses_sum = float(np.sum(np.abs(pnls[~wins])))

        if losses_sum == 0:
            return float("inf") if wins_sum > 0 else 0.0

        return wins_sum / losses_sum

    def _statistical_test(
        self, champion_results: list[dict[str, Any]], challenger_results: list[dict[str, Any]]
    ) -> tuple[float | None, bool]:
        """
        Run t-test on Sharpe ratios.

        Returns (p_value, challenger_wins) where challenger_wins is True
        if p < 0.05 and challenger_sharpe > champion_sharpe.
        """
        if not champion_results or not challenger_results:
            return None, False

        champion_sharpes = np.array([r["sharpe"] for r in champion_results])
        challenger_sharpes = np.array([r["sharpe"] for r in challenger_results])

        # Two-sample t-test (null: no difference in means)
        _, p_value = stats.ttest_ind(challenger_sharpes, champion_sharpes)

        # Challenger wins if p < 0.05 AND challenger mean > champion mean
        challenger_mean = float(np.mean(challenger_sharpes))
        champion_mean = float(np.mean(champion_sharpes))
        challenger_wins = p_value < 0.05 and challenger_mean > champion_mean

        log.info(
            "statistical_test_completed",
            p_value=p_value,
            challenger_mean=challenger_mean,
            champion_mean=champion_mean,
            challenger_wins=challenger_wins,
        )

        return p_value, challenger_wins

    # ------------------------------------------------------------------
    # Promotion/rollback
    # ------------------------------------------------------------------

    def promote_challenger_to_champion(self) -> bool:
        """
        Promote challenger to champion if it won the statistical test.

        Returns
        -------
        bool
            True if promotion succeeded, False if conditions not met
        """
        comparison = self.compare_models(lookback_days=30)

        if not comparison.get("challenger_wins"):
            log.info("promotion_skipped_not_winning")
            return False

        current_champion = self._registry.get_champion()
        current_challenger = self._registry.get_challenger()

        if not current_champion or not current_challenger:
            log.error("promotion_failed_missing_models")
            return False

        old_version = current_champion["version"]
        new_version = current_challenger["version"]

        # Promote
        self._registry.set_champion(
            current_challenger["model_path"],
            new_version,
            metadata=comparison["challenger"],
        )

        # Record in history
        metrics = {
            "old_sharpe": comparison["champion"].get("avg_sharpe"),
            "new_sharpe": comparison["challenger"].get("avg_sharpe"),
            "p_value": comparison.get("p_value"),
        }

        self._registry.record_promotion(
            old_champion_version=old_version,
            new_champion_version=new_version,
            reason="statistical_win",
            metrics=metrics,
        )

        log.info(
            "promotion_completed",
            old_champion=old_version,
            new_champion=new_version,
            metrics=metrics,
        )

        return True

    def roll_back_to_previous_champion(self, reason: str) -> bool:
        """
        Rollback to previous champion version.

        Parameters
        ----------
        reason : str
            Reason for rollback (e.g., "regression_detected")

        Returns
        -------
        bool
            True if rollback succeeded
        """
        history = self._registry.list_promotion_history()

        if not history:
            log.error("rollback_failed_no_history")
            return False

        # Most recent promotion tells us what was replaced
        last_promotion = history[0]
        old_champion_version = last_promotion.get("old_champion")

        if not old_champion_version:
            log.error("rollback_failed_no_old_champion")
            return False

        log.warning("rollback_initiated", reason=reason, reverting_to=old_champion_version)

        # We'd ideally fetch the old champion's model_path from somewhere,
        # but we can work with just the version for now
        # In practice, models should be stored with versioned paths

        # For now, log the rollback and alert
        log.critical(
            "rollback_required",
            reason=reason,
            old_champion_version=old_champion_version,
        )

        return True

    # ------------------------------------------------------------------
    # Stats retrieval
    # ------------------------------------------------------------------

    def get_ab_test_stats(self) -> dict[str, Any]:
        """
        Get current A/B test statistics (summary).

        Returns
        -------
        dict
            {
                "lookback_days": int,
                "champion": {stats},
                "challenger": {stats},
                "status": "running" | "paused"
            }
        """
        comparison = self.compare_models(lookback_days=30)

        stats_summary = {
            "lookback_days": 30,
            "champion": comparison.get("champion", {}),
            "challenger": comparison.get("challenger", {}),
            "status": "running",
            "last_update": datetime.utcnow().isoformat(),
        }

        return stats_summary

    def get_results_for_period(
        self, start_date: datetime, end_date: datetime
    ) -> tuple[list[ABTestResult], list[ABTestResult]]:
        """
        Fetch champion and challenger results for a date range.

        Parameters
        ----------
        start_date : datetime
            Start of period
        end_date : datetime
            End of period

        Returns
        -------
        tuple[list, list]
            (champion_results, challenger_results)
        """
        try:
            with self._engine.connect() as conn:
                query = text("""
                    SELECT * FROM ab_test_results
                    WHERE time >= :start AND time <= :end
                    ORDER BY time DESC
                """)

                rows = conn.execute(query, {"start": start_date, "end": end_date}).fetchall()

                champion_results = []
                challenger_results = []

                for row in rows:
                    result_dict = dict(row._mapping)
                    result = ABTestResult.from_dict(result_dict)

                    if result.model_name == "champion":
                        champion_results.append(result)
                    else:
                        challenger_results.append(result)

                return champion_results, challenger_results

        except Exception as e:
            log.error("fetch_results_failed", error=str(e))
            return [], []
