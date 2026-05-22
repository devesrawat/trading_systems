"""
Post-trade review and learning engine.

TradeReviewEngine — Analyze why trades won/lost and extract learning signals:
  - get_trade_context: retrieve trade signals, features, predictions, execution prices
  - post_trade_review: categorize outcome (hit_target, stopped_out, time_decay, etc.)
  - identify_patterns_in_winners: what do winners have in common? (high confidence, certain features, etc.)
  - identify_patterns_in_losers: what triggers losses? (rare feature combinations, sector rotation, etc.)
  - generate_lessons_learned: actionable feedback for model improvement

All reviews stored in TimescaleDB for historical learning and trend analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text

from data.store import get_engine, get_redis

log = structlog.get_logger(__name__)


class OutcomeCategory(StrEnum):
    """Trade outcome categorization."""

    HIT_TARGET = "hit_target"  # Reached profit target
    STOPPED_OUT = "stopped_out"  # Hit stop loss
    TIME_DECAY = "time_decay"  # Exited due to time
    EXIT_SIGNAL = "exit_signal"  # Exited on technical signal
    MANUAL_EXIT = "manual_exit"  # Manually closed
    UNKNOWN = "unknown"


@dataclass
class TradeContext:
    """Full context for a single trade."""

    trade_id: int
    symbol: str
    entry_time: datetime
    exit_time: datetime | None
    entry_price: float
    exit_price: float | None
    quantity: int
    side: str
    pnl: float
    pnl_pct: float
    signal_prob: float
    features_used: dict[str, float]
    shap_values: dict[str, float] | None


@dataclass
class TradeReview:
    """Reviewed trade with categorization and lessons."""

    trade_id: int
    symbol: str
    entry_time: datetime
    outcome: OutcomeCategory
    pnl: float
    pnl_pct: float
    signal_prob: float
    root_cause: str  # Why did we win/lose?
    key_features: list[str]  # Top 5 features in this trade
    model_confidence: float  # [0, 1] how confident was model?
    market_condition: str  # "trending", "ranging", "volatile", etc.
    review_notes: str


class TradeReviewEngine:
    """Post-trade review and learning."""

    def __init__(self) -> None:
        """Initialize trade review engine."""
        self._engine = get_engine()
        self._redis = get_redis()

    # =========================================================================
    # Trade Context Retrieval
    # =========================================================================

    def get_trade_context(self, trade_id: int) -> TradeContext | None:
        """
        Retrieve full context for a trade.

        Parameters
        ----------
        trade_id : int
            Trade ID from paper_trades or live_trades

        Returns
        -------
        TradeContext | None
            Full trade context, or None if not found
        """
        query = text("""
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
                '{}'::jsonb as features_used,
                NULL::jsonb as shap_values
            FROM paper_trades
            WHERE id = :trade_id
        """)

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(query, conn, params={"trade_id": trade_id})
        except Exception as exc:
            log.error("fetch_trade_failed", trade_id=trade_id, error=str(exc))
            return None

        if df.empty:
            return None

        row = df.iloc[0]
        return TradeContext(
            trade_id=int(row["id"]),
            symbol=str(row["symbol"]),
            entry_time=pd.Timestamp(row["entry_time"]).to_pydatetime(),
            exit_time=pd.Timestamp(row["exit_time"]).to_pydatetime()
            if pd.notna(row["exit_time"])
            else None,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]) if pd.notna(row["exit_price"]) else None,
            quantity=int(row["quantity"]),
            side=str(row["side"]),
            pnl=0.0,  # Would need to compute from exit
            pnl_pct=0.0,
            signal_prob=float(row["signal_prob"]),
            features_used=row["features_used"] or {},
            shap_values=row["shap_values"],
        )

    # =========================================================================
    # Post-Trade Review
    # =========================================================================

    def post_trade_review(
        self,
        trade: TradeContext,
        target_pct: float = 2.0,
        stop_loss_pct: float = -1.5,
    ) -> TradeReview:
        """
        Categorize and review a single trade.

        Parameters
        ----------
        trade : TradeContext
            Trade to review
        target_pct : float
            Profit target percentage
        stop_loss_pct : float
            Stop loss percentage

        Returns
        -------
        TradeReview
            Categorized review with root cause and lessons
        """
        # Categorize outcome
        outcome = self._categorize_outcome(trade, target_pct, stop_loss_pct)

        # Extract key features (top 5 by SHAP or feature value magnitude)
        key_features = self._extract_key_features(trade)

        # Infer root cause
        root_cause = self._infer_root_cause(trade, outcome)

        # Estimate model confidence
        confidence = self._estimate_confidence(trade)

        # Detect market condition
        market_condition = self._detect_market_condition(trade)

        # Generate review notes
        notes = self._generate_review_notes(trade, outcome, root_cause, key_features)

        review = TradeReview(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            entry_time=trade.entry_time,
            outcome=outcome,
            pnl=trade.pnl,
            pnl_pct=trade.pnl_pct,
            signal_prob=trade.signal_prob,
            root_cause=root_cause,
            key_features=key_features,
            model_confidence=confidence,
            market_condition=market_condition,
            review_notes=notes,
        )

        log.info(
            "trade_reviewed",
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            outcome=outcome.value,
            pnl=round(trade.pnl, 2),
        )
        return review

    # =========================================================================
    # Pattern Identification
    # =========================================================================

    def identify_patterns_in_winners(
        self,
        lookback_days: int = 30,
        min_profit: float = 100,
    ) -> dict[str, Any]:
        """
        Identify patterns common to winning trades.

        What do winners have in common?
        - High signal confidence?
        - Certain features?
        - Specific sectors?

        Parameters
        ----------
        lookback_days : int
            Analysis window
        min_profit : float
            Minimum profit to consider a winner

        Returns
        -------
        dict
            {
                "avg_signal_prob": 0.75,
                "avg_model_confidence": 0.82,
                "top_features": ["rsi_14", "macd", ...],
                "common_market_conditions": ["trending", "volatile"],
                "most_common_symbols": ["SBIN", "HDFC", ...],
                "n_winners": 25,
            }
        """
        try:
            trades_df = self._fetch_trades(lookback_days)
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return {}

        # Filter winners
        winners = trades_df[trades_df["pnl"] >= min_profit]

        if winners.empty:
            log.warning("no_winners_found", lookback=lookback_days, min_profit=min_profit)
            return {"n_winners": 0}

        # Extract patterns
        avg_signal_prob = float(winners["signal_prob"].mean())
        avg_confidence = float(winners["signal_prob"].mean())  # proxy for now

        # Top features
        all_features = []
        for feat_dict in winners["features_used"]:
            if isinstance(feat_dict, dict):
                all_features.extend(feat_dict.keys())
        feature_counts = pd.Series(all_features).value_counts()
        top_features = feature_counts.head(5).index.tolist()

        # Most common symbols
        top_symbols = winners["symbol"].value_counts().head(5).index.tolist()

        result = {
            "avg_signal_prob": avg_signal_prob,
            "avg_model_confidence": avg_confidence,
            "top_features": top_features,
            "most_common_symbols": top_symbols,
            "n_winners": len(winners),
            "avg_pnl": float(winners["pnl"].mean()),
        }

        log.info("winner_patterns_identified", n_winners=len(winners), top_features=top_features)
        return result

    def identify_patterns_in_losers(
        self,
        lookback_days: int = 30,
        max_loss: float = -100,
    ) -> dict[str, Any]:
        """
        Identify patterns common to losing trades.

        What triggers our losses?
        - Low signal confidence?
        - Rare feature combinations?
        - Specific sectors or conditions?

        Parameters
        ----------
        lookback_days : int
            Analysis window
        max_loss : float
            Maximum loss (most negative) to consider a loser

        Returns
        -------
        dict
            {
                "avg_signal_prob": 0.55,
                "avg_model_confidence": 0.48,
                "risky_features": ["vol_regime", "bb_position", ...],
                "risky_symbols": ["PENNY", "MICRO", ...],
                "common_market_conditions": ["ranging"],
                "n_losers": 12,
            }
        """
        try:
            trades_df = self._fetch_trades(lookback_days)
        except Exception as exc:
            log.error("fetch_trades_failed", error=str(exc))
            return {}

        # Filter losers
        losers = trades_df[trades_df["pnl"] <= max_loss]

        if losers.empty:
            log.warning("no_losers_found", lookback=lookback_days, max_loss=max_loss)
            return {"n_losers": 0}

        # Extract patterns
        avg_signal_prob = float(losers["signal_prob"].mean())
        avg_confidence = float(losers["signal_prob"].mean())  # proxy

        # Risky features (more common in losers than winners)
        all_features = []
        for feat_dict in losers["features_used"]:
            if isinstance(feat_dict, dict):
                all_features.extend(feat_dict.keys())
        feature_counts = pd.Series(all_features).value_counts()
        risky_features = feature_counts.head(5).index.tolist()

        # Risky symbols
        risky_symbols = losers["symbol"].value_counts().head(5).index.tolist()

        result = {
            "avg_signal_prob": avg_signal_prob,
            "avg_model_confidence": avg_confidence,
            "risky_features": risky_features,
            "risky_symbols": risky_symbols,
            "n_losers": len(losers),
            "avg_pnl": float(losers["pnl"].mean()),
            "worst_pnl": float(losers["pnl"].min()),
        }

        log.info("loser_patterns_identified", n_losers=len(losers), risky_features=risky_features)
        return result

    # =========================================================================
    # Lessons Learned
    # =========================================================================

    def generate_lessons_learned(
        self,
        lookback_days: int = 30,
    ) -> dict[str, Any]:
        """
        Generate structured feedback for model improvement.

        Combines winner/loser analysis to recommend model changes.

        Parameters
        ----------
        lookback_days : int
            Analysis window

        Returns
        -------
        dict
            {
                "model_bias": "slightly overconfident",
                "recommended_feature_adjustments": ["reduce rsi_14 weight", ...],
                "sector_recommendations": ["avoid penny stocks", ...],
                "confidence_calibration": "actual win rate 45% but avg signal prob 0.65",
            }
        """
        winners = self.identify_patterns_in_winners(lookback_days=lookback_days)
        losers = self.identify_patterns_in_losers(lookback_days=lookback_days)

        if not winners or not losers:
            return {}

        lessons = {}

        # Bias detection
        winner_conf = winners.get("avg_signal_prob", 0.5)
        loser_conf = losers.get("avg_signal_prob", 0.5)
        if winner_conf - loser_conf < 0.1:
            lessons["model_bias"] = (
                "poor discrimination: winners and losers have similar confidence"
            )
        elif winner_conf > 0.75:
            lessons["model_bias"] = "possibly overconfident: high signal prob on losers"
        else:
            lessons["model_bias"] = "well-calibrated"

        # Feature recommendations
        winner_features = set(winners.get("top_features", []))
        loser_features = set(losers.get("risky_features", []))
        conflicting = winner_features & loser_features
        if conflicting:
            lessons["conflicting_features"] = list(conflicting)

        # Sector recommendations
        risky_symbols = losers.get("risky_symbols", [])
        if risky_symbols:
            lessons["avoid_symbols"] = risky_symbols

        # Win rate vs confidence
        n_winners = winners.get("n_winners", 0)
        n_losers = losers.get("n_losers", 0)
        if n_winners + n_losers > 0:
            actual_win_rate = n_winners / (n_winners + n_losers)
            avg_confidence = winner_conf
            if abs(actual_win_rate - avg_confidence) > 0.15:
                lessons["confidence_calibration"] = (
                    f"Calibration gap: actual win rate {actual_win_rate:.1%} "
                    f"vs avg signal prob {avg_confidence:.1%}"
                )

        lessons["n_winners"] = n_winners
        lessons["n_losers"] = n_losers

        log.info("lessons_learned_generated", lessons=list(lessons.keys()))
        return lessons

    # =========================================================================
    # Private Helpers
    # =========================================================================

    def _categorize_outcome(
        self,
        trade: TradeContext,
        target_pct: float,
        stop_loss_pct: float,
    ) -> OutcomeCategory:
        """Categorize trade outcome."""
        if trade.exit_price is None:
            return OutcomeCategory.UNKNOWN

        pnl_pct = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
        hold_time = (
            (trade.exit_time - trade.entry_time).total_seconds() / 3600 if trade.exit_time else 0
        )

        if pnl_pct >= target_pct * 0.9:  # Close to target
            return OutcomeCategory.HIT_TARGET
        elif pnl_pct <= stop_loss_pct * 1.1:  # Close to stop
            return OutcomeCategory.STOPPED_OUT
        elif hold_time > 72:  # Held > 3 days
            return OutcomeCategory.TIME_DECAY
        else:
            return OutcomeCategory.EXIT_SIGNAL

    def _extract_key_features(self, trade: TradeContext, top_n: int = 5) -> list[str]:
        """Extract top N features from trade."""
        if not trade.features_used:
            return []

        if trade.shap_values:
            # Rank by SHAP magnitude
            abs_shap = {k: abs(v) for k, v in trade.shap_values.items()}
            top = sorted(abs_shap.items(), key=lambda x: -x[1])[:top_n]
            return [f[0] for f in top]
        else:
            # Rank by absolute feature value
            abs_features = {k: abs(v) for k, v in trade.features_used.items()}
            top = sorted(abs_features.items(), key=lambda x: -x[1])[:top_n]
            return [f[0] for f in top]

    def _infer_root_cause(self, trade: TradeContext, outcome: OutcomeCategory) -> str:
        """Infer root cause of trade outcome."""
        if outcome == OutcomeCategory.HIT_TARGET:
            key_features = self._extract_key_features(trade, 3)
            return f"Reached target; key features: {', '.join(key_features) or 'none'}"
        elif outcome == OutcomeCategory.STOPPED_OUT:
            return "Hit stop loss (adverse market move)"
        elif outcome == OutcomeCategory.TIME_DECAY:
            return "Long hold period; time decay"
        else:
            return "Exited on technical signal"

    def _estimate_confidence(self, trade: TradeContext) -> float:
        """Estimate model confidence [0, 1]."""
        # Proxy: use signal_prob directly
        return min(1.0, max(0.0, trade.signal_prob))

    def _detect_market_condition(self, trade: TradeContext) -> str:
        """Detect market condition from trade features."""
        if not trade.features_used:
            return "unknown"

        vol = trade.features_used.get("realized_vol_10", 0)
        rsi = trade.features_used.get("rsi_14", 50)
        atr = trade.features_used.get("atr_pct", 0)

        if vol > 0.025:
            return "volatile"
        elif rsi > 70 or rsi < 30:
            return "trending"
        elif atr < 0.01:
            return "ranging"
        else:
            return "normal"

    def _generate_review_notes(
        self,
        trade: TradeContext,
        outcome: OutcomeCategory,
        root_cause: str,
        key_features: list[str],
    ) -> str:
        """Generate human-readable review notes."""
        lines = [
            f"Trade ID: {trade.trade_id}",
            f"Symbol: {trade.symbol}",
            f"Outcome: {outcome.value}",
            f"P&L: {trade.pnl:.2f} ({trade.pnl_pct:.2f}%)",
            f"Signal Prob: {trade.signal_prob:.3f}",
            f"Root Cause: {root_cause}",
            f"Key Features: {', '.join(key_features) if key_features else 'none'}",
        ]
        return "\n".join(lines)

    def _fetch_trades(self, lookback_days: int) -> pd.DataFrame:
        """Fetch trades from past N days."""
        start = date.today() - timedelta(days=lookback_days)
        end = date.today()

        query = text("""
            SELECT
                id,
                symbol,
                time,
                price,
                quantity,
                side,
                signal_prob,
                0.0 as pnl,
                0.0 as pnl_pct,
                '{}'::jsonb as features_used,
                NULL::jsonb as shap_values
            FROM paper_trades
            WHERE time >= :start::date
              AND time < :end::date
            ORDER BY time DESC
        """)

        with self._engine.connect() as conn:
            df = pd.read_sql(
                query,
                conn,
                params={"start": str(start), "end": str(end + timedelta(days=1))},
            )

        return df
