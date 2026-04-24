"""
Multibagger watchlist management.

Maintain in-memory and Redis-backed watchlist with scoring and filtering.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from data.store import get_redis
from fundamentals.ranking import compute_composite_rank, compute_percentile
from fundamentals.schema import FundamentalsScores

log = structlog.get_logger(__name__)


class MultibaggerWatchlist:
    """
    In-memory watchlist with Redis backing.

    Maintains a set of stocks with composite fundamentals scores.
    """

    REDIS_KEY = "trading:fundamentals:watchlist"
    REDIS_TTL = 30 * 24 * 3600  # 30 days

    def __init__(self):
        """Initialize watchlist."""
        self.scores: dict[str, FundamentalsScores] = {}
        self._load_from_redis()

    def _load_from_redis(self) -> None:
        """Load watchlist from Redis."""
        try:
            redis = get_redis()
            data = redis.get(self.REDIS_KEY)
            if data:
                watchlist_data = json.loads(data)
                for symbol, scores_dict in watchlist_data.items():
                    try:
                        self.scores[symbol] = FundamentalsScores(**scores_dict)
                    except Exception as e:
                        log.warning("parse_watchlist_score_error", symbol=symbol, error=str(e))
                log.info("loaded_watchlist_from_redis", count=len(self.scores))
        except Exception as e:
            log.warning("load_watchlist_error", error=str(e))

    def _save_to_redis(self) -> None:
        """Persist watchlist to Redis."""
        try:
            redis = get_redis()
            data = {
                symbol: json.loads(scores.model_dump_json())
                for symbol, scores in self.scores.items()
            }
            redis.setex(
                self.REDIS_KEY,
                self.REDIS_TTL,
                json.dumps(data, default=str),
            )
            log.debug("saved_watchlist_to_redis", count=len(self.scores))
        except Exception as e:
            log.warning("save_watchlist_error", error=str(e))

    def add_scores(self, symbol: str, scores: FundamentalsScores) -> None:
        """
        Add or update scores for a symbol.

        Args:
            symbol: Stock symbol
            scores: FundamentalsScores object
        """
        self.scores[symbol] = scores
        self._save_to_redis()
        log.info(
            "added_to_watchlist",
            symbol=symbol,
            composite_rank=scores.composite_rank,
        )

    def remove(self, symbol: str) -> bool:
        """
        Remove symbol from watchlist.

        Args:
            symbol: Stock symbol

        Returns:
            True if removed, False if not found
        """
        if symbol in self.scores:
            del self.scores[symbol]
            self._save_to_redis()
            log.info("removed_from_watchlist", symbol=symbol)
            return True
        return False

    def is_watchlisted(self, symbol: str) -> bool:
        """
        Check if symbol is on watchlist.

        Args:
            symbol: Stock symbol

        Returns:
            True if on watchlist
        """
        return symbol in self.scores

    def get_scores(self, symbol: str) -> FundamentalsScores | None:
        """
        Get scores for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            FundamentalsScores or None if not found
        """
        return self.scores.get(symbol)

    def get_all(self) -> dict[str, FundamentalsScores]:
        """
        Get all watchlist scores.

        Returns:
            Dict of symbol → FundamentalsScores
        """
        return dict(self.scores)

    def get_top_n(
        self,
        n: int = 20,
        sector: str | None = None,
        min_growth: float | None = None,
        min_quality: float | None = None,
        min_balance_sheet: float | None = None,
        min_composite: float | None = None,
        sort_by: str = "composite_rank",
    ) -> list[tuple[str, FundamentalsScores]]:
        """
        Get top N candidates with optional filters.

        Args:
            n: Number of results to return
            sector: Filter by sector (not implemented — placeholder)
            min_growth: Minimum growth score
            min_quality: Minimum quality score
            min_balance_sheet: Minimum balance sheet score
            min_composite: Minimum composite rank
            sort_by: Sort field: composite_rank, growth_score, quality_score, etc.

        Returns:
            List of (symbol, scores) tuples sorted by sort_by
        """
        filtered = []

        for symbol, scores in self.scores.items():
            # Apply filters
            if min_growth is not None and scores.growth_score < min_growth:
                continue
            if min_quality is not None and scores.quality_score < min_quality:
                continue
            if min_balance_sheet is not None and scores.balance_sheet_score < min_balance_sheet:
                continue
            if min_composite is not None and scores.composite_rank < min_composite:
                continue

            filtered.append((symbol, scores))

        # Sort by requested field
        def sort_key(item):
            return getattr(item[1], sort_by, 0)

        filtered.sort(key=sort_key, reverse=True)

        return filtered[:n]

    def update_scores(
        self,
        symbol: str,
        growth_score: float,
        quality_score: float,
        balance_sheet_score: float,
        valuation_score: float,
        momentum_score: float,
    ) -> FundamentalsScores:
        """
        Update scores for a symbol and recompute composite rank.

        Args:
            symbol: Stock symbol
            growth_score: Growth score (0-100)
            quality_score: Quality score (0-100)
            balance_sheet_score: Balance sheet score (0-100)
            valuation_score: Valuation score (0-100)
            momentum_score: Momentum score (0-100)

        Returns:
            Updated FundamentalsScores
        """
        # Compute composite rank
        composite, growth_weighted = compute_composite_rank(
            growth_score,
            quality_score,
            balance_sheet_score,
            valuation_score,
            momentum_score,
        )

        # Compute percentile
        all_composites = [s.composite_rank for s in self.scores.values()]
        percentile = compute_percentile(composite, all_composites)

        # Create scores object
        scores = FundamentalsScores(
            symbol=symbol,
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=growth_score,
            quality_score=quality_score,
            balance_sheet_score=balance_sheet_score,
            valuation_score=valuation_score,
            momentum_score=momentum_score,
            composite_rank=composite,
            percentile=percentile,
            growth_weighted=growth_weighted,
            data_completeness=1.0,
        )

        self.add_scores(symbol, scores)
        return scores

    def export_to_csv(self, path: str | Path) -> None:
        """
        Export watchlist to CSV for analysis.

        Args:
            path: Output file path
        """
        import csv

        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Sort by composite rank descending
            sorted_watchlist = sorted(
                self.scores.items(),
                key=lambda x: x[1].composite_rank,
                reverse=True,
            )

            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "symbol",
                        "composite_rank",
                        "growth_weighted",
                        "percentile",
                        "growth_score",
                        "quality_score",
                        "balance_sheet_score",
                        "valuation_score",
                        "momentum_score",
                        "timestamp",
                        "data_completeness",
                    ],
                )
                writer.writeheader()

                for symbol, scores in sorted_watchlist:
                    writer.writerow(
                        {
                            "symbol": symbol,
                            "composite_rank": round(scores.composite_rank, 2),
                            "growth_weighted": round(scores.growth_weighted or 0, 2),
                            "percentile": round(scores.percentile or 0, 2),
                            "growth_score": round(scores.growth_score, 2),
                            "quality_score": round(scores.quality_score, 2),
                            "balance_sheet_score": round(scores.balance_sheet_score, 2),
                            "valuation_score": round(scores.valuation_score, 2),
                            "momentum_score": round(scores.momentum_score, 2),
                            "timestamp": scores.timestamp.isoformat(),
                            "data_completeness": round(scores.data_completeness, 2),
                        }
                    )

            log.info("watchlist_exported", path=str(path), count=len(sorted_watchlist))
        except Exception as e:
            log.error("export_watchlist_error", path=str(path), error=str(e))

    def __len__(self) -> int:
        """Get watchlist size."""
        return len(self.scores)

    def __repr__(self) -> str:
        """String representation."""
        return f"MultibaggerWatchlist(size={len(self.scores)})"
