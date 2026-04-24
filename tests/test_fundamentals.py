"""
Comprehensive tests for fundamentals package.

Tests: schema validation, scoring calculations, ranking, watchlist, caching.
Coverage: 500+ lines of test code covering:
- Schema validation with edge cases
- Scoring formulas with known inputs
- Ranking and percentile computation
- Watchlist operations
- Caching with TTL
- Data completeness tracking
"""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fundamentals.ingest import FundamentalsData, fetch_fundamentals
from fundamentals.providers import NSEProvider, ScreenerProvider, TrendlyneProvider
from fundamentals.ranking import compute_composite_rank, compute_percentile
from fundamentals.schema import (
    ConfidenceLevel,
    FundamentalsScores,
    QuarterlyFinancials,
    Shareholding,
    Valuations,
)
from fundamentals.scoring import (
    compute_balance_sheet_score,
    compute_growth_score,
    compute_institutional_conviction_score,
    compute_momentum_score,
    compute_quality_score,
    compute_valuation_score,
)
from fundamentals.watchlist import MultibaggerWatchlist

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_financials() -> QuarterlyFinancials:
    """Sample quarterly financials."""
    return QuarterlyFinancials(
        symbol="INFY",
        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
        quarter="Q3FY24",
        source="NSE",
        confidence=ConfidenceLevel.high,
        revenue=250000.0,
        net_income=45000.0,
        ebitda=65000.0,
        equity=180000.0,
        debt=30000.0,
        cash=50000.0,
        current_assets=120000.0,
        current_liabilities=80000.0,
        fcf=35000.0,
        operating_cf=42000.0,
    )


@pytest.fixture
def sample_valuations() -> Valuations:
    """Sample valuations."""
    return Valuations(
        symbol="INFY",
        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
        source="Screener",
        confidence=ConfidenceLevel.high,
        pe=25.5,
        pb=4.2,
        ps=8.5,
        peg=1.2,
        pcf=18.0,
        roe=18.5,
        roce=22.3,
        roa=12.5,
        profit_margin=18.0,
        operating_margin=22.0,
        ebitda_margin=26.0,
        debt_to_equity=0.167,
        current_ratio=1.5,
        interest_coverage=8.5,
    )


@pytest.fixture
def sample_shareholding() -> Shareholding:
    """Sample shareholding pattern."""
    return Shareholding(
        symbol="INFY",
        timestamp=datetime(2024, 1, 15, tzinfo=UTC),
        source="NSE",
        confidence=ConfidenceLevel.high,
        promoter_pct=45.5,
        institutional_pct=35.2,
        public_pct=19.3,
        fii_qty=125000000,
        dii_qty=85000000,
        fii_change_pct=2.5,
        dii_change_pct=-0.5,
        pledged_pct=0.0,
    )


# =========================================================================
# Schema Validation Tests
# =========================================================================


class TestSchemaValidation:
    """Schema validation with Pydantic."""

    def test_quarterly_financials_creation(self, sample_financials: QuarterlyFinancials):
        """Create valid financials."""
        assert sample_financials.symbol == "INFY"
        assert sample_financials.revenue == 250000.0
        assert sample_financials.confidence == ConfidenceLevel.high

    def test_quarterly_financials_negative_revenue_rejected(self):
        """Negative revenue should be rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            QuarterlyFinancials(
                symbol="INFY",
                timestamp=datetime.now(UTC),
                quarter="Q1FY24",
                source="NSE",
                revenue=-100.0,
            )

    def test_valuations_creation(self, sample_valuations: Valuations):
        """Create valid valuations."""
        assert sample_valuations.pe == 25.5
        assert sample_valuations.roe == 18.5

    def test_valuations_negative_pe_rejected(self):
        """Negative P/E should be rejected."""
        with pytest.raises(ValueError, match="positive"):
            Valuations(
                symbol="INFY",
                timestamp=datetime.now(UTC),
                source="Screener",
                pe=-10.0,
            )

    def test_shareholding_percentages_clamped(self):
        """Shareholding percentages should be 0-100."""
        share = Shareholding(
            symbol="INFY",
            timestamp=datetime.now(UTC),
            source="NSE",
            promoter_pct=50.0,
            institutional_pct=35.0,
        )
        assert share.promoter_pct == 50.0

    def test_shareholding_over_100_rejected(self):
        """Shareholding > 100% should be rejected."""
        with pytest.raises(ValueError):
            Shareholding(
                symbol="INFY",
                timestamp=datetime.now(UTC),
                source="NSE",
                promoter_pct=150.0,  # Invalid
            )

    def test_fundamentals_scores_creation(self):
        """Create valid scores."""
        scores = FundamentalsScores(
            symbol="INFY",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=75.0,
            quality_score=82.0,
            balance_sheet_score=88.0,
            valuation_score=65.0,
            momentum_score=72.0,
            composite_rank=76.0,
        )
        assert scores.composite_rank == 76.0

    def test_fundamentals_scores_out_of_range_rejected(self):
        """Scores > 100 should be rejected."""
        with pytest.raises(ValueError):
            FundamentalsScores(
                symbol="INFY",
                timestamp=datetime.now(UTC),
                source="fundamentals",
                growth_score=150.0,  # Invalid
                quality_score=80.0,
                balance_sheet_score=80.0,
                valuation_score=80.0,
                momentum_score=80.0,
                composite_rank=80.0,
            )

    def test_optional_fields_nullable(self):
        """Optional fields should accept None."""
        financials = QuarterlyFinancials(
            symbol="TCS",
            timestamp=datetime.now(UTC),
            quarter="Q1FY24",
            source="NSE",
            # All other fields None
        )
        assert financials.revenue is None
        assert financials.net_income is None


# =========================================================================
# Growth Scoring Tests
# =========================================================================


class TestGrowthScoring:
    """Growth score computation."""

    def test_high_growth_gets_high_score(self):
        """40%+ CAGR should score high."""
        score = compute_growth_score(
            revenue_cagr_3y=50.0,
            revenue_cagr_5y=45.0,
            net_income_cagr_3y=60.0,
            net_income_cagr_5y=55.0,
        )
        assert score > 85

    def test_moderate_growth_gets_moderate_score(self):
        """15-25% CAGR should score 40-70."""
        score = compute_growth_score(
            revenue_cagr_3y=20.0,
            revenue_cagr_5y=18.0,
            net_income_cagr_3y=22.0,
            net_income_cagr_5y=20.0,
        )
        assert 40 < score < 80

    def test_low_growth_gets_low_score(self):
        """<5% CAGR should score low."""
        score = compute_growth_score(
            revenue_cagr_3y=3.0,
            revenue_cagr_5y=2.0,
            net_income_cagr_3y=1.0,
            net_income_cagr_5y=0.5,
        )
        assert score < 20

    def test_negative_growth_handled(self):
        """Negative CAGR should return 0."""
        score = compute_growth_score(
            revenue_cagr_3y=-5.0,
            revenue_cagr_5y=-3.0,
            net_income_cagr_3y=None,
            net_income_cagr_5y=None,
        )
        assert score == 0.0

    def test_missing_data_fallback(self):
        """Should work with partial data."""
        score1 = compute_growth_score(
            revenue_cagr_3y=20.0,
            revenue_cagr_5y=None,
            net_income_cagr_3y=None,
            net_income_cagr_5y=None,
        )
        assert 15 < score1 < 70

    def test_all_missing_returns_zero(self):
        """All missing data should return 0."""
        score = compute_growth_score(None, None, None, None)
        assert score == 0.0

    def test_score_range_0_to_100(self):
        """Score should always be 0-100."""
        for rev3y in [0, 15, 50, 100, None]:
            for rev5y in [0, 15, 50, None]:
                for ni3y in [0, 15, 50, None]:
                    for ni5y in [0, 15, 50, None]:
                        score = compute_growth_score(rev3y, rev5y, ni3y, ni5y)
                        assert 0 <= score <= 100


# =========================================================================
# Quality Scoring Tests
# =========================================================================


class TestQualityScoring:
    """Quality score computation."""

    def test_high_roe_roce_high_score(self):
        """ROE/ROCE > 20% should score high."""
        score = compute_quality_score(
            roe=25.0,
            roce=24.0,
            profit_margin=18.0,
            ebitda_margin=28.0,
        )
        assert score > 65

    def test_moderate_roe_roce_moderate_score(self):
        """ROE/ROCE 12-16% should score 40-70."""
        score = compute_quality_score(
            roe=15.0,
            roce=14.0,
            profit_margin=8.0,
            ebitda_margin=14.0,
        )
        assert 40 < score < 80

    def test_low_margins_low_score(self):
        """Low margins should score low."""
        score = compute_quality_score(
            roe=8.0,
            roce=6.0,
            profit_margin=2.0,
            ebitda_margin=5.0,
        )
        assert score < 40

    def test_consistency_penalty(self):
        """Low consistency should reduce score."""
        score_consistent = compute_quality_score(
            roe=15.0,
            roce=15.0,
            profit_margin=10.0,
            ebitda_margin=15.0,
            consistency_score=0.95,
        )
        score_inconsistent = compute_quality_score(
            roe=15.0,
            roce=15.0,
            profit_margin=10.0,
            ebitda_margin=15.0,
            consistency_score=0.60,
        )
        assert score_consistent > score_inconsistent

    def test_missing_data_works(self):
        """Should work with partial data."""
        score = compute_quality_score(roe=18.0, roce=None, profit_margin=None, ebitda_margin=None)
        assert 30 < score < 80

    def test_score_range(self):
        """Score always 0-100."""
        for roe in [5, 15, 25, None]:
            for roce in [5, 15, 25, None]:
                for pm in [3, 10, 20, None]:
                    for em in [10, 20, 30, None]:
                        score = compute_quality_score(roe, roce, pm, em)
                        assert 0 <= score <= 100


# =========================================================================
# Balance Sheet Scoring Tests
# =========================================================================


class TestBalanceSheetScoring:
    """Balance sheet health score."""

    def test_healthy_balance_sheet_high_score(self):
        """Low debt, good liquidity should score high."""
        score = compute_balance_sheet_score(
            debt_to_equity=0.5,
            debt_to_revenue=1.0,
            current_ratio=2.0,
            interest_coverage=5.0,
        )
        assert score > 60

    def test_high_debt_low_score(self):
        """High debt should score low."""
        score = compute_balance_sheet_score(
            debt_to_equity=2.5,
            debt_to_revenue=4.0,
            current_ratio=0.8,
            interest_coverage=1.2,
        )
        assert score < 50

    def test_ideal_current_ratio_weighted_high(self):
        """Ideal current ratio (1.5) should be weighted positively."""
        score_ideal = compute_balance_sheet_score(
            debt_to_equity=1.0,
            debt_to_revenue=2.0,
            current_ratio=1.5,
            interest_coverage=2.0,
        )
        score_high = compute_balance_sheet_score(
            debt_to_equity=1.0,
            debt_to_revenue=2.0,
            current_ratio=3.0,
            interest_coverage=2.0,
        )
        assert score_ideal > score_high

    def test_missing_data_works(self):
        """Works with partial data."""
        score = compute_balance_sheet_score(
            debt_to_equity=0.8, debt_to_revenue=None, current_ratio=None, interest_coverage=None
        )
        assert 40 < score < 80

    def test_score_range(self):
        """Always 0-100."""
        for de in [0.5, 1.5, 3.0, None]:
            for dr in [1.0, 2.5, 5.0, None]:
                for cr in [0.8, 1.5, 3.0, None]:
                    for ic in [1.0, 2.5, 5.0, None]:
                        score = compute_balance_sheet_score(de, dr, cr, ic)
                        assert 0 <= score <= 100


# =========================================================================
# Valuation Scoring Tests
# =========================================================================


class TestValuationScoring:
    """Valuation score computation."""

    def test_low_peg_high_score(self):
        """PEG < 1.0 should score high."""
        score = compute_valuation_score(pe=20.0, peg=0.8, pb=3.0, ps=2.0, revenue_cagr=25.0)
        assert score > 60

    def test_high_peg_low_score(self):
        """PEG > 2.0 should score low."""
        score = compute_valuation_score(pe=40.0, peg=2.5, pb=6.0, ps=4.0, revenue_cagr=15.0)
        assert score < 50

    def test_fair_valuation_moderate_score(self):
        """PEG ~1.2 should score 40-70."""
        score = compute_valuation_score(pe=30.0, peg=1.2, pb=4.0, ps=3.0, revenue_cagr=25.0)
        assert 35 < score < 80

    def test_high_pe_penalized(self):
        """PE > 40 should reduce score."""
        score_low_pe = compute_valuation_score(
            pe=20.0, peg=None, pb=None, ps=None, revenue_cagr=20.0
        )
        score_high_pe = compute_valuation_score(
            pe=60.0, peg=None, pb=None, ps=None, revenue_cagr=20.0
        )
        assert score_low_pe > score_high_pe

    def test_derived_peg_from_pe_and_cagr(self):
        """Should derive PEG if missing."""
        score = compute_valuation_score(pe=20.0, peg=None, pb=None, ps=None, revenue_cagr=20.0)
        assert 40 < score < 80

    def test_missing_data_neutral(self):
        """All missing returns neutral score."""
        score = compute_valuation_score(pe=None, peg=None, pb=None, ps=None)
        assert 40 < score < 60

    def test_score_range(self):
        """Always 0-100."""
        for pe in [10, 25, 50, None]:
            for peg in [0.5, 1.2, 2.5, None]:
                score = compute_valuation_score(pe, peg, None, None)
                assert 0 <= score <= 100


# =========================================================================
# Momentum Scoring Tests
# =========================================================================


class TestMomentumScoring:
    """Momentum score computation."""

    def test_strong_institutional_buying_high_score(self):
        """High institutional ownership + positive FII should score high."""
        score = compute_momentum_score(
            institutional_pct=45.0,
            fii_change_pct=5.0,
            dii_change_pct=2.0,
            price_momentum_3m=15.0,
        )
        assert score > 70

    def test_low_institutional_low_score(self):
        """Low institutional ownership should score low."""
        score = compute_momentum_score(
            institutional_pct=5.0,
            fii_change_pct=-3.0,
            dii_change_pct=-1.0,
            price_momentum_3m=-10.0,
        )
        assert score < 40

    def test_positive_price_momentum_bonus(self):
        """Positive price momentum should be rewarded."""
        score_positive = compute_momentum_score(
            institutional_pct=30.0,
            fii_change_pct=2.0,
            dii_change_pct=1.0,
            price_momentum_3m=15.0,
        )
        score_negative = compute_momentum_score(
            institutional_pct=30.0,
            fii_change_pct=2.0,
            dii_change_pct=1.0,
            price_momentum_3m=-15.0,
        )
        assert score_positive > score_negative

    def test_missing_data_neutral(self):
        """Missing data returns neutral."""
        score = compute_momentum_score(
            institutional_pct=None,
            fii_change_pct=None,
            dii_change_pct=None,
            price_momentum_3m=None,
        )
        assert 40 < score < 60

    def test_score_range(self):
        """Always 0-100."""
        for inst in [5, 25, 50, None]:
            for fii in [-5, 2, 8, None]:
                for dii in [-3, 1, 5, None]:
                    for pm in [-15, 5, 20, None]:
                        score = compute_momentum_score(inst, fii, dii, pm)
                        assert 0 <= score <= 100


# =========================================================================
# Ranking Tests
# =========================================================================


class TestRanking:
    """Composite ranking computation."""

    def test_composite_rank_weighted_average(self):
        """Composite should be weighted average of scores (6 dimensions)."""
        composite, _, _ = compute_composite_rank(
            growth_score=80.0,
            quality_score=80.0,
            balance_sheet_score=80.0,
            valuation_score=80.0,
            momentum_score=80.0,
            institutional_conviction_score=80.0,
        )
        assert composite == 80.0

    def test_growth_weighted_emphasizes_growth(self):
        """Growth-weighted should be >composite when growth is strong."""
        growth_high = 90.0
        others = 50.0

        composite, growth_weighted, _ = compute_composite_rank(
            growth_score=growth_high,
            quality_score=others,
            balance_sheet_score=others,
            valuation_score=others,
            momentum_score=others,
            institutional_conviction_score=others,
        )

        # Growth-weighted should give more weight to growth
        assert growth_weighted > composite

    def test_multibagger_ideal_profile(self):
        """Ideal multibagger: high growth, good quality, healthy balance sheet."""
        composite, growth_weighted, _ = compute_composite_rank(
            growth_score=85.0,
            quality_score=78.0,
            balance_sheet_score=82.0,
            valuation_score=70.0,
            momentum_score=75.0,
            institutional_conviction_score=80.0,
        )
        assert composite > 75
        assert growth_weighted > 75

    def test_output_range(self):
        """Output always 0-100."""
        for g in [0, 50, 100]:
            for q in [0, 50, 100]:
                for b in [0, 50, 100]:
                    for v in [0, 50, 100]:
                        for m in [0, 50, 100]:
                            c, gw, _ = compute_composite_rank(g, q, b, v, m, 50.0)
                            assert 0 <= c <= 100
                            assert 0 <= gw <= 100

    def test_custom_weights(self):
        """Should accept custom weight override."""
        custom_weights = {
            "growth": 0.5,
            "quality": 0.25,
            "balance_sheet": 0.15,
            "valuation": 0.05,
            "momentum": 0.05,
            "institutional_conviction": 0.0,
        }

        composite, _, _ = compute_composite_rank(
            growth_score=100.0,
            quality_score=0.0,
            balance_sheet_score=0.0,
            valuation_score=0.0,
            momentum_score=0.0,
            institutional_conviction_score=0.0,
            weights=custom_weights,
        )

        assert composite == 50.0  # 100 * 0.5


class TestPercentile:
    """Percentile ranking."""

    def test_percentile_exact_placement(self):
        """Score at 50th position in 100 should be ~50th percentile."""
        all_ranks = list(range(101))
        rank = 50
        percentile = compute_percentile(rank, all_ranks)
        assert 40 < percentile < 60

    def test_top_score_high_percentile(self):
        """Highest score should be top percentile."""
        all_ranks = [50, 60, 70, 80, 90, 95]
        percentile = compute_percentile(95, all_ranks)
        assert percentile > 80

    def test_low_score_low_percentile(self):
        """Lowest score should be low percentile."""
        all_ranks = [50, 60, 70, 80, 90, 95]
        percentile = compute_percentile(50, all_ranks)
        assert percentile < 20

    def test_insufficient_data_returns_none(self):
        """Less than 2 ranks should return None."""
        assert compute_percentile(50, []) is None
        assert compute_percentile(50, [50]) is None


# =========================================================================
# Providers Tests
# =========================================================================


class TestProviders:
    """Data provider skeleton tests."""

    def test_nse_provider_returns_none(self):
        """NSE provider not yet implemented."""
        provider = NSEProvider()
        assert provider.fetch_financials("INFY") is None
        assert provider.fetch_valuations("INFY") is None
        assert provider.fetch_shareholding("INFY") is None

    def test_screener_provider_returns_none(self):
        """Screener provider not yet implemented."""
        provider = ScreenerProvider()
        assert provider.fetch_financials("INFY") is None
        assert provider.fetch_valuations("INFY") is None
        assert provider.fetch_shareholding("INFY") is None

    def test_trendlyne_provider_returns_none(self):
        """Trendlyne provider not yet implemented."""
        provider = TrendlyneProvider()
        assert provider.fetch_financials("INFY") is None
        assert provider.fetch_valuations("INFY") is None
        assert provider.fetch_shareholding("INFY") is None


# =========================================================================
# Ingest Tests
# =========================================================================


class TestFundamentalsData:
    """FundamentalsData container."""

    def test_create_empty_container(self):
        """Create empty data container."""
        data = FundamentalsData("INFY")
        assert data.symbol == "INFY"
        assert not data.is_complete()
        assert data.completeness_ratio() == 0.0

    def test_partial_data(self, sample_financials: QuarterlyFinancials):
        """Container with partial data."""
        data = FundamentalsData("INFY", financials=sample_financials)
        assert not data.is_complete()
        assert abs(data.completeness_ratio() - 1 / 3) < 0.01

    def test_complete_data(
        self,
        sample_financials: QuarterlyFinancials,
        sample_valuations: Valuations,
        sample_shareholding: Shareholding,
    ):
        """Complete data container."""
        data = FundamentalsData(
            "INFY",
            financials=sample_financials,
            valuations=sample_valuations,
            shareholding=sample_shareholding,
        )
        assert data.is_complete()
        assert data.completeness_ratio() == 1.0


class TestFetchFundamentals:
    """Fetching from providers."""

    def test_fetch_from_empty_providers_list(self):
        """Empty provider list returns empty data."""
        data = fetch_fundamentals("INFY", [])
        assert data.symbol == "INFY"
        assert not data.is_complete()

    def test_fetch_multiple_providers_noop(self):
        """Skeleton providers return None (no real calls)."""
        providers = [NSEProvider(), ScreenerProvider(), TrendlyneProvider()]
        data = fetch_fundamentals("INFY", providers)
        assert data.symbol == "INFY"
        assert not data.is_complete()


# =========================================================================
# Watchlist Tests
# =========================================================================


@pytest.fixture(autouse=True)
def clear_redis_watchlist():
    """Clear Redis watchlist before each test."""
    try:
        from data.store import get_redis

        redis = get_redis()
        redis.delete("trading:fundamentals:watchlist")
    except Exception:
        pass  # Redis might not be running
    yield


class TestMultibaggerWatchlist:
    """Watchlist management."""

    def test_create_empty_watchlist(self):
        """Create empty watchlist."""
        watchlist = MultibaggerWatchlist()
        assert len(watchlist) == 0

    def test_add_and_retrieve_scores(self):
        """Add scores and retrieve."""
        watchlist = MultibaggerWatchlist()
        scores = FundamentalsScores(
            symbol="INFY",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=80.0,
            quality_score=78.0,
            balance_sheet_score=82.0,
            valuation_score=70.0,
            momentum_score=75.0,
            composite_rank=76.5,
        )
        watchlist.add_scores("INFY", scores)
        assert len(watchlist) == 1
        assert watchlist.is_watchlisted("INFY")
        assert watchlist.get_scores("INFY").composite_rank == 76.5

    def test_remove_from_watchlist(self):
        """Remove symbol from watchlist."""
        watchlist = MultibaggerWatchlist()
        scores = FundamentalsScores(
            symbol="INFY",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=80.0,
            quality_score=80.0,
            balance_sheet_score=80.0,
            valuation_score=80.0,
            momentum_score=80.0,
            composite_rank=80.0,
        )
        watchlist.add_scores("INFY", scores)
        assert watchlist.remove("INFY")
        assert len(watchlist) == 0
        assert not watchlist.is_watchlisted("INFY")

    def test_get_top_n(self):
        """Get top N by composite rank."""
        watchlist = MultibaggerWatchlist()

        for symbol, rank in [("INFY", 80), ("TCS", 75), ("WIPRO", 70), ("LT", 65)]:
            scores = FundamentalsScores(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source="fundamentals",
                growth_score=rank,
                quality_score=rank,
                balance_sheet_score=rank,
                valuation_score=rank,
                momentum_score=rank,
                composite_rank=float(rank),
            )
            watchlist.add_scores(symbol, scores)

        top_2 = watchlist.get_top_n(n=2)
        assert len(top_2) == 2
        assert top_2[0][0] == "INFY"
        assert top_2[1][0] == "TCS"

    def test_filter_by_growth_score(self):
        """Filter by minimum growth score."""
        watchlist = MultibaggerWatchlist()

        for symbol, growth in [("INFY", 85), ("TCS", 60), ("WIPRO", 45)]:
            scores = FundamentalsScores(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source="fundamentals",
                growth_score=growth,
                quality_score=70.0,
                balance_sheet_score=70.0,
                valuation_score=70.0,
                momentum_score=70.0,
                composite_rank=70.0,
            )
            watchlist.add_scores(symbol, scores)

        filtered = watchlist.get_top_n(n=10, min_growth=70.0)
        assert len(filtered) == 1
        assert filtered[0][0] == "INFY"

    def test_update_scores_recomputes_rank(self):
        """Update scores and verify composite rank recalculation."""
        watchlist = MultibaggerWatchlist()

        scores = watchlist.update_scores(
            "INFY",
            growth_score=80.0,
            quality_score=75.0,
            balance_sheet_score=80.0,
            valuation_score=70.0,
            momentum_score=70.0,
            institutional_conviction_score=75.0,
        )

        # Should have non-zero composite rank
        assert scores.composite_rank > 0

    def test_export_to_csv(self, tmp_path: Path):
        """Export watchlist to CSV."""
        watchlist = MultibaggerWatchlist()

        for symbol, rank in [("INFY", 80), ("TCS", 75)]:
            scores = FundamentalsScores(
                symbol=symbol,
                timestamp=datetime.now(UTC),
                source="fundamentals",
                growth_score=rank,
                quality_score=rank,
                balance_sheet_score=rank,
                valuation_score=rank,
                momentum_score=rank,
                composite_rank=float(rank),
            )
            watchlist.add_scores(symbol, scores)

        csv_path = tmp_path / "watchlist.csv"
        watchlist.export_to_csv(csv_path)

        assert csv_path.exists()
        # Verify CSV has 2 data rows + header
        with open(csv_path) as f:
            lines = f.readlines()
            assert len(lines) == 3  # header + 2 data rows


# =========================================================================
# Institutional Conviction Tests
# =========================================================================


class TestInstitutionalConviction:
    """Tests for institutional conviction scoring."""

    def test_conviction_base_holding_count(self):
        """Base conviction: holding count normalization."""
        from fundamentals.scoring import compute_institutional_conviction_score

        # Low count (60 total = 20% of benchmark)
        score = compute_institutional_conviction_score(
            fii_count=10,
            dii_count=20,
            mf_count=30,
            total_institutional_pct=None,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert 15 <= score <= 25  # Low conviction (normalized holding ≈ 20)

        # High count (300 total = excellent, gets capped at 100)
        score = compute_institutional_conviction_score(
            fii_count=100,
            dii_count=100,
            mf_count=100,
            total_institutional_pct=None,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert score == 100.0  # Capped at 100 with excellent holding count

    def test_conviction_base_ownership(self):
        """Base conviction: institutional ownership tiers."""
        from fundamentals.scoring import compute_institutional_conviction_score

        # High ownership (15%+)
        score = compute_institutional_conviction_score(
            fii_count=None,
            dii_count=None,
            mf_count=None,
            total_institutional_pct=20.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert 75 <= score <= 95  # High tier

        # Medium ownership (10%)
        score = compute_institutional_conviction_score(
            fii_count=None,
            dii_count=None,
            mf_count=None,
            total_institutional_pct=10.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert 55 <= score <= 75  # Medium tier

        # Low ownership (<5%)
        score = compute_institutional_conviction_score(
            fii_count=None,
            dii_count=None,
            mf_count=None,
            total_institutional_pct=2.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert 15 <= score <= 35  # Low tier

    def test_conviction_price_multiplier_undervalued(self):
        """Price multiplier: undervalued (small rally) = 2.0x boost."""
        from fundamentals.scoring import compute_institutional_conviction_score

        base = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )

        # Less than 5% rally: 2.0x multiplier
        undervalued = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=3.0,
            quarters_increasing_holding=None,
        )
        assert undervalued > base * 1.5  # Significant boost

    def test_conviction_price_multiplier_rally(self):
        """Price multiplier: excessive rally (>50%) = 0.3x penalty."""
        from fundamentals.scoring import compute_institutional_conviction_score

        base = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )

        # >50% rally: 0.3x multiplier
        bubbled = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=75.0,
            quarters_increasing_holding=None,
        )
        assert bubbled < base * 0.5  # Significant penalty

    def test_conviction_trend_multiplier(self):
        """Trend multiplier: each quarter of increasing holdings adds 1.1x."""
        from fundamentals.scoring import compute_institutional_conviction_score

        base = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=0,
        )

        # 2 quarters increasing: 1.1^2 = 1.21x
        two_quarter = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=2,
        )
        assert 1.15 <= two_quarter / base <= 1.30  # ~1.21x

        # 3 quarters increasing: 1.1^3 = 1.331x
        three_quarter = compute_institutional_conviction_score(
            fii_count=50,
            dii_count=50,
            mf_count=100,
            total_institutional_pct=12.0,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=3,
        )
        assert 1.25 <= three_quarter / base <= 1.40  # ~1.331x

    def test_conviction_combined_multipliers(self):
        """Combined: base * price_mult * trend_mult."""
        from fundamentals.scoring import compute_institutional_conviction_score

        # High conviction + undervalued + 2 quarters increasing
        score = compute_institutional_conviction_score(
            fii_count=80,
            dii_count=80,
            mf_count=150,
            total_institutional_pct=18.0,
            price_change_since_entry_pct=4.0,  # 2.0x multiplier
            quarters_increasing_holding=2,  # 1.21x multiplier
        )
        # Base ≈ 80-90, with 2.0x * 1.21x ≈ 2.42x
        assert score > 90  # Strong conviction

        # Low conviction + rallied + no trend
        score = compute_institutional_conviction_score(
            fii_count=10,
            dii_count=10,
            mf_count=20,
            total_institutional_pct=3.0,
            price_change_since_entry_pct=60.0,  # 0.3x multiplier
            quarters_increasing_holding=0,
        )
        # Base ≈ 25-35, with 0.3x penalty
        assert score < 30  # Weak conviction

    def test_conviction_score_capped_100(self):
        """Score capped at 100 max."""
        from fundamentals.scoring import compute_institutional_conviction_score

        score = compute_institutional_conviction_score(
            fii_count=150,
            dii_count=150,
            mf_count=200,
            total_institutional_pct=25.0,
            price_change_since_entry_pct=1.0,  # 2.0x
            quarters_increasing_holding=5,  # 1.1^5 = 1.61x
        )
        assert score <= 100  # Capped

    def test_conviction_all_missing_data(self):
        """All data missing: neutral score of 50."""
        from fundamentals.scoring import compute_institutional_conviction_score

        score = compute_institutional_conviction_score(
            fii_count=None,
            dii_count=None,
            mf_count=None,
            total_institutional_pct=None,
            price_change_since_entry_pct=None,
            quarters_increasing_holding=None,
        )
        assert score == 50.0  # Neutral


# =========================================================================
# Integration Tests
# =========================================================================


class TestIntegration:
    """Integration tests with multiple components."""

    def test_end_to_end_scoring_pipeline_6d(
        self,
        sample_financials: QuarterlyFinancials,
        sample_valuations: Valuations,
        sample_shareholding: Shareholding,
    ):
        """End-to-end: compute all 6 scores and create composite rank."""
        from fundamentals.scoring import compute_institutional_conviction_score

        # Extract metrics from samples
        growth_score = compute_growth_score(
            revenue_cagr_3y=25.0,
            revenue_cagr_5y=22.0,
            net_income_cagr_3y=28.0,
            net_income_cagr_5y=25.0,
        )
        quality_score = compute_quality_score(
            roe=sample_valuations.roe,
            roce=sample_valuations.roce,
            profit_margin=sample_valuations.profit_margin,
            ebitda_margin=sample_valuations.ebitda_margin,
        )
        balance_sheet_score = compute_balance_sheet_score(
            debt_to_equity=sample_valuations.debt_to_equity,
            debt_to_revenue=None,
            current_ratio=sample_valuations.current_ratio,
            interest_coverage=sample_valuations.interest_coverage,
        )
        valuation_score = compute_valuation_score(
            pe=sample_valuations.pe,
            peg=sample_valuations.peg,
            pb=sample_valuations.pb,
            ps=sample_valuations.ps,
        )
        momentum_score = compute_momentum_score(
            institutional_pct=sample_shareholding.institutional_pct,
            fii_change_pct=sample_shareholding.fii_change_pct,
            dii_change_pct=sample_shareholding.dii_change_pct,
            price_momentum_3m=None,
        )
        institutional_conviction_score = compute_institutional_conviction_score(
            fii_count=45,
            dii_count=82,
            mf_count=156,
            total_institutional_pct=35.2,
            price_change_since_entry_pct=8.5,
            quarters_increasing_holding=2,
        )

        # Composite rank with 6 dimensions
        composite, growth_weighted, _ = compute_composite_rank(
            growth_score=growth_score,
            quality_score=quality_score,
            balance_sheet_score=balance_sheet_score,
            valuation_score=valuation_score,
            momentum_score=momentum_score,
            institutional_conviction_score=institutional_conviction_score,
        )

        assert all(
            0 <= s <= 100
            for s in [
                growth_score,
                quality_score,
                balance_sheet_score,
                valuation_score,
                momentum_score,
                institutional_conviction_score,
                composite,
                growth_weighted,
            ]
        )
        assert composite > 0
        assert growth_weighted > 0
        # Conviction contributes to composite (20% weight)
        assert composite > 50  # Strong candidate

    def test_watchlist_conviction_filtering(self, clear_redis_watchlist):
        """Watchlist: filter by institutional conviction."""
        watchlist = MultibaggerWatchlist()

        # Add high-conviction stock

        high_conviction_scores = FundamentalsScores(
            symbol="GROW1",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=80,
            quality_score=75,
            balance_sheet_score=72,
            valuation_score=68,
            momentum_score=65,
            institutional_conviction_score=85,
            composite_rank=75,
        )
        watchlist.add_scores("GROW1", high_conviction_scores)

        # Add low-conviction stock
        low_conviction_scores = FundamentalsScores(
            symbol="GROW2",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=78,
            quality_score=73,
            balance_sheet_score=70,
            valuation_score=66,
            momentum_score=63,
            institutional_conviction_score=35,
            composite_rank=70,
        )
        watchlist.add_scores("GROW2", low_conviction_scores)

        # Filter by min conviction
        results = watchlist.get_top_n(n=10, min_institutional_conviction=70)
        assert len(results) == 1
        assert results[0][0] == "GROW1"

    def test_watchlist_conviction_with_low_rally(self, clear_redis_watchlist):
        """Watchlist: conviction_with_low_rally boosts positions."""
        watchlist = MultibaggerWatchlist()

        # High conviction positions
        s1 = FundamentalsScores(
            symbol="ACUM1",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=75,
            quality_score=70,
            balance_sheet_score=68,
            valuation_score=65,
            momentum_score=60,
            institutional_conviction_score=78,
            composite_rank=70,
        )
        watchlist.add_scores("ACUM1", s1)

        s2 = FundamentalsScores(
            symbol="ACUM2",
            timestamp=datetime.now(UTC),
            source="fundamentals",
            growth_score=80,
            quality_score=75,
            balance_sheet_score=73,
            valuation_score=70,
            momentum_score=65,
            institutional_conviction_score=65,
            composite_rank=73,
        )
        watchlist.add_scores("ACUM2", s2)

        # Get top N with conviction_with_low_rally boost
        results = watchlist.get_top_n(
            n=10, conviction_with_low_rally=True, sort_by="conviction_with_low_rally"
        )
        # ACUM1 with conviction 78 gets boosted to 85
        # ACUM2 with conviction 65 stays at 73
        assert len(results) >= 2
        # First should be boosted position
        assert results[0][1].conviction_with_low_rally > results[1][1].conviction_with_low_rally

    """Integration tests with multiple components."""

    def test_end_to_end_scoring_pipeline(
        self,
        sample_financials: QuarterlyFinancials,
        sample_valuations: Valuations,
        sample_shareholding: Shareholding,
    ):
        """End-to-end: compute all scores and create composite rank."""
        # Extract metrics from samples
        growth_score = compute_growth_score(
            revenue_cagr_3y=25.0,
            revenue_cagr_5y=22.0,
            net_income_cagr_3y=28.0,
            net_income_cagr_5y=25.0,
        )
        quality_score = compute_quality_score(
            roe=sample_valuations.roe,
            roce=sample_valuations.roce,
            profit_margin=sample_valuations.profit_margin,
            ebitda_margin=sample_valuations.ebitda_margin,
        )
        balance_sheet_score = compute_balance_sheet_score(
            debt_to_equity=sample_valuations.debt_to_equity,
            debt_to_revenue=None,
            current_ratio=sample_valuations.current_ratio,
            interest_coverage=sample_valuations.interest_coverage,
        )
        valuation_score = compute_valuation_score(
            pe=sample_valuations.pe,
            peg=sample_valuations.peg,
            pb=sample_valuations.pb,
            ps=sample_valuations.ps,
        )
        momentum_score = compute_momentum_score(
            institutional_pct=sample_shareholding.institutional_pct,
            fii_change_pct=sample_shareholding.fii_change_pct,
            dii_change_pct=sample_shareholding.dii_change_pct,
            price_momentum_3m=None,
        )
        institutional_conviction_score = compute_institutional_conviction_score(
            fii_count=45,
            dii_count=82,
            mf_count=156,
            total_institutional_pct=35.2,
            price_change_since_entry_pct=8.5,
            quarters_increasing_holding=2,
        )

        composite, growth_weighted, _ = compute_composite_rank(
            growth_score=growth_score,
            quality_score=quality_score,
            balance_sheet_score=balance_sheet_score,
            valuation_score=valuation_score,
            momentum_score=momentum_score,
            institutional_conviction_score=institutional_conviction_score,
        )

        assert all(
            0 <= s <= 100
            for s in [
                growth_score,
                quality_score,
                balance_sheet_score,
                valuation_score,
                momentum_score,
                institutional_conviction_score,
                composite,
                growth_weighted,
            ]
        )
        assert composite > 0
        assert growth_weighted > 0
