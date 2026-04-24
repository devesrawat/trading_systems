# Fundamentals Package: Multibagger Discovery

Deterministic scoring and ranking system for identifying growth companies with capital appreciation potential.

## Architecture

```
Financial Data (Quarterly Filings)
       │
       ├─→ Growth Score (Revenue/Net Income CAGR)
       ├─→ Quality Score (ROE, ROCE, Profit Margins)
       ├─→ Balance Sheet Score (Debt ratios, Liquidity)
       ├─→ Valuation Score (PE/PEG, Enterprise Value)
       └─→ Momentum Score (Institutional ownership, Price trends)
       │
       ▼
Composite Multibagger Rank (0-100)
       │
       └─→ Watchlist with filtering and export
```

## Modules

### schema.py
Pydantic data models with validation:
- **QuarterlyFinancials**: Revenue, net income, EBITDA, FCF, balance sheet metrics
- **Valuations**: P/E, P/B, P/S, PEG, profitability ratios (ROE, ROCE)
- **Shareholding**: Promoter/institutional/public ownership, FII/DII positions
- **FundamentalsScores**: Individual and composite scores (0-100)
- **ConfidenceLevel**: data source confidence (high/medium/low)

All with `source`, `timestamp`, and `confidence_level` for data provenance.

### scoring.py
Deterministic, formula-based scoring (no ML):

#### Growth Score (0-100)
```
Formula: Revenue CAGR sigmoid + Net Income CAGR sigmoid (average)
Thresholds:
  - 3-year CAGR weighted at 60%, 5-year at 40%
  - Revenue: 0-40% maps to 0-100 (sigmoid)
  - Net income: 0-35% maps to 0-100 (higher growth bar)
```

#### Quality Score (0-100)
```
Formula: Average of ROE, ROCE, Profit margin, EBITDA margin
Sigmoid calibration:
  - ROE: 15% benchmark, 25% excellent
  - ROCE: 15% benchmark, 25% excellent
  - Profit margin: 8% inflection point
  - EBITDA margin: 15% inflection point
Penalty: Consistency score <80% reduces score by up to 20 points
```

#### Balance Sheet Score (0-100)
```
Formula: Inverted sigmoid for debt (lower better) + Gaussian for liquidity + coverage sigmoid
Metrics:
  - Debt/Equity: <1.0 healthy, >2.0 risky (inverted sigmoid)
  - Debt/Revenue: <2.0 healthy
  - Current Ratio: Gaussian centered at 1.5 (ideal)
  - Interest Coverage: >2.5x is safe (sigmoid)
```

#### Valuation Score (0-100)
```
Formula: Weighted combination of PEG, PE, PB, PS
Weights: PEG (60%), PE (25%), PB+PS (15%)
PEG Sigmoid: <1.2 undervalued, >1.5 overvalued
PE: 15-30 is fair, >40 is expensive
```

#### Momentum Score (0-100)
```
Formula: Institutional ownership + FII/DII buying + price momentum
Signals:
  - Institutional >30% positive
  - FII/DII change: positive buying is good
  - Price momentum >10% in 3M awards +80 points
```

### ranking.py
Composite ranking from individual scores:

**Default Weights** (optimized for multibagger discovery):
- Growth: 35% (primary signal)
- Quality: 25% (earnings stability)
- Balance sheet: 20% (financial health)
- Valuation: 15% (entry point)
- Momentum: 5% (confirmation)

**Growth-Weighted Rank** (for aggressive portfolios):
- Growth: 50%
- Quality: 20%
- Balance sheet: 15%
- Valuation: 10%
- Momentum: 5%

**Percentile Calculation**: Score position in universe (0-100)

### providers.py
Abstract base for data sources (skeleton implementations, no API calls yet):

- **NSEProvider**: NSE filings (quarterly data, shareholding patterns)
- **ScreenerProvider**: Screener.in API (valuations, momentum, quality metrics)
- **TrendlyneProvider**: Trendlyne API (normalized metrics, score consistency)

Each provider implements:
- `fetch_financials(symbol)` → QuarterlyFinancials or None
- `fetch_valuations(symbol)` → Valuations or None
- `fetch_shareholding(symbol)` → Shareholding or None

### ingest.py
Data fetching and caching:

- **FundamentalsData**: Container for all fundamentals for a symbol
  - `is_complete()`: All three data types available
  - `completeness_ratio()`: Fraction of available data (0-1)

- **fetch_fundamentals(symbol, providers)**: Multi-provider fallback
  - Tries each provider in order
  - Returns first successful result for each metric type
  - Best-effort combined data

- **get_cached(symbol, max_age_days, providers)**: Cache-first fetching
  - Checks Redis cache (TTL: 7 days default)
  - Falls back to fresh fetch if stale or missing
  - Returns partial cached data if fetch unavailable

- **store_in_cache()**: Persist to Redis with TTL

- **store_in_db()**: Placeholder for TimescaleDB persistence

### watchlist.py
In-memory watchlist with Redis backing:

- **MultibaggerWatchlist** class
  - `add_scores(symbol, scores)`: Add or update
  - `remove(symbol)`: Remove from watchlist
  - `is_watchlisted(symbol)`: Check membership
  - `get_top_n(n=20, filters={...})`: Sorted by composite rank with filtering
    - min_growth, min_quality, min_balance_sheet, min_composite
  - `update_scores(symbol, ...)`: Recompute composite rank and percentile
  - `export_to_csv(path)`: Export for analysis

Redis persistence:
- Key: `trading:fundamentals:watchlist`
- TTL: 30 days
- Auto-load on creation

## Usage Example

```python
from fundamentals import (
    compute_growth_score, compute_quality_score, compute_balance_sheet_score,
    compute_valuation_score, compute_momentum_score,
    compute_composite_rank, MultibaggerWatchlist
)

# Compute individual scores
growth = compute_growth_score(
    revenue_cagr_3y=25.0, revenue_cagr_5y=22.0,
    net_income_cagr_3y=28.0, net_income_cagr_5y=25.0
)
quality = compute_quality_score(roe=18.5, roce=22.3, profit_margin=18.0)
balance_sheet = compute_balance_sheet_score(debt_to_equity=0.167, current_ratio=1.5)
valuation = compute_valuation_score(pe=25.5, peg=1.2, pb=4.2)
momentum = compute_momentum_score(institutional_pct=35.2, fii_change_pct=2.5)

# Compute composite rank
composite, growth_weighted = compute_composite_rank(
    growth, quality, balance_sheet, valuation, momentum
)

# Add to watchlist
watchlist = MultibaggerWatchlist()
watchlist.update_scores("INFY", growth, quality, balance_sheet, valuation, momentum)

# Get top candidates
top_20 = watchlist.get_top_n(n=20, min_growth=70)

# Export for analysis
watchlist.export_to_csv("watchlist.csv")
```

## Data Completeness

Scores adapt gracefully to missing data:
- If some metrics missing, score computed from available ones
- `data_completeness` field tracks fraction of data used (0-1)
- Confidence level indicates source reliability (high/medium/low)
- Watchlist can filter by min_completeness if desired

## Calibration Notes

All sigmoids are calibrated for **Indian small-cap multibaggers**:
- Growth thresholds favor 20%+ CAGR companies
- Quality bar (ROE/ROCE 15%+) reflects small-cap performance
- Valuation allows for premium on high-growth (PEG <1.2)
- Momentum weights institutional ownership (DII index buying)
- Penalty for low ownership (<5%) to avoid illiquid penny stocks

## Testing

500+ lines of comprehensive tests in `tests/test_fundamentals.py`:

- **Schema validation**: Edge cases, bounds checking, optional fields
- **Scoring formulas**: Each scorer tested with known inputs
- **Ranking**: Composite weights, percentile calculation, custom weights
- **Watchlist**: Add, filter, update, export operations
- **Caching**: TTL behavior, partial cache hits, fallbacks
- **Integration**: End-to-end pipeline with all scorers

Run tests:
```bash
uv run pytest tests/test_fundamentals.py -v
```

## Future Work (Phase 5)

- Implement real API calls to NSE, Screener, Trendlyne (currently skeleton)
- TimescaleDB persistence of historical fundamentals (for trend analysis)
- Sector-relative scoring (compare within sector, not absolute)
- Analyst coverage tracking and estimate revisions
- Earnings quality scoring (accruals analysis)
- Integration with orchestrator for continuous rescoring
- Telegram alerts for new watchlist entries
- Portfolio manager consumption of top multibagger candidates

## Notes

- All scoring is deterministic (reproducible, no randomness)
- Formulas documented inline with sigmoid inflection points calibrated
- Zero ML models in fundamentals layer (reserved for signals/model.py)
- Data freshness tracked (7-day cache TTL, can be overridden)
- Designed for Phase 5: Portfolio Risk Manager consumption
