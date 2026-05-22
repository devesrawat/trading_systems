# Multi-Strategy Alpha Engine Implementation Plan

## Background & Motivation
The current trading system relies on standard technical indicators and news sentiment to feed an XGBoost model. While robust, these "public alpha" sources often suffer from noise and false breakouts. To elevate the system to high profitability, we need to incorporate "institutional footprints." This plan introduces a Multi-Strategy Alpha Engine that layers Sector Relative Strength, Options Flow (OI/PCR), and Market Regimes to calculate an `Alpha Multiplier`. This multiplier adjusts the baseline XGBoost probability, filtering out low-probability trades and scaling up high-conviction setups.

## Scope & Impact
*   **New Modules**: `signals/sector_alpha.py`, `signals/options_alpha.py`, `signals/alpha_composite.py`.
*   **Modified Modules**: `orchestrator/main.py`, `risk/sizer.py`, `signals/features.py`.
*   **Impact**: Significantly improves the Win Rate and Profit Factor by eliminating trades that fight the broader sector or institutional options flow.

## Proposed Solution: The 4-Component Architecture

1.  **Sector Relative Strength (The Macro Filter)**
    *   Calculate 5-day and 20-day Relative Strength (RS) of NSE sectoral indices (Nifty Bank, IT, Auto, etc.) against the Nifty 50.
    *   Rank sectors. Long trades are only permitted or boosted in Top 3 sectors; short trades in Bottom 3 sectors.

2.  **Options Flow (The Institutional Footprint)**
    *   Track Intraday Open Interest (OI) changes and Put-Call Ratio (PCR).
    *   A rising PCR (put selling) + rising OI indicates bullish institutional positioning.

3.  **Regime-Aware Tactical Pivot**
    *   Integrate with existing `signals/regime.py` (HMM).
    *   High Volatility: Favor Momentum (breakouts).
    *   Low Volatility: Favor Mean Reversion (pullbacks).

4.  **Composite Alpha Multiplier**
    *   Combine the above into a scalar value (e.g., `0.5` to `1.5`).
    *   `Final_Probability = XGBoost_Probability * Alpha_Multiplier`.

## Implementation Steps

### Phase 1: Data Integration & Feature Building
1.  **Create `signals/sector_alpha.py`**:
    *   Implement `SectorRanker` class.
    *   Method to fetch historical OHLCV for sector indices (e.g., `NIFTY BANK`, `NIFTY IT`).
    *   Method to calculate and cache RS scores and rank sectors daily.
2.  **Create `signals/options_alpha.py`**:
    *   Implement `OptionsFlowAnalyzer` class.
    *   Fetch near-month expiry OI and PCR for the specific stock and its parent index.
    *   Calculate `oi_change_pct` and `pcr_trend`.

### Phase 2: The Composite Engine
3.  **Create `signals/alpha_composite.py`**:
    *   Implement `AlphaEngine` class.
    *   Method `calculate_multiplier(symbol, sector, current_regime)`:
        *   Base multiplier = 1.0
        *   If sector in Top 3: +0.2 (for longs)
        *   If options flow bullish: +0.2
        *   If strategy mismatch with regime (e.g., mean reversion in high vol): -0.3
        *   Return bounded multiplier `[0.5, 1.5]`.

### Phase 3: Orchestrator & Risk Integration
4.  **Update `orchestrator/main.py`**:
    *   During the trading loop, after fetching XGBoost `signal_prob`, call `AlphaEngine.calculate_multiplier()`.
    *   Calculate `final_prob = signal_prob * multiplier`.
    *   Apply `SIGNAL_THRESHOLD` against `final_prob` instead of the raw model score.
5.  **Update `risk/sizer.py`**:
    *   Pass the `alpha_multiplier` into the `PositionSizer`.
    *   Increase Kelly fraction for high multiplier (high institutional conviction), shrink for low multiplier.

### Phase 4: Testing & Validation
6.  **Unit Tests**:
    *   Add tests for `SectorRanker` and `OptionsFlowAnalyzer` mocking Kite API responses.
    *   Add logic tests for `AlphaEngine` multiplier boundaries.
7.  **Walk-Forward Backtest**:
    *   Run `backtest/walk_forward.py` with the new Alpha Engine enabled.
    *   **Success Criteria**: Aggregate Sharpe ratio increases to > 1.5, Profit Factor > 2.0.

## Migration & Rollback
*   The Alpha Engine will be feature-flagged via `.env` (`ENABLE_ALPHA_ENGINE=true`).
*   If live metrics drop below backtest thresholds, we can seamlessly toggle it off and revert to raw XGBoost probabilities.
