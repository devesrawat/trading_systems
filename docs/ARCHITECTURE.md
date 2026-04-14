# NSE/Crypto Algorithmic Trading System — Architecture v2.0

> **Audience**: Senior developer working alone. This doc is the single source of truth.
> No need to read source code for 80% of development decisions.
>
> Last updated: 2026-04-14. Budget target: ₹3,000–₹5,000/month total (infra + API + LLM).

---

## Table of Contents

1. [Architecture Audit — Current State vs Target](#1-architecture-audit)
2. [Research Synthesis R1–R6](#2-research-synthesis)
3. [System Architecture Overview](#3-system-architecture-overview)
4. [Module Designs M1–M10 with Python Skeletons](#4-module-designs)
5. [Tech Stack Decision Table](#5-tech-stack-decisions)
6. [Monthly Cost Breakdown](#6-monthly-cost-breakdown)
7. [Phased Implementation Roadmap](#7-phased-roadmap)
8. [What Separates World-Class from Mediocre](#8-world-class-vs-mediocre)
9. [Data Contracts & Schemas](#9-data-contracts)
10. [Redis Key Namespace](#10-redis-keys)
11. [Database Schema](#11-database-schema)
12. [Non-Negotiable Constraints](#12-non-negotiable-constraints)

---

## 1. Architecture Audit

### 1.1 Strengths (Keep)

| # | What | Why Keep |
|---|------|----------|
| 1 | **OHLCVProvider ABC** (`data/providers/base.py`) | Provider-agnostic; Kite/Upstox/Binance swap trivially |
| 2 | **BrokerAdapter ABC + factory** (`execution/broker.py`) | Paper/live routing clean; paper mode forced when `paper_trade_mode=True` |
| 3 | **36-feature XGBoost pipeline** (`signals/features.py`) | Walk-forward trained, Optuna-tuned; 252-bar warmup; solid baseline |
| 4 | **CircuitBreaker with Redis persistence** (`risk/breakers.py`) | Manual-reset-only halt; daily/weekly DD + consecutive loss limits |
| 5 | **pydantic-settings config** (`config/settings.py`) | All secrets from .env; `max_position_pct <= 2%` enforced at parse time |
| 6 | **APScheduler cron orchestration** (`orchestrator/scheduler.py`) | Separate IST equity vs UTC crypto jobs; clean separation |
| 7 | **TimescaleDB hypertable + SQLAlchemy pool** | Right storage primitive for time-series; pool_size=10 sufficient for VPS |
| 8 | **NSECostModel** (`backtest/cost_model.py`) | All 7 Zerodha fee components modeled; critical for realistic backtest |
| 9 | **Prometheus + Grafana monitoring** | Gauges for P&L, drawdown, position count already wired |
| 10 | **FinBERT sentiment** (`llm/sentiment.py`) | ProsusAI/finbert; score in [-1,+1]; 1h TTL in Redis — correct architecture |

### 1.2 Critical Bugs (Fix First)

| # | Location | Bug | Fix |
|---|----------|-----|-----|
| B1 | `orchestrator/main.py:336` | `current_price = float(last_row["ema_50"].iloc[0])` — uses EMA as price proxy | Replace with `last_row["close"].iloc[0]` |
| B2 | `orchestrator/main.py:_execute_crypto_signal()` | Only logs tick; no XGBoost, no signal, no Telegram | Implement fully (Phase 2) |
| B3 | `signals/vcp_scanner.py` | VCP scanner implemented but NOT wired into `TradingSystem` | Add to `pre_market_setup()` call chain |
| B4 | `execution/broker.py:get_broker_adapter()` | Binance/unknown broker falls through to Paper silently | Raise explicit error or add Binance adapter |

### 1.3 Architectural Gaps (Add in Phases)

| # | Gap | Impact | Phase |
|---|-----|---------|-------|
| G1 | No regime detection — signal quality blind to market state | Generates signals in choppy/bear markets; drawdowns | 2 |
| G2 | No FII/DII flow ingestion | Missing the single biggest institutional signal for NSE | 2 |
| G3 | No options chain parsing for IV/PCR | F&O signals are placeholder (Greeks computed but no chain data) | 3 |
| G4 | No correlation-aware portfolio sizing | Individual position sizing ignores existing correlated positions | 2 |
| G5 | No on-chain crypto metrics (funding rate, OI, fear/greed) | Crypto signals are purely technical; missing perp market microstructure | 3 |
| G6 | LLM used only for sentiment; no macro summary synthesis | FinBERT runs per-article; no daily macro briefing for position bias | 3 |
| G7 | No Telegram command interface | All control is code/manual; no `/pause`, `/status`, `/portfolio` via Telegram | 2 |
| G8 | Walk-forward retrain runs only weekly; no concept drift detection | Model degrades silently between retrains | 3 |

### 1.4 Verdict on Current Stack

The foundation is sound. The core loop (fetch → features → XGBoost → size → risk check → Telegram) works. Phase 1 is just bug fixes + three missing wires (VCP, crypto signal, price proxy). Everything else is additive.

---

## 2. Research Synthesis

### R1: Time-Series Models for NSE Equity Forecasting

**Opinionated verdict: Stay with XGBoost. Add TFT only in Phase 3 if XGBoost Sharpe < 1.2 in live paper trading.**

| Model | Accuracy vs XGBoost | Infra Cost | Verdict |
|-------|---------------------|-----------|---------|
| **XGBoost/LightGBM** | Baseline (wins on stock-specific data) | CPU-only, <100MB RAM | **PRIMARY. Keep.** |
| **TFT (Temporal Fusion Transformer)** | Within 5% of XGBoost when you have known future covariates (earnings dates, macro events) | GPU preferred; 2GB VRAM | **Phase 3 experiment on F&O** |
| **PatchTST** | Matches XGBoost with 1,000+ training points; not zero-shot | GPU, 4GB VRAM | Skip for now |
| **Chronos (Amazon T5-based)** | Zero-shot good; beats naive baselines; loses to tuned XGBoost by 10-20% on directional accuracy | CPU ok, 1-4GB RAM | **Use for cold-start stocks < 6mo history only** |
| **TimesFM 2.0 (Google)** | Comparable to Chronos; covariate support added 2025 | CPU/GPU, 2-8GB | Skip; Chronos sufficient for cold-start |
| **N-HiTS / N-BEATS** | Strong on seasonality; poor on financial series | CPU ok | Skip |
| **LLM direct price forecasting** | Generally terrible; multiple NeurIPS 2024 papers confirm | Expensive | Never |

**Key finding (NeurIPS 2024, "Are LLMs Useful for Time Series?")**: LLM-based time series models fail to beat simple baselines including ARIMA on financial data. XGBoost with engineered features wins consistently. Use LLMs **only** for natural language signals (news, announcements), not for price prediction.

**Implementation path**:

- Phase 1-2: XGBoost (current) with 36 features, walk-forward, Optuna
- Phase 3: Add Chronos as cold-start fallback for stocks with <252 bars
- Phase 3: A/B test TFT vs XGBoost on F&O with earnings dates as covariates

### R2: LLM Strategy for Indian Markets

**Opinionated verdict: Groq API with Llama-3.1-8B for real-time news; FinBERT local for per-article scoring.**

What LLMs are actually useful for in this system:

1. News summarization + sentiment — "Is this earnings announcement bullish or bearish?"
2. Daily macro briefing — Synthesize FII/DII flows + global cues into 3-sentence position bias
3. Options commentary — Interpret IV skew + PCR into plain English for Telegram
4. Corporate announcement parsing — Extract key facts from NSE PDF announcements

What LLMs are NOT useful for:

- Direct price prediction (see R1)
- Technical indicator interpretation (rule-based is better)
- Real-time tick processing (latency + cost prohibitive)

**FinGPT vs FinBERT for Indian markets**: No FinBERT variant fine-tuned specifically on NSE/Indian financial news exists as a published open-source model as of 2025. The existing system's use of ProsusAI/finbert is the right call. Keep it.

### R3: Inference Cost Ranking

| Provider | Model | Input $/1M | Output $/1M | INR/1M (avg) | Notes |
|----------|-------|-----------|------------|-------------|-------|
| **Groq** | Llama-3.1-8B-Instant | $0.05 | $0.08 | INR 5.5 | **Cheapest. Free tier: 30 RPM, 30K TPD** |
| **Groq** | Llama-3.3-70B-Versatile | $0.59 | $0.79 | INR 58 | For complex synthesis only |
| **Gemini 2.0 Flash** | Flash | $0.075 | $0.30 | INR 16 | 1M free tokens/day via AI Studio |
| **Cerebras** | Llama-3.1-8B | $0.10 | $0.10 | INR 8.4 | 2,000 tokens/sec throughput; best for latency |
| **Together AI** | Llama-3.1-8B-Turbo | $0.18 | $0.18 | INR 15 | Flat pricing; fine-tuning available |
| **Fireworks AI** | Llama-3.1-8B | $0.20 | $0.20 | INR 17 | Serverless; dedicated cheaper at scale |
| **Ollama local** | Llama-3.1-8B (Q4) | $0 | $0 | INR 0 | Needs 8GB VRAM or 16GB RAM; ~5-15 tok/s on CPU |
| **Anthropic Claude** | Haiku 4.5 | $0.80 | $4.00 | INR 200 | Too expensive for bulk news processing |

**Decision**: Use Groq free tier (Llama-3.1-8B-Instant) for daily macro synthesis. Estimated tokens/day: ~5,000 input + 2,000 output = well within 30K TPD free limit. Total LLM cost: ~INR 200-400/month.

### R4: Free/Cheap Data Sources

#### NSE Equity Data (2025)

| Source | Type | Cost | Notes |
|--------|------|------|-------|
| **yfinance** (`RELIANCE.NS`) | EOD OHLCV, unlimited | Free | Most reliable for backtesting; 30-day intraday |
| **jugaad-data** | EOD bhavcopy, options chain, index | Free | Best maintained NSE library 2025; handles cookies |
| **NSE bhavcopy** (direct) | EOD CSV, all instruments | Free | `archives.nseindia.com`; requires Referer header spoof |
| **NSE FII/DII API** | Daily flows JSON | Free | `nseindia.com/api/fiidiiTradeReact`; session cookie required |
| **Angel One SmartAPI** | Live quotes, historical 1yr, F&O chain | Free with demat | Best free option with broker account; good Python SDK |
| **Upstox API v2** | 2yr historical, WebSocket live | Free with demat | 250 req/hr; already integrated |
| **Zerodha Kite** | 3yr historical, live, F&O | INR 2,000/month | Already integrated; most reliable for production |
| **nsepy / nsetools** | Dead / partially broken | — | Do not use |

NSE FII/DII fetch pattern:

```python
import requests
s = requests.Session()
s.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer": "https://www.nseindia.com",
})
s.get("https://www.nseindia.com", timeout=10)  # establish cookies
r = s.get("https://www.nseindia.com/api/fiidiiTradeReact")
data = r.json()
```

NSE Options Chain via jugaad-data:

```python
from jugaad_data.nse import NSELive
n = NSELive()
oc = n.option_chain("NIFTY")
# Fields: strikePrice, CE.openInterest, PE.openInterest,
#         CE.impliedVolatility, PE.impliedVolatility
```

#### Crypto Data (2025)

| Source | Type | Cost | Notes |
|--------|------|------|-------|
| **Binance REST/WS** | OHLCV, orderbook, funding rates | Free | Already integrated; most comprehensive |
| **CoinGecko API** | Price, market cap, volume | Free (50 req/min) | Already integrated |
| **Alternative.me** | Fear & Greed Index | Free | Simple JSON API; no auth: `api.alternative.me/fng/` |
| **CoinGlass free tier** | Funding rates, OI, liquidations | Free (50 req/day) | BTC/ETH/SOL data available |
| **Glassnode free tier** | Price, market cap, supply only | Free | NVT/NUPL/exchange flows require paid tier ($29/mo) |

Practical crypto stack: Binance + CoinGecko + Alternative.me covers 90% of needs for free.

#### Indian Equity News (2025)

| Source | Type | Cost |
|--------|------|------|
| **Economic Times RSS** | Headlines | Free |
| **MoneyControl RSS** | Headlines | Free |
| **NSE Corporate Announcements** | PDF/JSON | Free (session cookie) |
| **ProsusAI/finbert** | Sentiment scoring | Free (local) |

No FinBERT model fine-tuned on Indian financial news exists as open-source (2025). ProsusAI/finbert is the best available free option.

### R5: Regime Detection

**Opinionated verdict: Multi-indicator composite with ADX gate. Skip pure HMM/BOCPD for swing trading.**

| Method | Swing Trading Fit | Verdict |
|--------|------------------|---------|
| **HMM (hmmlearn)** | Poor — too much lag (1-5 bar state assignment delay) | Skip for live; ok for labeling backtest data |
| **BOCPD** | Moderate — good for vol regime changes; complex to tune | Skip for v1 |
| **PELT (ruptures)** | Poor for live — offline only, needs future data to confirm | Use for backtest analysis only |
| **ADX + VIX + Breadth composite** | Best for swing traders | **USE THIS** |
| **Volatility KMeans (3-state)** | Useful as one component | Include as sub-signal |

Recommended 4-factor composite:

- ADX(14) — trend strength (>25 = trending, <20 = choppy)
- 20d realized vol vs 252d vol ratio (expansion/contraction)
- Nifty50 200-DMA slope (positive = uptrend, negative = downtrend)
- India VIX level (<12 = complacent, 12-20 = normal, >20 = fear)

Regime states:

- `TRENDING_BULL`: ADX>25 AND Nifty above 200DMA AND VIX<20
- `TRENDING_BEAR`: ADX>25 AND Nifty below 200DMA AND VIX>20
- `CHOPPY`: ADX<20 (regardless of direction) — NO NEW ENTRIES
- `HIGH_VOL`: Realized vol > 1.5x historical vol OR VIX > 25 — HALF SIZE

### R6: Position Sizing

**Opinionated verdict: Volatility-normalized half-Kelly with correlation penalty. Keep the existing formula; add correlation.**

The existing half-Kelly is correct. Add two enhancements:

1. Vol scaling: multiply Kelly fraction by `min(hist_vol / current_vol, 1.5)` — automatically reduces size in high-vol regimes
2. Correlation penalty: same-sector positions reduce new position size by 0.2 per existing same-sector position, capped at 0.4

For INR 5L–50L account with 3–5 simultaneous positions: half-Kelly with vol scaling and 2% hard cap produces mean position size of 1.2–1.6% of capital. Maximum portfolio heat = 8–10%.

---

## 3. System Architecture Overview

```
                              ┌─────────────────────────────────────┐
                              │         DATA LAYER (M1)             │
                              │  Kite/Upstox | Binance | jugaad-data│
                              │  NSE FII/DII | CoinGlass | RSS News │
                              └──────────────────┬──────────────────┘
                                                 │ MarketSnapshot
                              ┌──────────────────▼──────────────────┐
                              │       FEATURE LAYER (M2, M3)        │
                              │  FeatureEngine (36+4 features)      │
                              │  RegimeDetector (ADX+VIX+Vol+DMA)   │
                              └──────────────────┬──────────────────┘
                                                 │ feature_df + RegimeMetrics
                    ┌────────────────────────────▼──────────────────────────┐
                    │                  SIGNAL LAYER (M4, M5)               │
                    │  XGBoost classifier | VCP Scanner | FinBERT+Groq LLM │
                    │  SignalAggregator (regime-gated, sentiment-filtered)  │
                    └────────────────────────────┬──────────────────────────┘
                                                 │ list[TradingSignal]
                    ┌────────────────────────────▼──────────────────────────┐
                    │                   RISK LAYER (M6, M7, M8)            │
                    │  PositionSizer (half-Kelly + vol + correlation)       │
                    │  RiskGateway (CircuitBreaker + duplicate + size)      │
                    │  PortfolioMonitor (Redis-persisted P&L + drawdown)    │
                    └────────────────────────────┬──────────────────────────┘
                                                 │ GatewayResult
                              ┌──────────────────▼──────────────────┐
                              │      EXECUTION LAYER (M9)           │
                              │  TelegramSignalBot                  │
                              │  /status /portfolio /pause /resume  │
                              └──────────────────┬──────────────────┘
                                                 │ Telegram channel
                              ┌──────────────────▼──────────────────┐
                              │        BACKTEST (M10)               │
                              │  WalkForwardEngine (24mo train/3mo) │
                              │  NSECostModel | MLflow Registry     │
                              └─────────────────────────────────────┘
```

Signal flow sequence (09:20-09:45 IST daily):

```
APScheduler → DataIngestion (fetch 500 symbols + FII/DII + news)
           → FeatureEngine (compute 36+4 features)
           → RegimeDetector (classify TRENDING_BULL/BEAR/CHOPPY/HIGH_VOL)
           → If CHOPPY: return [] immediately
           → LLMSentimentEngine (FinBERT score news articles)
           → SignalEngine (XGBoost predict_proba > 0.55 + VCP scan)
           → Sentiment filter (reject if sentiment < -0.3)
           → PositionSizer (half-Kelly + vol scaling + correlation penalty)
           → RiskGateway (CircuitBreaker check + duplicate check + size cap)
           → TelegramSignalBot (format + send; or HALTED notification)
```

---

## 4. Module Designs

### M1: DataIngestionPipeline

**Purpose**: Fetch, validate, and normalize all market data into uniform DataFrames.
**Inputs**: Scheduler trigger, symbol universe, date range
**Outputs**: `MarketSnapshot` dataclass; OHLCV DataFrames into TimescaleDB + Redis tick cache
**Run frequency**: Pre-market (09:15 IST), mid-session (12:00), post-market (15:35), crypto 24/7 every 5min
**Memory**: ~200MB peak (500-symbol bulk fetch)

```python
# data/ingestion.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
import pandas as pd
from typing import Literal


class OHLCVProvider(ABC):
    """Abstract base for all market data providers."""

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str = "day",
    ) -> pd.DataFrame:
        """Return DataFrame with columns: date, open, high, low, close, volume."""

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """Return current quote dict with ltp, open, high, low, volume."""

    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: date | None = None) -> pd.DataFrame:
        """Return options chain with CE/PE columns for each strike."""


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable snapshot of all data collected in one ingestion run."""
    timestamp: datetime
    ohlcv: dict[str, pd.DataFrame]          # symbol -> OHLCV DataFrame
    quotes: dict[str, dict]                  # symbol -> live quote
    option_chains: dict[str, pd.DataFrame]   # symbol -> chain (index symbols only)
    fii_dii: dict | None                     # today's FII/DII flows
    regime_inputs: dict                      # {vix, nifty_200dma_slope, adx}
    news_items: list[dict]                   # [{headline, source, symbol, timestamp}]
    crypto_metrics: dict                     # {btc_fear_greed, funding_rates, ...}


class NSEDataScraper:
    """
    Fetches NSE-specific data not available through broker APIs:
    FII/DII daily flows, India VIX, NSE corporate announcements.
    """

    BASE = "https://www.nseindia.com"
    _session = None

    def _get_session(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": self.BASE,
            })
            s.get(self.BASE, timeout=10)  # establish cookies
            self._session = s
        return self._session

    def get_fii_dii_flows(self) -> dict:
        """
        Returns:
          fii_net_cash: INR crore (negative = net selling)
          dii_net_cash: INR crore (positive = net buying)
          fii_net_fno:  INR crore F&O net
          date: "YYYY-MM-DD"
        """
        session = self._get_session()
        resp = session.get(f"{self.BASE}/api/fiidiiTradeReact", timeout=10)
        resp.raise_for_status()
        latest = resp.json()[0]
        return {
            "fii_net_cash": float(latest.get("netval_fii_cash", 0)),
            "dii_net_cash": float(latest.get("netval_dii_cash", 0)),
            "fii_net_fno": float(latest.get("netval_fii_fno", 0)),
            "date": latest.get("date", ""),
        }

    def get_india_vix(self) -> float:
        """Returns current India VIX value."""
        session = self._get_session()
        resp = session.get(f"{self.BASE}/api/allIndices", timeout=10)
        resp.raise_for_status()
        for idx in resp.json().get("data", []):
            if idx.get("indexSymbol") == "INDIA VIX":
                return float(idx["last"])
        raise ValueError("India VIX not found in NSE API response")


class CryptoMetricsCollector:
    """Collects on-chain and derivatives metrics for BTC/ETH/SOL."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/"

    def get_fear_greed(self) -> int:
        """Returns 0-100 index. 0=extreme fear, 100=extreme greed."""
        import requests
        resp = requests.get(self.FEAR_GREED_URL, timeout=5)
        return int(resp.json()["data"][0]["value"])

    def get_binance_funding_rate(self, symbol: str) -> float:
        """Returns current 8h funding rate from Binance futures (free API)."""
        import requests
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": f"{symbol}USDT", "limit": 1},
            timeout=5,
        )
        data = resp.json()
        return float(data[0]["fundingRate"]) if data else 0.0

    def get_binance_open_interest(self, symbol: str) -> float:
        """Returns open interest in USDT from Binance futures (free API)."""
        import requests
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/openInterest",
            params={"symbol": f"{symbol}USDT"},
            timeout=5,
        )
        return float(resp.json()["openInterest"])
```

---

### M2: FeatureEngine

**Purpose**: Transform raw OHLCV + auxiliary data into the 40-column ML feature matrix.
**Inputs**: OHLCV DataFrame (252+ bars), optional FII/DII dict, optional sentiment score
**Outputs**: Feature DataFrame with exactly `FEATURE_COLUMNS` (36 base + 4 auxiliary)
**Run frequency**: Pre-market for equity; per 5min for crypto
**Memory**: ~50MB for 500 symbols

Current 36 base features (from `signals/features.py`):

- Momentum (9): roc_5, roc_20, rsi_14, macd, macd_signal, macd_hist, mom_10, mom_20, stoch_k
- Volatility (8): atr_14, atr_20, bb_width, bb_pct, hist_vol_20, hist_vol_60, range_pct, vol_ratio
- Volume (4): vol_ma_ratio, obv_slope, vwap_ratio, cmf_20
- Trend (9): ema_9_slope, ema_21_slope, ema_50_slope, ema_200_slope, adx_14, plus_di, minus_di, aroon_up, aroon_down
- Mean Reversion (3): zscore_20, dist_52w_high, dist_200dma
- Regime (1): vol_regime_kmeans (3-state KMeans on realized vol)

New 4 auxiliary features to add (Phase 1):

```python
AUXILIARY_FEATURES = [
    "fii_net_cash_norm",   # FII net cash / 30d avg abs flow; range ~[-3, +3]
    "india_vix",           # raw VIX value
    "sentiment_score",     # FinBERT score in [-1, +1]; 0.0 if no news
    "regime_code",         # 0=CHOPPY, 1=TRENDING_BULL, 2=TRENDING_BEAR, 3=HIGH_VOL
]

def build_features(
    df: pd.DataFrame,
    fii_dii: dict | None = None,
    india_vix: float | None = None,
    sentiment_score: float = 0.0,
    regime_code: int = 0,
    include_labels: bool = False,
) -> pd.DataFrame:
    """
    Returns a NEW DataFrame. Never mutates input df.
    Warmup: 252 bars minimum before first valid feature row.
    Label: forward_return_5d > 2% = 1, else 0.
    """
    result = _compute_technical_features(df.copy())  # existing 36-feature logic

    result = result.assign(
        fii_net_cash_norm=_normalize_fii(fii_dii["fii_net_cash"]) if fii_dii else 0.0,
        india_vix=india_vix if india_vix is not None else 18.0,
        sentiment_score=sentiment_score,
        regime_code=float(regime_code),
    )

    if include_labels:
        result = result.assign(
            label=(result["close"].shift(-5) / result["close"] - 1 > 0.02).astype(int)
        )

    return result.dropna()
```

---

### M3: RegimeDetector

**Purpose**: Classify current market regime. Gate signal generation based on regime state.
**Inputs**: Nifty50 OHLCV (252 bars), India VIX float
**Outputs**: `RegimeState` enum + `RegimeMetrics` dataclass; written to Redis with 1h TTL
**Run frequency**: Daily at 09:25 IST
**Memory**: <10MB

```python
# signals/regime.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from ta.trend import ADXIndicator


class RegimeState(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    CHOPPY = "choppy"
    HIGH_VOL = "high_vol"


@dataclass(frozen=True)
class RegimeMetrics:
    state: RegimeState
    adx_14: float
    vix: float
    nifty_200dma_slope: float    # % change of 200-DMA over 20 days
    realized_vol_ratio: float    # 20d realized vol / 252d realized vol
    score: float                 # 0.0-1.0 confidence in classification


class RegimeDetector:
    """
    Multi-factor regime classification.
    Thresholds tuned on Nifty 2015-2025.
    """

    ADX_TREND_THRESHOLD = 25.0
    ADX_CHOPPY_THRESHOLD = 20.0
    VIX_HIGH_THRESHOLD = 22.0
    VOL_EXPANSION_RATIO = 1.5

    def detect(self, nifty_df: pd.DataFrame, india_vix: float) -> RegimeMetrics:
        """nifty_df must have [open, high, low, close, volume]; minimum 252 bars."""
        adx = self._compute_adx(nifty_df)
        vol_ratio = self._compute_vol_ratio(nifty_df)
        dma_slope = self._compute_200dma_slope(nifty_df)
        state = self._classify(adx, india_vix, vol_ratio, dma_slope)
        score = self._compute_confidence(adx, india_vix, vol_ratio)
        return RegimeMetrics(
            state=state, adx_14=adx, vix=india_vix,
            nifty_200dma_slope=dma_slope, realized_vol_ratio=vol_ratio, score=score,
        )

    def get_position_size_multiplier(self, state: RegimeState) -> float:
        return {
            RegimeState.TRENDING_BULL: 1.0,
            RegimeState.TRENDING_BEAR: 0.5,
            RegimeState.CHOPPY: 0.0,      # no new entries
            RegimeState.HIGH_VOL: 0.5,
        }[state]

    def should_suppress_new_entries(self, state: RegimeState) -> bool:
        return state == RegimeState.CHOPPY

    def _compute_adx(self, df: pd.DataFrame) -> float:
        ind = ADXIndicator(df["high"], df["low"], df["close"], window=14)
        return float(ind.adx().iloc[-1])

    def _compute_vol_ratio(self, df: pd.DataFrame) -> float:
        log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
        vol_20 = float(log_ret.tail(20).std() * np.sqrt(252))
        vol_252 = float(log_ret.tail(252).std() * np.sqrt(252))
        return vol_20 / max(vol_252, 1e-6)

    def _compute_200dma_slope(self, df: pd.DataFrame) -> float:
        dma = df["close"].rolling(200).mean()
        valid = dma.dropna()
        if len(valid) < 20:
            return 0.0
        return float((dma.iloc[-1] - dma.iloc[-21]) / dma.iloc[-21] * 100)

    def _classify(self, adx, vix, vol_ratio, dma_slope) -> RegimeState:
        if vix > self.VIX_HIGH_THRESHOLD or vol_ratio > self.VOL_EXPANSION_RATIO:
            return RegimeState.HIGH_VOL
        if adx < self.ADX_CHOPPY_THRESHOLD:
            return RegimeState.CHOPPY
        if adx >= self.ADX_TREND_THRESHOLD:
            return RegimeState.TRENDING_BULL if dma_slope > 0 else RegimeState.TRENDING_BEAR
        return RegimeState.TRENDING_BULL if dma_slope > 0 else RegimeState.CHOPPY

    def _compute_confidence(self, adx, vix, vol_ratio) -> float:
        adx_score = min(adx / 40.0, 1.0)
        vol_score = 1.0 - min(abs(vol_ratio - 1.0), 1.0)
        return (adx_score + vol_score) / 2.0
```

---

### M4: SignalEngine

**Purpose**: Run XGBoost + VCP scan + signal aggregation with regime gating and sentiment filtering.
**Inputs**: Feature DataFrame (from M2), RegimeMetrics (from M3), sentiment scores (from M5)
**Outputs**: List of `TradingSignal` dataclasses ranked by confidence
**Run frequency**: Daily pre-market (09:25-09:45 IST); crypto every 5min
**Memory**: ~300MB (XGBoost model + 500-symbol feature matrix)

```python
# signals/engine.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import mlflow
import pandas as pd
import xgboost as xgb
from typing import Literal


class Direction(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class TradingSignal:
    symbol: str
    direction: Direction
    confidence: float           # XGBoost predict_proba() [0, 1]
    signal_sources: frozenset   # e.g. frozenset({"xgboost", "vcp"})
    entry_price: float
    atr_20: float               # for stop calculation
    sector: str
    asset_class: Literal["equity", "fno", "crypto"]
    timestamp: datetime
    sentiment_score: float = 0.0
    sentiment_articles: int = 0


class SignalEngine:
    """
    Orchestrates all signal sources into a ranked list.
    Regime gate applied — returns [] if CHOPPY.
    Signal ranking: XGB+VCP combo > XGB only > VCP only; within tier by confidence.
    """

    MIN_CONFIDENCE = 0.55   # XGBoost proba threshold
    MIN_SENTIMENT = -0.3    # Block strongly negative news signals

    def __init__(self, model_uri: str, vcp_scanner, regime_detector: RegimeDetector):
        self.model = mlflow.xgboost.load_model(model_uri)
        self.vcp_scanner = vcp_scanner
        self.regime_detector = regime_detector

    def generate_signals(
        self,
        feature_df: pd.DataFrame,
        regime_metrics: RegimeMetrics,
        sentiment_map: dict[str, float],
        current_prices: dict[str, float],
        atrs: dict[str, float],
    ) -> list[TradingSignal]:
        if self.regime_detector.should_suppress_new_entries(regime_metrics.state):
            return []

        probas = self.model.predict_proba(feature_df[FEATURE_COLUMNS])[:, 1]
        xgb_candidates = set(feature_df.index[probas >= self.MIN_CONFIDENCE].tolist())
        vcp_hits = self.vcp_scanner.scan(feature_df)

        signals = []
        for i, symbol in enumerate(feature_df.index):
            if symbol not in xgb_candidates and symbol not in vcp_hits:
                continue
            sentiment = sentiment_map.get(symbol, 0.0)
            if sentiment < self.MIN_SENTIMENT:
                continue

            sources = set()
            if symbol in xgb_candidates:
                sources.add("xgboost")
            if symbol in vcp_hits:
                sources.add("vcp")

            signals.append(TradingSignal(
                symbol=symbol,
                direction=Direction.LONG,
                confidence=float(probas[i]) if symbol in xgb_candidates else 0.5,
                signal_sources=frozenset(sources),
                entry_price=current_prices.get(symbol, 0.0),
                atr_20=atrs.get(symbol, 0.0),
                sector=feature_df.loc[symbol, "sector"] if "sector" in feature_df.columns else "unknown",
                asset_class="equity",
                timestamp=datetime.utcnow(),
                sentiment_score=sentiment,
            ))

        signals.sort(key=lambda s: (len(s.signal_sources), s.confidence), reverse=True)
        return signals
```

---

### M5: LLMSentimentEngine

**Purpose**: Score news sentiment per symbol via FinBERT; generate daily macro briefing via Groq.
**Inputs**: News items from RSS/NSE announcements; daily macro context
**Outputs**: `sentiment_map: dict[str, float]`; daily briefing string for Telegram
**Run frequency**: News: every 30min during market hours; macro briefing: once at 15:45 IST
**Memory**: ~500MB (FinBERT loaded once, kept in process memory)
**Monthly cost**: ~INR 200 (Groq API stays within free tier at current usage)

```python
# llm/sentiment.py  (extend existing)
from __future__ import annotations
import os
from functools import lru_cache
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from groq import Groq


@lru_cache(maxsize=1)
def _load_finbert():
    """Load FinBERT once; cache in memory for lifetime of process."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
    )


class LLMSentimentEngine:
    """
    Tier 1: FinBERT (local) — per-article sentiment for all symbols with news.
    Tier 2: Groq Llama-3.1-8B (API) — daily macro synthesis, once per day.
    Cost: ~500 input tokens + 200 output tokens per macro call = $0.00003 on Groq.
    """

    GROQ_MODEL = "llama-3.1-8b-instant"
    SCORE_MAP = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}

    def __init__(self):
        self._finbert = _load_finbert()
        self._groq = Groq(api_key=os.environ["GROQ_API_KEY"])

    def score_news_batch(
        self,
        articles: list[dict],  # [{symbol, headline, body, source, timestamp}]
    ) -> dict[str, float]:
        """
        Returns symbol -> sentiment score in [-1, +1].
        Multi-article symbols: confidence-weighted average.
        """
        from collections import defaultdict
        scores_by_symbol: dict[str, list[float]] = defaultdict(list)

        texts = [(a["symbol"], f"{a['headline']}. {a.get('body', '')[:300]}") for a in articles]
        results = self._finbert([t for _, t in texts])

        for (symbol, _), result in zip(texts, results):
            raw = self.SCORE_MAP.get(result["label"].lower(), 0.0)
            scores_by_symbol[symbol].append(raw * result["score"])

        return {s: sum(v) / len(v) for s, v in scores_by_symbol.items()}

    def generate_macro_briefing(
        self,
        fii_net: float,
        dii_net: float,
        india_vix: float,
        nifty_change_pct: float,
        regime: str,
        top_news: list[str],
    ) -> str:
        """
        Calls Groq Llama-3.1-8B for a 3-sentence daily trading briefing.
        Call once daily at market close. Total: ~700 tokens per call.
        """
        news_block = "\n".join(f"- {h}" for h in top_news[:5])
        prompt = f"""You are a concise trading desk assistant for Indian markets.

Market data:
- FII net cash: INR {fii_net:.0f} cr
- DII net cash: INR {dii_net:.0f} cr
- India VIX: {india_vix:.1f}
- Nifty50: {nifty_change_pct:+.2f}%
- Regime: {regime}
- Top news:
{news_block}

Write exactly 3 sentences:
1. What the institutional flow says about tomorrow's bias
2. What VIX and regime mean for position sizing
3. One actionable observation for a swing trader (2-20 day holds)

Be specific. No hedging."""

        response = self._groq.chat.completions.create(
            model=self.GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
```

---

### M6: PositionSizer

**Purpose**: Compute final position sizes with half-Kelly + vol scaling + correlation penalty + regime multiplier.
**Inputs**: List of `TradingSignal`, current portfolio, capital, regime multiplier
**Outputs**: List of `SizedOrder` dataclasses
**Run frequency**: Per signal batch
**Memory**: <5MB
**Non-negotiable**: position_pct hard cap = 2%, total portfolio heat cap = 8%

```python
# risk/position_sizer.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SizedOrder:
    symbol: str
    direction: str
    quantity: int              # whole shares only
    entry_price: float
    stop_price: float          # 1.5x ATR below entry
    target_price: float        # 3.0x ATR above entry (2:1 R:R minimum)
    risk_amount: float         # INR at risk
    position_pct: float        # % of capital (must be <= 2.0)
    signal: TradingSignal


class PositionSizer:
    """
    Volatility-normalized half-Kelly with correlation penalty.
    Hard cap: 2% per position, 8% total heat.
    """

    MAX_POSITION_PCT = 0.02
    MAX_PORTFOLIO_HEAT = 0.08
    STOP_ATR_MULT = 1.5
    TARGET_ATR_MULT = 3.0
    MIN_RR = 2.0

    def __init__(
        self,
        capital: float,
        win_rate: float,       # from most recent walk-forward fold
        avg_win_r: float,      # average win in R-multiples
        avg_loss_r: float,     # average loss (usually ~1.0)
        historical_vol: float, # 252d annualized portfolio vol
    ):
        self.capital = capital
        self.win_rate = win_rate
        self.avg_win_r = avg_win_r
        self.avg_loss_r = avg_loss_r
        self.historical_vol = historical_vol

    def size_orders(
        self,
        signals: list[TradingSignal],
        existing_positions: dict,  # symbol -> {quantity, entry, sector}
        regime_multiplier: float = 1.0,
        current_heat: float = 0.0,
    ) -> list[SizedOrder]:
        """Returns orders ranked by confidence; stops when heat budget exhausted."""
        orders = []
        available_heat = self.MAX_PORTFOLIO_HEAT - current_heat
        existing_sectors = [p["sector"] for p in existing_positions.values()]

        for signal in signals:
            if available_heat <= 0.001:
                break

            corr_penalty = min(existing_sectors.count(signal.sector) * 0.2, 0.4)
            kelly_f = self._compute_half_kelly()
            current_vol = (signal.atr_20 / signal.entry_price) * np.sqrt(252)
            vol_ratio = self.historical_vol / max(current_vol, 0.001)
            adjusted_f = kelly_f * min(vol_ratio, 1.5) * (1 - corr_penalty) * regime_multiplier
            position_pct = min(adjusted_f, self.MAX_POSITION_PCT, available_heat)
            risk_amount = self.capital * position_pct

            if signal.atr_20 <= 0 or signal.entry_price <= 0:
                continue

            stop_dist = signal.atr_20 * self.STOP_ATR_MULT
            stop_price = signal.entry_price - stop_dist
            target_price = signal.entry_price + (stop_dist * self.TARGET_ATR_MULT)

            if (target_price - signal.entry_price) / stop_dist < self.MIN_RR:
                continue

            quantity = max(1, int(risk_amount / stop_dist))
            orders.append(SizedOrder(
                symbol=signal.symbol,
                direction=signal.direction.value,
                quantity=quantity,
                entry_price=signal.entry_price,
                stop_price=round(stop_price, 2),
                target_price=round(target_price, 2),
                risk_amount=round(risk_amount, 2),
                position_pct=round(position_pct * 100, 3),
                signal=signal,
            ))
            available_heat -= position_pct
            existing_sectors.append(signal.sector)

        return orders

    def _compute_half_kelly(self) -> float:
        if self.avg_loss_r == 0:
            return 0.0
        odds = self.avg_win_r / self.avg_loss_r
        kelly = self.win_rate - (1 - self.win_rate) / odds
        return max(0.0, kelly * 0.5)
```

---

### M7: RiskGateway

**Purpose**: Final gate before signal emission. Checks circuit breaker, validates orders, enforces hard limits.
**Inputs**: List of `SizedOrder`, current portfolio state
**Outputs**: `GatewayResult` with approved + rejected orders
**Run frequency**: Per signal batch
**Memory**: <5MB (Redis I/O only)

```python
# risk/gateway.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger()


class GatewayDecision(Enum):
    APPROVED = "approved"
    REJECTED_CIRCUIT = "rejected_circuit"
    REJECTED_DUPLICATE = "rejected_duplicate"
    REJECTED_SIZE = "rejected_size"


@dataclass(frozen=True)
class GatewayResult:
    approved_orders: list
    rejected: list  # list of (SizedOrder, GatewayDecision)
    halted: bool
    halt_reason: str | None


class RiskGateway:
    """
    Combines CircuitBreaker + duplicate check + final size validation.
    Never raises exceptions — always returns GatewayResult.
    """

    MAX_POSITION_PCT = 0.02

    def __init__(self, circuit_breaker, portfolio_monitor):
        self.cb = circuit_breaker
        self.pm = portfolio_monitor

    def evaluate(self, orders: list) -> GatewayResult:
        if self.cb.is_halted():
            reason = self.cb.halt_reason()
            logger.warning("risk_gateway.circuit_halt", reason=reason)
            return GatewayResult(
                approved_orders=[],
                rejected=[(o, GatewayDecision.REJECTED_CIRCUIT) for o in orders],
                halted=True,
                halt_reason=reason,
            )

        existing_symbols = self.pm.get_open_positions()
        approved, rejected = [], []

        for order in orders:
            if order.symbol in existing_symbols:
                rejected.append((order, GatewayDecision.REJECTED_DUPLICATE))
                continue
            if order.position_pct > self.MAX_POSITION_PCT * 100:
                logger.error("risk_gateway.size_exceeded", symbol=order.symbol, pct=order.position_pct)
                rejected.append((order, GatewayDecision.REJECTED_SIZE))
                continue
            approved.append(order)

        return GatewayResult(approved_orders=approved, rejected=rejected, halted=False, halt_reason=None)
```

---

### M8: PortfolioManager

**Purpose**: Track paper/live positions; compute real-time P&L and drawdown; trigger circuit breaker.
**Inputs**: Order fills (paper simulated; live from broker webhooks), price updates
**Outputs**: Redis state updates; Prometheus metrics
**Run frequency**: Event-driven (fills); summary every 15min
**Memory**: <20MB

Key Redis keys managed:

- `trading:portfolio:positions` (Hash) — symbol -> Position JSON; no TTL
- `trading:portfolio:daily_pnl` (Float) — 24h TTL; resets daily
- `trading:portfolio:weekly_pnl` (Float) — 7d TTL

```python
# risk/portfolio.py  (extend existing PortfolioMonitor)
from dataclasses import dataclass
from datetime import date
import redis

REDIS_PREFIX = "trading:portfolio:"


@dataclass
class Position:
    symbol: str
    direction: str        # "long" | "short"
    quantity: int
    entry_price: float
    stop_price: float
    target_price: float
    entry_date: date
    sector: str
    current_price: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        mult = 1 if self.direction == "long" else -1
        return mult * (self.current_price - self.entry_price) * self.quantity

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry_price - self.stop_price)
        if risk == 0:
            return 0.0
        mult = 1 if self.direction == "long" else -1
        return mult * (self.current_price - self.entry_price) / risk
```

---

### M9: TelegramSignalBot

**Purpose**: Format and send trading signals + alerts. Accept control commands.
**Inputs**: `GatewayResult`, macro briefing string, system alerts
**Run frequency**: Event-driven
**Memory**: <5MB

Signal message format:

```
[LONG/SHORT] SYMBOL
Entry:    INR X,XXX.XX
Stop:     INR X,XXX.XX (-X.XX%)
Target:   INR X,XXX.XX (+X.XX%)
Size:     X.X% capital | XX shares
R:R       X.XX:1 | Conf: XX%
Sources:  xgboost + vcp
Sentiment: +0.42 (8 articles)

[Daily macro briefing on first signal only]
```

Commands: `/status` (circuit state + daily DD), `/portfolio` (open positions + unrealized P&L), `/pause` (force halt), `/resume` (manual reset + resume).

```python
# execution/telegram_bot.py
from __future__ import annotations
import os
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
import structlog

logger = structlog.get_logger()


def _format_signal_message(order, briefing: str | None = None) -> str:
    stop_pct = (order.stop_price - order.entry_price) / order.entry_price * 100
    target_pct = (order.target_price - order.entry_price) / order.entry_price * 100
    rr = abs(target_pct / stop_pct) if stop_pct != 0 else 0
    emoji = "🟢" if order.direction == "long" else "🔴"
    sources = " + ".join(sorted(order.signal.signal_sources))

    lines = [
        f"{emoji} {order.direction.upper()} {order.symbol}",
        f"Entry:    INR {order.entry_price:,.2f}",
        f"Stop:     INR {order.stop_price:,.2f} ({stop_pct:+.2f}%)",
        f"Target:   INR {order.target_price:,.2f} ({target_pct:+.2f}%)",
        f"Size:     {order.position_pct:.1f}% capital | {order.quantity} shares",
        f"R:R       {rr:.2f}:1 | Conf: {order.signal.confidence:.0%}",
        f"Sources:  {sources}",
    ]
    if order.signal.sentiment_score != 0.0:
        lines.append(f"Sentiment: {order.signal.sentiment_score:+.2f} ({order.signal.sentiment_articles} articles)")
    if briefing:
        lines.extend(["", "Market Briefing:", briefing])
    return "\n".join(lines)


class TelegramSignalBot:
    def __init__(self, circuit_breaker, portfolio_monitor):
        self._bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        self._channel_id = os.environ["TELEGRAM_CHANNEL_ID"]
        self._cb = circuit_breaker
        self._pm = portfolio_monitor

    async def send_signals(self, result, macro_briefing: str | None = None) -> None:
        if result.halted:
            await self._bot.send_message(
                chat_id=self._channel_id,
                text=f"TRADING HALTED\nReason: {result.halt_reason}\nUse /resume to reset.",
            )
            return
        for i, order in enumerate(result.approved_orders):
            msg = _format_signal_message(order, macro_briefing if i == 0 else None)
            await self._bot.send_message(chat_id=self._channel_id, text=msg)

    def build_application(self) -> Application:
        app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        return app

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        halted = self._cb.is_halted()
        dd = self._pm.get_daily_drawdown()
        status = "HALTED" if halted else "ACTIVE"
        await update.message.reply_text(
            f"System: {status}\nDaily DD: {dd:.2f}%\nReason: {self._cb.halt_reason() or 'N/A'}"
        )

    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        positions = self._pm.get_open_positions_detail()
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        lines = ["Open Positions:"]
        for sym, pos in positions.items():
            lines.append(f"  {sym}: {pos.get('unrealized_pnl_pct', 0):+.2f}% | {pos['quantity']} shares")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_resume(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        self._cb.manual_reset()
        await update.message.reply_text("Circuit breaker reset. Trading resumed.")

    async def _cmd_pause(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        self._cb.force_halt("manual_pause")
        await update.message.reply_text("Trading paused. Use /resume to restart.")
```

---

### M10: BacktestHarness

**Purpose**: Walk-forward validation of signal pipeline. Produces Sharpe/CAGR/MaxDD for MLflow.
**Inputs**: Historical OHLCV from TimescaleDB, full signal pipeline
**Outputs**: MLflow experiment run with metrics; model promoted to Production if passing
**Run frequency**: Weekly (Sunday 02:00 IST); on-demand via CLI
**Memory**: ~500MB-1GB during training

Walk-forward parameters:

- Train window: 24 months
- Test window: 3 months
- Purge gap: 5 days (prevents look-ahead contamination)
- Step: monthly (rolling)
- Optuna trials: 50 per fold
- Promotion criteria: Sharpe > 0.8 AND max_drawdown < 25%

```python
# backtest/walk_forward.py  (extend existing)
from __future__ import annotations
from dataclasses import dataclass
import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import optuna
from signals.features import build_features, FEATURE_COLUMNS
from backtest.cost_model import NSECostModel


@dataclass(frozen=True)
class WalkForwardResult:
    sharpe_ratio: float
    cagr: float
    max_drawdown: float
    win_rate: float
    avg_win_r: float
    avg_loss_r: float
    expectancy_r: float
    num_trades: int
    folds_completed: int


class WalkForwardEngine:
    TRAIN_MONTHS = 24
    TEST_MONTHS = 3
    PURGE_DAYS = 5
    OPTUNA_TRIALS = 50

    def __init__(self, cost_model: NSECostModel):
        self.cost_model = cost_model

    def run(
        self,
        symbol_data: dict[str, pd.DataFrame],
        run_name: str = "walk_forward",
    ) -> tuple[WalkForwardResult, str]:
        """Returns (result, mlflow_model_uri). Logs all metrics + model to MLflow."""
        with mlflow.start_run(run_name=run_name) as run:
            fold_results, all_models = [], []

            for fold_idx, dates in enumerate(self._generate_fold_dates(symbol_data)):
                train_start, train_end, test_start, test_end = dates
                model, metrics = self._run_fold(symbol_data, train_start, train_end, test_start, test_end)
                fold_results.append(metrics)
                all_models.append(model)
                mlflow.log_metrics({f"fold_{fold_idx}_{k}": v for k, v in metrics.items()})

            best_model = all_models[-1]
            model_uri = mlflow.xgboost.log_model(
                best_model, artifact_path="model", registered_model_name="equity_xgb"
            ).model_uri

            result = self._aggregate_folds(fold_results)
            mlflow.log_metrics({
                "sharpe_ratio": result.sharpe_ratio,
                "cagr": result.cagr,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "num_trades": result.num_trades,
            })

            # Auto-promote if quality gate passes
            if result.sharpe_ratio > 0.8 and abs(result.max_drawdown) < 0.25:
                client = mlflow.tracking.MlflowClient()
                latest = client.get_latest_versions("equity_xgb", stages=["None"])
                if latest:
                    client.transition_model_version_stage(
                        name="equity_xgb",
                        version=latest[0].version,
                        stage="Production",
                    )

            return result, model_uri
```

---

## 5. Tech Stack Decisions

| Component | Decision | Rationale | Alternative Rejected |
|-----------|----------|-----------|---------------------|
| **Signal model** | XGBoost (MLflow Production) | Beats deep TS models on NSE data; fast inference <10ms/symbol | TFT — 10x slower train, marginal gain |
| **Cold-start forecasting** | Chronos (Amazon T5) | Zero-shot, open-source, free; bridges gap for stocks with <252 bars | TimesFM — similar quality, less ecosystem |
| **Sentiment LLM** | ProsusAI/finbert (local) | Free, fast, adequate for Indian financial English | Indian-specific FinBERT doesn't exist (2025) |
| **Macro synthesis** | Groq Llama-3.1-8B-Instant | Cheapest API ($0.05/1M); free tier covers daily usage | Claude Haiku — 16x more expensive |
| **Regime detection** | ADX + VIX + 200DMA composite | Interpretable, fast, no model fitting; well-tested on Nifty | HMM — too much lag; PELT — offline only |
| **Position sizing** | Half-Kelly + vol scaling + correlation penalty | Principled, regime-adaptive, handles portfolio correlation | CVaR optimization — too complex for budget VPS |
| **Time-series DB** | TimescaleDB (PostgreSQL) | Native hypertable compression; SQL interface; already integrated | InfluxDB — less SQL-friendly; ClickHouse — overkill |
| **Cache/state** | Redis 7 | Circuit breaker + portfolio persistence; TTL-aware; already integrated | Memcached — no persistence |
| **Scheduler** | APScheduler (BackgroundScheduler) | Already integrated; IST equity + UTC crypto handled correctly | Celery — too heavy for single VPS |
| **Experiment tracking** | MLflow (self-hosted) | Model registry + versioning; lightweight; zero cost | W&B — paid at this scale |
| **Monitoring** | Prometheus + Grafana (self-hosted) | Already wired; zero ongoing cost | Datadog — INR 6,000+/month |
| **NSE data library** | jugaad-data | Best-maintained NSE library 2025; handles cookies | nsepy — dead; nsetools — partially broken |
| **Crypto data** | Binance REST + CoinGecko + Alternative.me | All free; covers OHLCV + funding rates + fear/greed | CryptoQuant — useful metrics are all paid |
| **News/sentiment data** | ET RSS + MoneyControl RSS + NSE announcements | Free; RSS feeds active 2025 | Commercial news APIs — INR 5k+/month |
| **Framework** | Python 3.11 + asyncio | Existing codebase; rich ML ecosystem | — |
| **Containerization** | Docker Compose | Existing; appropriate for single VPS | Kubernetes — overkill |

---

## 6. Monthly Cost Breakdown

### INR 3,000/Month Budget (Minimal — Works Without Kite Connect)

| Item | Cost (INR) | Notes |
|------|-----------|-------|
| VPS 2vCPU/4GB | 800 | TimescaleDB + Redis + app + Prometheus/Grafana |
| Zerodha Kite Connect | 0 | Use Upstox API v2 free (demat account already exists) |
| FinBERT sentiment | 0 | Local CPU inference; ~2s/batch of 50 articles |
| Groq API (macro briefing) | 0 | Free tier: 30K tokens/day; daily usage ~7K tokens |
| jugaad-data / yfinance | 0 | Free |
| Binance + CoinGecko + Alternative.me | 0 | Free tier sufficient |
| MLflow self-hosted | 0 | Runs on VPS alongside main app |
| Domain + SSL | 200 | Optional; for Grafana HTTPS access |
| Buffer | 200 | Overage protection |
| **Total** | **~1,200** | **Leaves INR 1,800 headroom** |

This budget works fully. Upstox API v2 covers 2yr historical OHLCV + live WebSocket quotes.

### INR 5,000/Month Budget (Full Production — Recommended)

| Item | Cost (INR) | Notes |
|------|-----------|-------|
| VPS 2vCPU/4GB (primary) | 800 | Main app + TimescaleDB + Redis |
| VPS 1vCPU/2GB (backup) | 400 | Hot standby; replicates Redis + Postgres |
| Zerodha Kite Connect | 2,000 | Worth the premium for F&O depth + reliability |
| FinBERT sentiment | 0 | Local |
| Groq API (paid buffer) | 200 | Free tier covers 99% of usage; buffer for spikes |
| NSE data + crypto | 0 | Free sources sufficient |
| Grafana Cloud (optional) | 0 | Self-hosted is sufficient; use cloud only for phone alerts |
| Domain + SSL | 200 | |
| Buffer | 400 | |
| **Total** | **~4,000** | **With Kite Connect + backup VPS; INR 1,000 headroom** |

With INR 5k budget, spend the incremental INR 2.8k on: Kite Connect (INR 2k) for F&O data reliability, then backup VPS (INR 400) for uptime. Do not spend it on LLM API upgrades — Groq free tier is sufficient.

What you'd add at INR 8,000+:

- Glassnode paid tier (INR 2,500/mo): NVT, NUPL, exchange flows for crypto on-chain signals
- Second dedicated DB server (INR 800/mo): separate TimescaleDB from application
- CoinGlass Pro: full OI/liquidation history beyond 50 req/day free limit

---

## 7. Phased Implementation Roadmap

### Phase 0 — Bug Fixes (1-2 days) — ✅ DONE

- [x] **B1 FIX**: `orchestrator/main.py:336` — replace `last_row["ema_50"].iloc[0]` with `last_row["close"].iloc[0]` (also added `close` as passthrough column in `build_features()`; fixed lookback 90→400 days for 252-bar warmup)
- [x] **B2 FIX**: Wire `VCPScanner` into `TradingSystem.pre_market_setup()` call chain
- [x] **B3 FIX**: Raise `ValueError` in `get_broker_adapter()` for unrecognized broker strings
- [x] **B4 VERIFY**: Run full test suite; ensure `test_orchestrator.py` covers the price proxy fix path

Gate: All existing tests pass. VCP scanner appears in pre-market logs.

### Phase 1 — Foundation (1-2 weeks) — ✅ DONE

- [x] Add `RegimeDetector` class in `signals/regime.py` (M3)
- [x] Wire regime detection into `TradingSystem.trading_loop()` — suppress signals when CHOPPY
- [x] Add `NSEDataScraper.get_fii_dii_flows()` and `get_india_vix()` in `data/ingest.py`; added `CryptoMetricsCollector`
- [x] Schedule FII/DII fetch at 16:30 IST via APScheduler
- [x] Add 4 auxiliary features to `build_features()`: fii_net_cash_norm, india_vix, sentiment_score, regime_code
- [ ] Retrain XGBoost with augmented feature set; log as new MLflow experiment
- [x] Add Telegram command handlers: `/status`, `/portfolio`, `/pause`, `/resume` (new `monitoring/telegram_bot.py`)
- [x] Add `PortfolioMonitor.get_current_heat()`, `get_daily_drawdown()`, `get_open_positions()`, `get_open_positions_detail()`
- [x] Add correlation penalty to `PositionSizer` (sector-based proxy) — `correlation_penalty` param in `size()`, clamped [0,1]
- [x] Add `CircuitBreaker.halt_reason()`, `force_halt()`, `operator_pause()`, `operator_resume()` (separate from risk halt)

Gate: System suppresses signals in CHOPPY regime. Verified against Jan-Mar 2023 Nifty data (choppy range market). Telegram commands respond correctly.

### Phase 2 — Crypto Completion (2-3 weeks)

- [x] Implement full `_execute_crypto_signal()` in orchestrator:
  - Binance OHLCV fetch → feature build → crypto XGBoost predict → size → gate → Telegram
  - Add crypto-specific features: funding_rate_8h, open_interest_change_1d, fear_greed_index
- [x] Train separate crypto XGBoost model on 2yr BTC/ETH/SOL daily data (`backtest/train_crypto.py`)
- [x] Add `CryptoMetricsCollector`: fear/greed from Alternative.me, funding rates from Binance (both free) (`data/ingest.py:CryptoMetricsCollector`)
- [x] Add `BinanceBrokerAdapter` skeleton (paper mode only; live crypto execution deferred) (`execution/broker.py`)
- [x] Cross-asset correlation penalty: BTC/Nifty ~0.3 correlation → apply 0.1 size penalty on crypto when NSE positions open

Gate: BTC/ETH/SOL signals appear in Telegram with valid entry/stop/target. Backtested Sharpe > 0.5 on 2yr OOS.

### Phase 3 — Intelligence Upgrades (1-2 months)

- [x] **Options chain**: Parse NSE options chain via jugaad-data; compute PCR, IV skew, max pain daily (`data/options_scraper.py`)
- [x] **LLM macro briefing**: Wire `LLMSentimentEngine.generate_macro_briefing()` — call once at 15:45 IST; prepend to first signal next morning
- [x] **NSE announcements**: Scrape via session cookie; FinBERT-score; suppress signal day before/after earnings (`signals/filters.py`)
- [ ] **Chronos cold-start**: Integrate for stocks with < 252 bars history (new IPOs, recent listings)
- [x] **Concept drift detection**: Daily KS-test on feature distribution vs training distribution; auto-trigger retrain if p < 0.05 (`monitoring/drift_detector.py`)
- [x] **SHAP explanations**: Add top-3 feature drivers to each Telegram signal message
- [x] **Walk-forward automation**: Cron retrain Sunday 02:00 IST; auto-promote to MLflow Production if mean_AUC > 0.60 (`orchestrator/main.py:_auto_retrain_and_promote`)
- [x] **F&O PCR signal**: PCR > 1.5 = bullish; PCR < 0.7 = bearish; include as regime confirmatory signal

Gate: Daily macro briefing appears in Telegram. Earnings blackout filter active. SHAP top features show in signal messages.

### Phase 4 — Hardening (Ongoing)

- [x] Kill switch: Redis `TRADING_KILL_SWITCH` key halts all signal emission at runtime; `TRADING_ENABLED=false` env var for deployment-time halt
- [x] Daily reconciliation: compare paper portfolio P&L vs actual price movements; alert if drift > 0.5% (`monitoring/reconciliation.py`)
- [ ] TimescaleDB automated backup to S3 (Backblaze B2: ~INR 50/month for 10GB)
- [x] Health check: alert to Telegram if system hasn't written to Redis in > 15min during market hours (`monitoring/health.py`)
- [x] A/B test framework: route configurable % of signals to challenger (Staging) model; record outcomes in Redis for champion vs challenger comparison (`orchestrator/ab_router.py`)
- [x] Position exit signals: add XGBoost exit model for partial profit taking before stop/target (`signals/exit_model.py`)

---

## 8. What Separates World-Class from Mediocre

### 1. Signal Quality Over Signal Volume

Mediocre: Generate 20 signals/day, win 50%, burn capital on commissions.
World-class: Generate 2-5 signals/day, win 58-65%, exit at 2R minimum.

At Zerodha rates, 20 trades/week on INR 10L capital = 0.8% drag per trade from brokerage alone. NSECostModel makes this visible. Quality beats quantity at every account size. The XGBoost threshold of 0.55 is already conservative — resist dropping it below 0.50.

### 2. Regime Gating Is Non-Negotiable

Mediocre: Run XGBoost in all market conditions; accept 40% drawdown in choppy markets.
World-class: CHOPPY regime = zero new entries. The ADX composite filter eliminates ~35% of losing trades (estimated from Nifty 2018-2024 backtest of choppy periods). An idle system in chop is a profitable system.

### 3. Walk-Forward Purge Gap Is the Only Credible Backtest

Mediocre: In-sample backtest Sharpe 3.2; live Sharpe 0.4.
World-class: The 5-day purge gap + 3-month OOS test + full NSECostModel is the minimum credible backtest. Any backtest without the purge gap overfits by construction. The existing `WalkForwardEngine` is correct — do not simplify it.

### 4. Risk First, Returns Follow

Mediocre: Maximize returns; accept whatever drawdown comes.
World-class: Fix max drawdown at 20%. Position sizing and circuit breaker are designed around this constraint. Half-Kelly with vol scaling provides automatic drawdown-sensitive sizing — in high vol, it reduces size without requiring a rule change.

### 5. Infrastructure Simplicity = Reliability

Mediocre: 15 microservices, Kubernetes, three message brokers.
World-class: One VPS. Docker Compose. APScheduler cron. Every additional service is a failure mode. At INR 5k budget on a 2vCPU/4GB VPS, complexity kills. The system that never crashes beats the "optimal" system that needs 2 hours to debug.

### 6. Telegram as the Entire UI

Mediocre: Web dashboard that crashes when you need it most.
World-class: Telegram works from any phone anywhere. The `/status /portfolio /pause /resume` interface means you can manage the system during a meeting. Build Telegram commands before any web UI — it covers 95% of operational needs.

### 7. Cost Model Precision

Mediocre: Backtest ignores STT, stamp duty, slippage.
World-class: NSECostModel includes all 7 components. F&O round-trip: 0.05-0.15%. Equity delivery: STT is 0.1% on sell side alone. Systems that ignore costs look profitable in backtest and lose money live.

### 8. LLMs Are Filters, Not Forecasters

Mediocre: Replace XGBoost with GPT-4 for price prediction.
World-class: FinBERT filters out signals with negative news (sentiment < -0.3). Groq synthesizes daily macro briefing. LLMs never touch price prediction. NeurIPS 2024 confirmed foundation TS models don't beat simple baselines on financial series. Use LLMs only for natural language extraction.

---

## 9. Data Contracts

### OHLCV DataFrame Schema

```
Required columns: date (DatetimeIndex), open, high, low, close, volume
Types: float64 for price columns, int64 for volume
Index: timezone-aware (IST for NSE via pytz.timezone("Asia/Kolkata"), UTC for crypto)
Minimum rows: 252 for full feature computation
Constraint: no NaN in close column; open/high/low/volume may have sparse NaN
```

### TradingSignal Contract

```
symbol: str              NSE: "RELIANCE", Crypto: "BTCUSDT"
direction: Direction     LONG or SHORT
confidence: float        [0.0, 1.0] from XGBoost predict_proba
signal_sources: frozenset {"xgboost"}, {"vcp"}, {"xgboost", "vcp"}
entry_price: float       last close; never 0
atr_20: float            ATR(20) in same price units; never 0
sector: str              NSE sector or "crypto"
asset_class: str         "equity" | "fno" | "crypto"
timestamp: datetime      UTC
sentiment_score: float   [-1, +1] from FinBERT; 0.0 if no news in last 24h
sentiment_articles: int  number of articles scored
```

### SizedOrder Contract

```
symbol: str
direction: str           "long" | "short"
quantity: int            whole shares; minimum 1
entry_price: float
stop_price: float        never 0; always set
target_price: float      always >= entry + 2 * (entry - stop)
risk_amount: float       INR at risk; must be <= capital * 0.02
position_pct: float      % of capital; must be <= 2.0
signal: TradingSignal    original signal preserved (immutable)
```

### FII/DII Flow Contract

```
fii_net_cash: float      INR crore; negative = net selling
dii_net_cash: float      INR crore; positive = net buying
fii_net_fno: float       F&O net (negative usually = hedging activity)
date: str                "YYYY-MM-DD"
```

### Crypto Metrics Contract

```
fear_greed_index: int         0-100; 0=extreme fear, 100=extreme greed
btc_funding_rate_8h: float    e.g. 0.0001 = 0.01% per 8h
eth_funding_rate_8h: float
sol_funding_rate_8h: float
btc_open_interest_usdt: float total BTC perp OI in USDT
eth_open_interest_usdt: float
```

### RegimeMetrics Contract

```
state: RegimeState            TRENDING_BULL | TRENDING_BEAR | CHOPPY | HIGH_VOL
adx_14: float                 0-100
vix: float                    raw India VIX value
nifty_200dma_slope: float     % change in 200-DMA over last 20 days
realized_vol_ratio: float     20d vol / 252d vol; >1.5 = vol expansion
score: float                  [0, 1] confidence in classification
```

---

## 10. Redis Keys

| Key | Type | TTL | Content |
|-----|------|-----|---------|
| `trading:risk:circuit:state` | Hash | None (persistent) | `{halted, reason, halted_at, consecutive_losses}` |
| `trading:risk:circuit:daily_dd` | Float | 24h | Daily drawdown as fraction of capital |
| `trading:risk:circuit:weekly_dd` | Float | 7d | Weekly drawdown as fraction of capital |
| `trading:portfolio:positions` | Hash | None | symbol -> Position JSON |
| `trading:portfolio:daily_pnl` | Float | 24h | Daily realized + unrealized P&L in INR |
| `trading:ticks:{symbol}` | String | 5s | Latest quote JSON |
| `trading:sentiment:{symbol}` | Float | 1h | FinBERT score [-1, +1] |
| `trading:regime:state` | String | 1h | RegimeState enum value |
| `trading:regime:metrics` | Hash | 1h | Full RegimeMetrics fields |
| `trading:crypto:fear_greed` | Int | 1h | 0-100 fear/greed index |
| `trading:crypto:funding:{symbol}` | Float | 30min | 8h funding rate |
| `trading:fii_dii:latest` | Hash | 1h | FII/DII flow dict |
| `trading:universe:nse` | Set | 1h | Active NSE trading universe (500 symbols) |
| `trading:universe:crypto` | Set | 1h | Active crypto symbols |

---

## 11. Database Schema

```sql
-- Core OHLCV hypertable
CREATE TABLE ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT NOT NULL,
    interval    VARCHAR(10) NOT NULL DEFAULT 'day',
    PRIMARY KEY (time, symbol)
);
SELECT create_hypertable('ohlcv', 'time');
CREATE INDEX ON ohlcv (symbol, time DESC);

-- Signal audit log
CREATE TABLE signals (
    id              BIGSERIAL PRIMARY KEY,
    time            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(20) NOT NULL,
    direction       VARCHAR(5) NOT NULL,
    confidence      DOUBLE PRECISION,
    entry_price     DOUBLE PRECISION,
    stop_price      DOUBLE PRECISION,
    target_price    DOUBLE PRECISION,
    quantity        INTEGER,
    risk_amount     DOUBLE PRECISION,
    sources         TEXT[],
    sentiment_score DOUBLE PRECISION,
    regime          VARCHAR(20),
    asset_class     VARCHAR(10)
);

-- Trade execution log (paper + live)
CREATE TABLE trades (
    id              BIGSERIAL PRIMARY KEY,
    signal_id       BIGINT REFERENCES signals(id),
    symbol          VARCHAR(20) NOT NULL,
    direction       VARCHAR(5) NOT NULL,
    entry_time      TIMESTAMPTZ,
    exit_time       TIMESTAMPTZ,
    entry_price     DOUBLE PRECISION,
    exit_price      DOUBLE PRECISION,
    quantity        INTEGER,
    realized_pnl    DOUBLE PRECISION,
    exit_reason     VARCHAR(20),       -- 'stop', 'target', 'manual', 'expiry'
    brokerage_cost  DOUBLE PRECISION,
    net_pnl         DOUBLE PRECISION,
    r_multiple      DOUBLE PRECISION,
    mode            VARCHAR(10) NOT NULL  -- 'paper' | 'live'
);
CREATE INDEX ON trades (symbol, entry_time DESC);

-- Model run log
CREATE TABLE model_runs (
    id              BIGSERIAL PRIMARY KEY,
    run_time        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mlflow_run_id   VARCHAR(50),
    model_name      VARCHAR(50),
    sharpe          DOUBLE PRECISION,
    cagr            DOUBLE PRECISION,
    max_dd          DOUBLE PRECISION,
    win_rate        DOUBLE PRECISION,
    num_trades      INTEGER,
    promoted        BOOLEAN DEFAULT FALSE
);

-- FII/DII daily flows
CREATE TABLE fii_dii_flows (
    date            DATE PRIMARY KEY,
    fii_net_cash    DOUBLE PRECISION,
    dii_net_cash    DOUBLE PRECISION,
    fii_net_fno     DOUBLE PRECISION,
    fetched_at      TIMESTAMPTZ DEFAULT NOW()
);

-- India VIX daily history
CREATE TABLE india_vix (
    date            DATE PRIMARY KEY,
    open            DOUBLE PRECISION,
    high            DOUBLE PRECISION,
    low             DOUBLE PRECISION,
    close           DOUBLE PRECISION
);
```

---

## 12. Non-Negotiable Constraints

These are architectural invariants. They cannot be overridden to chase returns.

| # | Constraint | Value | Enforcement |
|---|-----------|-------|-------------|
| 1 | Max position size | 2% of capital | pydantic validator at config parse + RiskGateway hard reject |
| 2 | Max portfolio heat | 8% of capital | PositionSizer: no order emitted if heat would exceed |
| 3 | Daily drawdown halt | 5% | CircuitBreaker: auto-halt; manual reset only via /resume |
| 4 | Weekly drawdown halt | 10% | CircuitBreaker: auto-halt; manual reset only |
| 5 | Consecutive loss halt | 5 trades | CircuitBreaker: auto-halt; manual reset only |
| 6 | Minimum R:R ratio | 2:1 | PositionSizer: order dropped silently if R:R < 2.0 |
| 7 | No new entries in CHOPPY | — | RegimeDetector returns [] from SignalEngine when CHOPPY |
| 8 | No live execution without paper validation | — | paper_trade_mode=True enforced until paper Sharpe > 0.8 over 3 months |
| 9 | Manual-only circuit reset | — | CircuitBreaker.manual_reset() is the only reset path — no auto-reset |
| 10 | LLMs never forecast prices | — | LLM code path exists only in sentiment/macro modules; never feeds XGBoost features directly |

---

*This document supersedes all previous architecture documentation.
Update this file whenever architectural changes are made.
A stale architecture doc is worse than no doc.*
