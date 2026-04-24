# Attribution and Feature Analysis Guide

## Overview

Phase 8, Workstream 4 implements a comprehensive performance attribution and feature analysis layer using SHAP values, feature importance trends, and trade outcome categorization.

**Goal:** Understand what drives returns and why we win/lose trades.

## Key Concepts

### 1. SHAP (SHapley Additive exPlanations) Values

SHAP values decompose each prediction into per-feature contributions, answering:
- "Which features moved the model most?"
- "Did feature X push the prediction up or down?"

**Example:**
```
Model prediction: P(return +2%) = 0.75

SHAP breakdown:
  rsi_14:           +0.08  (bullish signal)
  macd:             +0.05  (momentum up)
  realized_vol_10:  -0.02  (elevated vol)
  bb_position:      +0.03  (oversold bounce)
  ema_cross_9_21:   +0.02  (trend continuation)
```

### 2. Feature Importance Trends

How do feature contributions change over time?
- Week 1: RSI dominated predictions (high avg SHAP)
- Week 2: MACD became more important (volatility increase?)
- Week 3: Volume indicators mattered (regime shift?)

Tracks feature drift and model stability over rolling windows.

### 3. Strategy Contribution

Which strategy made money this month?

```
breakout:
  P&L: +₹2,500
  Trades: 10
  Win rate: 65%

meanrevert:
  P&L: +₹800
  Trades: 8
  Win rate: 50%
```

### 4. Loss/Profit Analysis

Top 10 winners and losers, with root cause analysis:
- **Winner pattern:** High signal confidence (0.8+), RSI extreme, MACD crossover
- **Loser pattern:** Low confidence (0.4-0.5), conflicting signals, rare features

### 5. Feature Correlation with Returns

Which features predict returns?

```
rsi_14:              +0.35  (best predictor)
macd_cross:          +0.28
realized_vol_10:     -0.15  (inverse: lower vol = better)
volume_zscore_20:    +0.12
```

Use to identify which features are actually predictive vs. noise.

### 6. Feature Drift Detection

Monitor for distribution shifts that signal regime change:
- RSI stopped working → sector rotation?
- Volume regime changed → liquidity crisis?
- Volatility spiked → earnings/news event?

Alerts notify team when features break down.

---

## Module Reference

### `monitoring/attribution.py`

#### PerformanceAttribution

Main class for SHAP-based attribution.

**Methods:**

```python
from monitoring.attribution import PerformanceAttribution
from datetime import date, timedelta

attr = PerformanceAttribution(lookback_days=90)

# 1. Compute SHAP values for predictions
shap_values = attr.compute_shap_values(
    model=my_xgboost_model,
    X_test=feature_df,
    y_test=label_series  # optional
)
# Returns: DataFrame (n_rows, n_features) with SHAP contributions

# 2. Feature importance trends over time
trend_df = attr.feature_importance_trend(
    date_range=(date(2024, 1, 1), date(2024, 1, 31)),
    window_days=7
)
# Returns: DataFrame (rolling windows) → (features with highest avg |SHAP|)

# 3. Strategy performance breakdown
strats = attr.strategy_contribution(
    symbol_or_all=None,  # None for all symbols
    lookback_days=30
)
# Returns: {
#     "breakout": {"pnl": 2500, "trades": 10, "win_rate": 0.65, ...},
#     "meanrevert": {"pnl": 800, "trades": 8, "win_rate": 0.50, ...},
# }

# 4. Top losing trades
losers = attr.loss_analysis(min_loss=-500)
# Returns: List[TradeRecord] with top 10 losers sorted by pnl

# 5. Top winning trades
winners = attr.profit_analysis(min_profit=500)
# Returns: List[TradeRecord] with top 10 winners sorted by pnl

# 6. Feature-return correlations
corr = attr.feature_correlation_with_returns(
    symbols=["SBIN", "HDFC"],  # optional
    lookback_days=90
)
# Returns: {"rsi_14": 0.35, "macd": 0.28, ...}

# 7. Full attribution report (integrates all above)
report = attr.generate_attribution_report(
    date_range=(date(2024, 1, 1), date(2024, 1, 31)),
    window_days=30
)
# Returns: AttributionReport dataclass with all metrics
```

**Interpreting SHAP Values:**

- **Large positive SHAP:** Feature pushed prediction toward "likely to win" (bullish)
- **Large negative SHAP:** Feature pushed prediction toward "unlikely to win" (bearish)
- **Small SHAP:** Feature had minimal impact on this prediction
- **Magnitude matters:** |0.15| is more important than |0.02|

---

### `monitoring/feature_drift_report.py`

Feature distribution monitoring and alerting.

#### FeatureDriftReporter

```python
from monitoring.feature_drift_report import FeatureDriftReporter
from datetime import date, timedelta

reporter = FeatureDriftReporter(alert_threshold=0.3)

# 1. Compute feature statistics for a period
stats = reporter.compute_feature_statistics(
    date_range=(date(2024, 1, 1), date(2024, 1, 7))
)
# Returns: FeatureStatistics with mean, std, min, max, median per feature

# 2. Detect distribution shifts using KL divergence
divergences = reporter.detect_distribution_shift(
    previous_stats=prev_week_stats,
    current_stats=curr_week_stats,
    n_bins=10
)
# Returns: {"rsi_14": 0.25, "macd": 0.18, ...}
# KL divergence > threshold → drift detected

# 3. Detect correlation breakdown
corr_changes = reporter.correlation_change_detection(
    date_range_1=(date(2024, 1, 1), date(2024, 1, 7)),
    date_range_2=(date(2024, 1, 8), date(2024, 1, 14)),
    threshold=0.3
)
# Returns: {
#     "rsi_14": {
#         "previous_corr": 0.40,
#         "current_corr": 0.05,
#         "change": 0.35  # ← above threshold!
#     }
# }

# 4. Generate alerts
alerts = reporter.alert_on_drift(
    current_stats=curr_week_stats,
    previous_stats=prev_week_stats
)
# Automatically sends Telegram if drift detected
# Returns: List[DriftAlert]

# 5. Generate weekly markdown report
report_md = reporter.generate_weekly_report(
    report_date=date(2024, 1, 14)
)
# Returns: Markdown with:
#   - Feature statistics table
#   - Distribution shifts ranked by KL divergence
#   - Correlation changes ranked by magnitude

# 6. Log report to MLflow
run_id = reporter.log_to_mlflow(report_md)
```

**Interpreting KL Divergence:**

- **0.0-0.1:** No shift, feature distribution stable
- **0.1-0.3:** Minor shift, monitor
- **0.3+:** Significant shift, alert + investigate
  - Possible causes: regime change, market event, data quality issue

---

### `monitoring/trade_review.py`

Post-trade review and learning.

#### TradeReviewEngine

```python
from monitoring.trade_review import TradeReviewEngine, TradeContext

engine = TradeReviewEngine()

# 1. Get full trade context
context = engine.get_trade_context(trade_id=42)
# Returns: TradeContext with entry/exit prices, features, signal_prob, etc.

# 2. Review trade and categorize outcome
review = engine.post_trade_review(
    trade=context,
    target_pct=2.0,   # Take-profit target
    stop_loss_pct=-1.5  # Stop-loss threshold
)
# Returns: TradeReview with:
#   - outcome: HIT_TARGET | STOPPED_OUT | TIME_DECAY | EXIT_SIGNAL
#   - root_cause: human-readable explanation
#   - key_features: top 5 features in this trade
#   - model_confidence: [0, 1]
#   - market_condition: "trending" | "ranging" | "volatile"

# 3. Identify winner patterns
winner_patterns = engine.identify_patterns_in_winners(
    lookback_days=30,
    min_profit=100
)
# Returns: {
#     "n_winners": 25,
#     "avg_signal_prob": 0.78,
#     "top_features": ["rsi_14", "macd_cross", ...],
#     "most_common_symbols": ["SBIN", "HDFC", ...],
#     "avg_pnl": 450.0,
# }

# 4. Identify loser patterns
loser_patterns = engine.identify_patterns_in_losers(
    lookback_days=30,
    max_loss=-100
)
# Returns: {
#     "n_losers": 12,
#     "avg_signal_prob": 0.52,
#     "risky_features": ["bb_position", "vol_regime", ...],
#     "risky_symbols": ["PENNY", "MICRO", ...],
#     "avg_pnl": -220.0,
#     "worst_pnl": -850.0,
# }

# 5. Generate lessons learned
lessons = engine.generate_lessons_learned(lookback_days=30)
# Returns: {
#     "model_bias": "possibly overconfident",
#     "conflicting_features": ["rsi_14", "bb_position"],
#     "avoid_symbols": ["PENNY", "MICRO"],
#     "confidence_calibration": "actual win rate 45% vs avg signal prob 65%",
#     "n_winners": 25,
#     "n_losers": 12,
# }
```

**Outcome Categories:**

| Outcome | Meaning | Action |
|---------|---------|--------|
| HIT_TARGET | Reached profit target | ✅ Ideal exit |
| STOPPED_OUT | Hit stop loss | Review risk sizing |
| TIME_DECAY | Long hold, didn't hit target | Check exit logic |
| EXIT_SIGNAL | Technical signal triggered exit | Verify signal quality |
| UNKNOWN | Exit price missing | Data issue |

---

### `monitoring/reporters.py` (Updated)

#### AttributionReport

Formats attribution data for human readability.

```python
from monitoring.reporters import AttributionReport

report_text = AttributionReport.generate(
    attribution={
        "total_trades": 15,
        "total_pnl": 2000,
        "win_rate": 0.60,
        "win_count": 9,
        "loss_count": 6,
        "top_features_by_shap": [("rsi_14", 0.15), ("macd", 0.12), ...],
        "strategy_contribution": {...},
        "feature_correlation_with_returns": {...},
        "top_winners": [TradeRecord(...), ...],
        "top_losers": [TradeRecord(...), ...],
    },
    date_range=("2024-01-01", "2024-01-31")
)

print(report_text)
# Output:
# 🔍 ATTRIBUTION REPORT — 2024-01-01 to 2024-01-31
#
# Summary:
#   Total trades: 15
#   P&L: +2000.00
#   Win rate: 60.0%
#   Wins/Losses: 9W / 6L
#
# Top Features by SHAP Importance:
#   1. rsi_14: 0.1500
#   2. macd: 0.1200
#   ...
```

---

## Best Practices

### 1. SHAP Interpretation

**Do:**
- Look at top 5 SHAP features per prediction
- Compare SHAP distributions across winners vs. losers
- Track whether the same features dominate consistently

**Don't:**
- Rely on single-trade SHAP values (noisy)
- Treat SHAP magnitude as feature "strength" (depends on scale)
- Ignore feature interactions (SHAP assumes no multicollinearity)

### 2. Drift Detection

**Alert Response:**
- KL div > 0.3: Investigate immediately
- Correlation change > 0.3: Feature likely broken
- Multiple features drifting: Regime change, reduce position size

**Common Causes:**
- Earnings season → volatility spike
- Fed announcement → vol regime change
- Holiday → low liquidity
- Data quality issue → check data source

### 3. Loss Analysis

**Pattern Review:**
1. Extract top 10 losers
2. Cluster by: symbol, time, features, market condition
3. Identify repeating triggers:
   - "Always lose on high volume spikes" → filter volumes
   - "Lose on RSI extremes" → adjust RSI threshold
   - "Lose in certain sectors" → add sector filter

**Action Items:**
- Reweight features that conflict with winners
- Increase stop loss for risky symbols
- Add market condition filter

### 4. Strategy Evaluation

**Monthly Checklist:**
1. Which strategy won most money? (contribution)
2. Which strategy has best win rate? (quality)
3. Which strategy has highest Sharpe? (consistency)
4. Which features drove the winners? (SHAP)
5. Which features appeared in losers? (pattern analysis)

**Rebalancing Decision:**
- If breakout > meanrevert: allocate 70% breakout, 30% meanrevert
- If meanrevert has 30% win rate: disable it temporarily

---

## Integration with MLflow

All reports are logged to MLflow for historical tracking.

```python
import mlflow
from monitoring.attribution import PerformanceAttribution

attr = PerformanceAttribution()
report = attr.generate_attribution_report()

mlflow.set_experiment("monthly_attribution")
with mlflow.start_run(run_name=f"attribution_{date.today()}"):
    mlflow.log_param("lookback_days", 30)
    mlflow.log_metric("total_pnl", report.total_pnl)
    mlflow.log_metric("win_rate", report.win_rate)
    # Features can be logged as nested dicts
    mlflow.log_dict(
        {
            "top_features": report.top_features_by_shap,
            "strategy_contrib": report.strategy_contribution,
        },
        "attribution_metrics.json"
    )
```

Query historical reports:
```python
runs = mlflow.search_runs(experiment_names=["monthly_attribution"])
for run in runs:
    print(f"Date: {run.start_time}, Win Rate: {run.data.metrics['win_rate']}")
```

---

## Troubleshooting

### Issue: SHAP values all zero

**Cause:** Model predictions not changing (constant prediction)
**Fix:**
1. Verify model is trained correctly
2. Check that X_test has variance in features
3. Ensure features are scaled properly

### Issue: No trades in date range

**Cause:** Date filtering or empty trades table
**Fix:**
1. Verify date range: `print(trades_df['time'].min(), trades_df['time'].max())`
2. Check trades exist: `SELECT COUNT(*) FROM paper_trades WHERE time >= ...`
3. Ensure features_used is populated

### Issue: Feature drift alerts spam

**Cause:** threshold too low (KL div sensitive to data quality)
**Fix:**
1. Increase threshold: `FeatureDriftReporter(alert_threshold=0.5)`
2. Check for data quality issues
3. Verify sampling consistency

### Issue: Correlation changes all NaN

**Cause:** Insufficient trades or zero variance in feature/returns
**Fix:**
1. Increase lookback_days
2. Filter by specific symbols
3. Check if returns all zero (no winning trades)

---

## Example Workflow

**Weekly Attribution Review:**

```python
from datetime import date, timedelta
from monitoring.attribution import PerformanceAttribution
from monitoring.feature_drift_report import FeatureDriftReporter
from monitoring.reporters import AttributionReport

# 1. Generate attribution report
attr = PerformanceAttribution(lookback_days=30)
report = attr.generate_attribution_report()

# 2. Generate drift report
drift = FeatureDriftReporter()
drift_report_md = drift.generate_weekly_report()

# 3. Format for Telegram
attr_text = AttributionReport.generate(
    report.__dict__,
    date_range=(str(report.period_start), str(report.period_end))
)

# 4. Send alerts
from monitoring.telegram_notifier import TelegramNotifier
notifier = TelegramNotifier()
notifier.send_alert(f"📊 Weekly Attribution:\n{attr_text}")

# 5. Log to MLflow
import mlflow
attr.log_to_mlflow(drift_report_md)
```

---

## See Also

- **signals/model.py** — SignalModel.explain() for SHAP computation
- **monitoring/mlflow_tracker.py** — MLflow integration
- **tests/test_attribution.py** — Comprehensive test suite
