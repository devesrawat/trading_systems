# Phase 8 — ML Evolution Integration Guide

**Complete walkthrough for ensemble models, A/B testing, feature engineering, concept drift detection, and automated retraining.**

---

## Overview

Phase 8 is the final phase of the trading system, integrating all advanced ML features into the production orchestrator:

- **Feature Engineering Pipeline** (`orchestrator/feature_engineer.py`) — Extract, validate, cache 30+ features
- **Ensemble Strategy** (`signals/training/ensemble_models.py`) — XGBoost + LightGBM + PatchTST voting
- **A/B Testing** (`orchestrator/ab_tester.py`) — Champion vs. Challenger 50/50 routing with statistical testing
- **Concept Drift Detection** (`signals/training/concept_drift.py`) — KL divergence-based drift detection
- **Walk-Forward Ensemble** (`signals/training/walk_forward_ensemble.py`) — Monthly retraining with drift checking
- **Hyperparameter Optimization** (`signals/training/hyperparameter_optimizer.py`) — Quarterly Bayesian HPO
- **Attribution & Reporting** (`monitoring/attribution.py`) — SHAP-based feature importance and trade analysis

---

## Architecture

```
Live Market Data (Kite/Binance)
    ↓
    ├─→ TimescaleDB (OHLCV)
    │   └─→ FeatureEngineer.extract_features()
    │       ├─ 30+ TA indicators (RSI, MACD, BB, ATR, etc.)
    │       ├─ Volatility metrics
    │       ├─ Regime indicators
    │       └─→ Redis cache (5 min TTL)
    ↓
AB Test Router (50/50)
    ├─→ Champion Ensemble (Production)
    │   └─→ Vote: XGBoost + LightGBM + PatchTST
    ├─→ Challenger Ensemble (Staging)
    │   └─→ Vote: XGBoost + LightGBM + PatchTST
    ↓
ABTestOrchestrator.log_ab_test_result()
    ├─→ TimescaleDB (permanent audit trail)
    └─→ Redis (30-day rolling window)
    ↓
Risk Layer (unchanged from Phase 1-7)
    ├─→ CircuitBreaker
    ├─→ PositionSizer (half-Kelly)
    └─→ ExecutionLayer
    ↓
Scheduled Jobs (scheduler.py)
    ├─→ Monthly (1st, 2 AM IST): WalkForwardEnsembleTrainer.run_walk_forward()
    ├─→ Weekly (Monday, 6 AM IST): ConceptDriftDetector.check()
    ├─→ Weekly (Friday, 5 PM IST): ABTestReporter.generate_weekly_report() + promotion
    └─→ Quarterly (1st of Q, 3 AM IST): BayesianHyperparameterOptimizer.optimize()
```

---

## 1. Feature Engineering

### FeatureEngineer Class

Located in: `orchestrator/feature_engineer.py`

**Core Methods:**

```python
from orchestrator.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()

# Extract features from OHLCV data
features = engineer.extract_features(symbol="INFY", df_ohlcv=df)
# Returns: pd.Series with 30+ features indexed by FEATURE_COLUMNS

# Validate features (NaN, outliers, inf)
is_valid = engineer.validate_features(features)

# Handle missing data (forward-fill, interpolate, drop)
cleaned = engineer.handle_missing_data(features)

# Cache in Redis for fast retrieval
engineer.cache_features(symbol="INFY", feature_series=features)

# All-in-one: extract → validate → cache
features = engineer.extract_and_validate(
    symbol="INFY",
    df_ohlcv=df,
    use_cache=True
)
```

### Feature Schema

The canonical feature list is defined in `signals/features.py::FEATURE_COLUMNS` (30+ features):

- **Momentum**: RSI (fast/slow), MACD, Stochastic
- **Volatility**: Bollinger Bands, ATR, realized volatility
- **Trend**: EMA (9, 21, 50), trend strength
- **Volume**: OBV, volume ratios
- **Regime**: Market regime (uptrend, downtrend, choppy)

**When adding new features:**

1. Add calculation to `signals/features.py::build_features()`
2. Add to `FEATURE_COLUMNS` list
3. **RETRAIN THE MODEL** — the XGBoost ensemble is coupled to this exact feature set
4. Update `monitoring/feature_drift_report.py` to track the new feature

```python
# Example: Add a new feature
# signals/features.py
FEATURE_COLUMNS = [
    # ... existing features ...
    "new_momentum_indicator",  # ← Add here
]

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    # ... existing logic ...

    # Add new feature
    features["new_momentum_indicator"] = compute_my_indicator(df)
    return features
```

---

## 2. Ensemble Model Training

### Manual Training

```python
from signals.training.ensemble_models import EnsembleStrategy
from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer
import pandas as pd

# Fetch training data
df = pd.read_sql(
    "SELECT * FROM ohlcv WHERE time >= NOW() - INTERVAL '60 months' ORDER BY time",
    engine,
    parse_dates=["time"],
    index_col="time",
)

# Create trainer
trainer = WalkForwardEnsembleTrainer(
    train_months=60,      # 5 years
    val_months=12,        # 1 year
    test_months=6,        # 6 months
    experiment_name="my_ensemble_v1",
)

# Run walk-forward training
report = trainer.run_walk_forward(
    symbols=["INFY", "TCS", "RELIANCE"],
    date_range=("2019-01-01", "2024-12-31"),
)

print(f"Aggregate AUC: {report.aggregate_auc:.4f}")
print(f"Folds: {report.total_folds}")
print(f"Drift events: {sum(1 for r in report.fold_results if r.drift_detected)}")
```

### Automatic Monthly Retraining

The scheduler automatically runs `WalkForwardEnsembleTrainer` on the **1st of each month at 2 AM IST**:

```python
# In orchestrator/scheduler.py (_monthly_ensemble_retrain)
self._scheduler.add_job(
    func=self._safe(self._monthly_ensemble_retrain),
    trigger=CronTrigger(hour=2, minute=0, day="1", timezone=_TZ_IST),
    id="monthly_ensemble_retrain",
)
```

This job:
1. Pulls 5 years of historical OHLCV from TimescaleDB
2. Runs walk-forward with expanding windows
3. Detects concept drift per fold
4. Logs all results to MLflow
5. Saves ensemble to `./models/` directory
6. Sends Telegram alert with aggregate metrics

---

## 3. A/B Testing

### How It Works

During `trading_loop()`, the system routes each signal 50/50 between:

- **Champion** (Production ensemble) — the model currently deployed
- **Challenger** (Staging ensemble) — a new candidate model

Each execution is logged with trade results (entry/exit, P&L, Sharpe ratio).

### ABTestOrchestrator

```python
from orchestrator.ab_tester import ABTestOrchestrator

ab_tester = ABTestOrchestrator()

# Route a signal (50/50 random)
model_name = ab_tester.route_signal_to_model(symbol="INFY")
# Returns: "champion" or "challenger"

# Log result after trade closes
ab_tester.log_ab_test_result(
    symbol="INFY",
    model_name="champion",
    entry_price=1200.0,
    exit_price=1240.0,
    pnl=40.0,
    sharpe=1.5,
    model_prediction=0.72,
    duration_minutes=120,
)

# Compare models
comparison = ab_tester.compare_models(lookback_days=30)
# Returns:
# {
#     "champion": {stats},
#     "challenger": {stats},
#     "p_value": 0.03,
#     "challenger_wins": True,  # if p < 0.05 & challenger_sharpe > champion_sharpe
# }

# Promote challenger if it won
if comparison["challenger_wins"]:
    ab_tester.promote_challenger_to_champion()
```

### Weekly A/B Test Report

Every **Friday at 5 PM IST**, the scheduler runs `_ab_test_reporting()`:

```python
# In orchestrator/scheduler.py
self._scheduler.add_job(
    func=self._safe(self._ab_test_reporting),
    trigger=CronTrigger(hour=17, minute=5, day_of_week="fri", timezone=_TZ_IST),
    id="ab_test_reporting",
)
```

This:
1. Fetches all A/B test results from last 7 days
2. Computes statistics (Sharpe, win rate, profit factor, P&L)
3. Runs two-sample t-test on Sharpe ratios
4. Sends Telegram report
5. **If challenger wins** → promotes to champion with congratulatory alert

---

## 4. Concept Drift Detection

### What It Is

Concept drift occurs when the statistical distribution of features or labels shifts—a sign that market conditions have changed and the model may no longer be reliable.

**Example:** In a trending market, momentum indicators work well. In choppy markets, they fail. If drift is detected → emergency retraining may be needed.

### ConceptDriftDetector

```python
from signals.training.concept_drift import ConceptDriftDetector
import pandas as pd

detector = ConceptDriftDetector(threshold=0.5)

# At training time: save reference distribution
detector.fit(df_training[FEATURE_COLUMNS])

# At inference time (daily): check for drift
drift_result = detector.check(df_recent[FEATURE_COLUMNS])
# Returns: {feature_name: p_value}  # low p → drift

# Check if regime has changed
is_drifting = detector.is_regime_change(
    kl_divergence=0.6,
    threshold=0.5  # if KL > 0.5, regime change detected
)
```

### Weekly Drift Check

Every **Monday at 6 AM IST**, the scheduler runs `_weekly_concept_drift_check()`:

```python
# In orchestrator/scheduler.py
self._scheduler.add_job(
    func=self._safe(self._weekly_concept_drift_check),
    trigger=CronTrigger(hour=6, minute=0, day_of_week="mon", timezone=_TZ_IST),
    id="weekly_concept_drift_check",
)
```

This:
1. Fetches last 30 days of OHLCV and features
2. Runs KS test per feature vs. training reference
3. Identifies features with p < 0.05 (drifted)
4. If drifts detected → sends Telegram alert
5. **If drift is severe** → triggers `retrain_check()` (may initiate emergency retraining)

---

## 5. Quarterly Hyperparameter Optimization

### BayesianHyperparameterOptimizer

Uses Optuna with Bayesian search to optimize XGBoost/LightGBM hyperparameters:

```python
from signals.training.hyperparameter_optimizer import BayesianHyperparameterOptimizer
import pandas as pd

optimizer = BayesianHyperparameterOptimizer(
    n_trials=20,
    warm_start=True,  # Use previous trials as seed
)

best_params, history = optimizer.optimize(
    X=df_features,
    y=df_labels,
)

print(f"Best params: {best_params}")
print(f"Best score: {history[-1]['score']:.4f}")
```

### Quarterly HPO Schedule

Every **1st of Q at 3 AM IST** (Jan 1, Apr 1, Jul 1, Oct 1):

```python
# In orchestrator/scheduler.py
for quarter_month, quarter_name in [(1, "Q1"), (4, "Q2"), (7, "Q3"), (10, "Q4")]:
    self._scheduler.add_job(
        func=self._safe(lambda q=quarter_name: self._quarterly_hpo(q)),
        trigger=CronTrigger(hour=3, minute=0, day="1", month=quarter_month, timezone=_TZ_IST),
        id=f"quarterly_hpo_{quarter_name}",
    )
```

This:
1. Fetches last 2 years of data
2. Runs Bayesian HPO for 20 trials
3. Logs best params and optimization curve to MLflow
4. Sends Telegram summary with best score
5. **Next retrain** will use these new hyperparams (if better)

---

## 6. Attribution & Performance Analysis

### PerformanceAttribution

Located in: `monitoring/attribution.py`

```python
from monitoring.attribution import PerformanceAttribution
import pandas as pd

attribution = PerformanceAttribution(model=trained_ensemble)

# Compute SHAP values for feature importance
shap_values = attribution.compute_shap_values(X_test)
# Returns: np.array of shape (n_samples, n_features)

# Feature importance over time
importance_trend = attribution.feature_importance_trend(
    X_rolling_windows,
    n_windows=52,  # 52 weeks
)

# Contribution by strategy
contrib = attribution.strategy_contribution(
    trades_df,
    group_by="strategy_name",
)

# Loss analysis: which features matter in losing trades?
top_losers = attribution.loss_analysis(X_test, y_test, top_n=5)

# Generate full attribution report
report = attribution.generate_attribution_report()
```

### FeatureDriftReporter

Track how feature distributions change over time:

```python
from monitoring.attribution import FeatureDriftReporter

reporter = FeatureDriftReporter()

# Detect distribution shift
drift_report = reporter.detect_distribution_shift(
    df_reference,
    df_current,
    alpha=0.05,  # KS test significance level
)

# Detect correlation changes
corr_change = reporter.correlation_change_detection(
    df_reference,
    df_current,
)

# Weekly report
weekly_report = reporter.weekly_report()
```

### TradeReviewEngine

Post-trade analysis and lesson learning:

```python
from monitoring.attribution import TradeReviewEngine

reviewer = TradeReviewEngine(model=trained_ensemble)

# Analyze winners
winner_patterns = reviewer.identify_patterns_in_winners(trades_df)
# Returns: {"top_features": [...], "regime": "uptrend", ...}

# Analyze losers
loser_patterns = reviewer.identify_patterns_in_losers(trades_df)

# Generate lessons learned
lessons = reviewer.generate_lessons_learned(trades_df)
```

---

## 7. Troubleshooting

### Problem: Concept drift detected

**Symptoms:** Weekly drift check alerts, multiple features with p < 0.05.

**Steps:**

1. Check market regime:
   ```python
   from signals.regime import RegimeDetector
   detector = RegimeDetector()
   regime = detector.detect(df_recent)
   print(f"Current regime: {regime}")
   ```

2. Examine feature distributions:
   ```python
   from monitoring.attribution import FeatureDriftReporter
   reporter = FeatureDriftReporter()
   report = reporter.detect_distribution_shift(df_train, df_recent)
   print(report)
   ```

3. Trigger manual retrain:
   ```python
   from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer
   trainer = WalkForwardEnsembleTrainer()
   report = trainer.run_walk_forward(
       symbols=["INFY", "TCS", "RELIANCE"],
       date_range=("2023-01-01", "2024-12-31"),
   )
   ```

### Problem: Challenger model performing worse

**Symptoms:** A/B test report shows challenger with lower Sharpe, wider confidence intervals.

**Steps:**

1. **Don't promote** — the system will automatically skip promotion if p > 0.05.

2. Check hyperparameters:
   ```python
   from signals.training.hyperparameter_optimizer import BayesianHyperparameterOptimizer
   # Run HPO to find better params
   ```

3. Analyze feature importance in challenger:
   ```python
   from monitoring.attribution import PerformanceAttribution
   attr = PerformanceAttribution(model=challenger_ensemble)
   shap = attr.compute_shap_values(X_test)
   print(attr.feature_importance_trend(...))
   ```

### Problem: Feature extraction fails on new symbol

**Symptoms:** `FeatureEngineer.extract_features()` raises ValueError.

**Steps:**

1. Check data freshness:
   ```python
   from data.store import get_ohlcv
   df = get_ohlcv(symbol="NEW_SYMBOL", start_date="2024-01-01")
   print(f"Rows: {len(df)}, Latest: {df.index[-1]}")
   ```

2. Ensure >= 50 rows:
   ```python
   if len(df) < 50:
       print("Insufficient data — wait for more bars")
   ```

3. Manually test feature extraction:
   ```python
   from orchestrator.feature_engineer import FeatureEngineer
   engineer = FeatureEngineer()
   features = engineer.extract_and_validate(
       symbol="NEW_SYMBOL",
       df_ohlcv=df,
       use_cache=False,  # Don't cache yet
   )
   print(features)
   ```

### Problem: Model degradation over time

**Symptoms:** Monthly retrain shows declining aggregate AUC, drift increasing.

**Steps:**

1. **Backtest latest model** on recent data:
   ```bash
   uv run python -m backtest.harness \
       --strategy ensemble \
       --start 2024-06-01 \
       --end 2024-12-31
   ```

2. **Run quarterly HPO** to find better hyperparams:
   ```python
   from signals.training.hyperparameter_optimizer import BayesianHyperparameterOptimizer
   optimizer = BayesianHyperparameterOptimizer(n_trials=50)
   best_params, history = optimizer.optimize(X, y)
   ```

3. **Add new features** if market has shifted:
   - Check correlation with forward returns
   - Add sentiment, macro indicators, volatility surface

4. **Investigate regime changes**:
   ```python
   from signals.regime import RegimeDetector
   detector = RegimeDetector()
   regime_history = detector.regime_history(df_recent, lookback_days=252)
   # Plot to visualize regime transitions
   ```

---

## 8. Complete Workflow Example: "Debug Why Strategy X Stopped Working"

Suppose your VCP (Volume-Price Breakout) strategy used to work great, but now it's losing trades.

### Step 1: Identify the problem

```python
from monitoring.attribution import TradeReviewEngine
from data.store import get_engine
import pandas as pd

# Fetch recent trades for VCP strategy
engine = get_engine()
trades = pd.read_sql("""
    SELECT * FROM trades
    WHERE strategy_name = 'vcp'
    AND entry_time >= NOW() - INTERVAL '30 days'
    ORDER BY entry_time DESC
""", engine)

print(f"Total trades: {len(trades)}")
print(f"Win rate: {trades['win'].sum() / len(trades):.1%}")
print(f"Avg P&L: {trades['pnl'].mean():.2f}")

# Most recent 10 trades
print(trades.tail(10)[['entry_time', 'entry_price', 'exit_price', 'pnl', 'win']])
```

### Step 2: Analyze feature patterns in recent losers

```python
from monitoring.attribution import TradeReviewEngine
from signals.training.ensemble_models import EnsembleStrategy

# Load ensemble model
ensemble = EnsembleStrategy()  # or load from registry

# Create reviewer
reviewer = TradeReviewEngine(model=ensemble)

# Analyze recent losses
recent_trades = trades[trades['entry_time'] >= (pd.Timestamp.utcnow() - pd.Timedelta(days=30))]
loser_patterns = reviewer.identify_patterns_in_losers(recent_trades)

print(f"Top 5 features in losing trades:")
for feature, importance in loser_patterns["top_features"][:5]:
    print(f"  {feature}: {importance:.3f}")

print(f"Regime during losses: {loser_patterns.get('regime', 'unknown')}")
```

### Step 3: Check for concept drift

```python
from signals.training.concept_drift import ConceptDriftDetector
from signals.features import FEATURE_COLUMNS
import pandas as pd

# Get reference distribution (from training)
detector = ConceptDriftDetector(threshold=0.5)
detector.fit(df_training[FEATURE_COLUMNS])

# Check recent features
drift_result = detector.check(df_recent_30days[FEATURE_COLUMNS])

# Which features have drifted?
drifted = {f: p for f, p in drift_result.items() if p < 0.05}
print(f"Drifted features: {list(drifted.keys())}")

# Example: VCP strategy might work on strong trends, but drift in volatility
# could indicate choppy market where breakouts fail
```

### Step 4: Compare with A/B test results

```python
from orchestrator.ab_tester import ABTestOrchestrator

ab_tester = ABTestOrchestrator()
comparison = ab_tester.compare_models(lookback_days=30)

print(f"Champion Sharpe: {comparison['champion']['avg_sharpe']:.2f}")
print(f"Challenger Sharpe: {comparison['challenger']['avg_sharpe']:.2f}")
print(f"Challenger wins: {comparison['challenger_wins']}")

# If challenger is winning, prepare to promote on next Friday
```

### Step 5: Decide on action

**Scenario A: Normal drawdown** (drift moderate, challenger OK)
- Nothing needed. Strategy naturally has periods of underperformance.
- Reduce position size temporarily if desired.

**Scenario B: Significant drift detected, challenger winning**
- Promote challenger on Friday (automatic).
- Monitor new model closely.

**Scenario C: Significant drift, both models losing**
- Market regime has fundamentally changed.
- Run `WalkForwardEnsembleTrainer` to retrain on recent data.
- Consider adding new features (e.g., VIX, sentiment).

**Scenario D: Specific feature failure**
- Example: "VCP_Volume is always low in recent trades"
- Either adjust strategy's volume threshold, or remove that feature and retrain.

---

## 9. Configuration

### Environment Variables (.env)

```bash
# Phase 8: Ensemble and A/B testing
AB_TEST_ENABLED=true
AB_TEST_PCT=0.5  # 50% to challenger
ENSEMBLE_STRATEGY=majority  # majority or weighted
CONCEPT_DRIFT_THRESHOLD=0.5
```

### strategy_params.yaml

```yaml
ensemble:
  champion_model_path: ./models/production/champion_ensemble_v3.pkl
  challenger_model_path: ./models/staging/challenger_ensemble_v4.pkl
  ab_test_enabled: true
  ab_test_enabled_for: [equity]
  ensemble_voting_strategy: majority
  xgb_weight: 0.4
  lgb_weight: 0.3
  patchtst_weight: 0.3

retraining:
  frequency: monthly
  lookback_years: 5
  train_window_years: 4
  val_window_years: 1
  test_window_months: 6
  concept_drift_threshold: 0.5
  trigger_on_drift: true
  quarterly_hpo_enabled: true
  hpo_n_trials: 20
  hpo_warm_start: true
```

---

## 10. Adding New Features

**Never break the pipeline!** Follow this exact protocol:

### Step 1: Add feature calculation

```python
# signals/features.py
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... existing code ...

    # NEW FEATURE
    features["my_new_indicator"] = calculate_new_indicator(df)
    return features
```

### Step 2: Update FEATURE_COLUMNS

```python
# signals/features.py
FEATURE_COLUMNS = [
    # ... existing 30 features ...
    "my_new_indicator",  # ← Add here
]
```

### Step 3: Verify on test data

```python
from orchestrator.feature_engineer import FeatureEngineer
from signals.features import FEATURE_COLUMNS

engineer = FeatureEngineer()
features = engineer.extract_and_validate("INFY", df_test)

# Check all features present
missing = set(FEATURE_COLUMNS) - set(features.index)
assert len(missing) == 0, f"Missing features: {missing}"

# Check no NaN
assert features.isna().sum() == 0, "NaN in features"
```

### Step 4: Retrain ensemble

```bash
# Run manual retraining with new feature
uv run python -c "
from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer
trainer = WalkForwardEnsembleTrainer(experiment_name='with_new_feature')
report = trainer.run_walk_forward(
    symbols=['INFY', 'TCS'],
    date_range=('2019-01-01', '2024-12-31')
)
print(f'New model AUC: {report.aggregate_auc:.4f}')
"
```

### Step 5: Backtest

```bash
uv run pytest tests/test_features_engineering.py -v -k "my_new_indicator"
uv run python -m backtest.harness --strategy ensemble --start 2024-01-01 --end 2024-12-31
```

### Step 6: Deploy

If AUC ≥ 0.60 and backtest looks good:
```bash
uv run mlflow runs promote --stage Production  # via CLI or API
```

---

## 11. Maintenance Checklist

**Weekly:**
- [ ] Check Telegram alerts for concept drift
- [ ] Verify A/B test report (Friday 5 PM IST)
- [ ] Monitor trade count and win rate

**Monthly:**
- [ ] Retrain ensemble completes successfully (1st, 2 AM IST)
- [ ] Review aggregate AUC trend
- [ ] Check drift detection results

**Quarterly:**
- [ ] HPO completes (1st of Q, 3 AM IST)
- [ ] Review feature importance trends
- [ ] Update strategy parameters if needed

**Annually:**
- [ ] Full backtest on new year's data
- [ ] Audit Phase 8 components for drift/degradation
- [ ] Consider adding new features

---

## 12. API Reference

### FeatureEngineer

```python
from orchestrator.feature_engineer import FeatureEngineer

engineer = FeatureEngineer()

# Extract all features
features = engineer.extract_features(symbol, df_ohlcv) → pd.Series

# Validate
is_valid = engineer.validate_features(features) → bool

# Handle missing
cleaned = engineer.handle_missing_data(features) → pd.Series

# Cache in Redis
engineer.cache_features(symbol, features) → None

# Retrieve from cache
cached = engineer.get_cached_features(symbol) → pd.Series | None

# All-in-one
features = engineer.extract_and_validate(symbol, df_ohlcv, use_cache=True) → pd.Series | None
```

### EnsembleStrategy

```python
from signals.training.ensemble_models import EnsembleStrategy

ensemble = EnsembleStrategy()

# Train
ensemble.train_xgboost(X_train, y_train, hyperparams)
ensemble.train_lgb(X_train, y_train, hyperparams)
ensemble.train_patchtst(X_train, y_train, hyperparams)

# Predict (voting)
predictions = ensemble.predict(X_test) → np.array

# Get feature importance
importance = ensemble.get_feature_importance() → dict

# Save/load
ensemble.save_to_pickle("path")
ensemble = EnsembleStrategy.load_from_pickle("path")
```

### ABTestOrchestrator

```python
from orchestrator.ab_tester import ABTestOrchestrator

ab_tester = ABTestOrchestrator()

# Route signal
model = ab_tester.route_signal_to_model(symbol) → "champion" | "challenger"

# Log result
ab_tester.log_ab_test_result(symbol, model_name, entry, exit, pnl, sharpe, prediction)

# Compare
comparison = ab_tester.compare_models(lookback_days=30) → dict

# Promote
ab_tester.promote_challenger_to_champion() → bool

# Rollback
ab_tester.roll_back_to_previous_champion(reason="regression") → bool

# Stats
stats = ab_tester.get_ab_test_stats() → dict
```

### ConceptDriftDetector

```python
from signals.training.concept_drift import ConceptDriftDetector

detector = ConceptDriftDetector(threshold=0.5)

# Fit (training time)
detector.fit(df_train[FEATURE_COLUMNS])

# Check (inference time)
drift = detector.check(df_recent[FEATURE_COLUMNS]) → dict

# Is regime change?
is_changing = detector.is_regime_change(kl_div=0.6, threshold=0.5) → bool

# KL divergence
kl = detector.compute_kl_divergence(dist1, dist2) → float
```

### WalkForwardEnsembleTrainer

```python
from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer

trainer = WalkForwardEnsembleTrainer(
    train_months=60,
    val_months=12,
    test_months=6,
    experiment_name="my_run"
)

# Run
report = trainer.run_walk_forward(symbols, date_range) → TrainingReport

# Report contains:
report.fold_results → list[WalkForwardResult]
report.aggregate_auc → float
report.best_hyperparams → dict
report.feature_importance → dict
```

---

## 13. Common Commands

```bash
# Train ensemble manually
uv run python -c "
from signals.training.walk_forward_ensemble import WalkForwardEnsembleTrainer
trainer = WalkForwardEnsembleTrainer()
report = trainer.run_walk_forward(['INFY', 'TCS'], ('2019-01-01', '2024-12-31'))
print(f'AUC: {report.aggregate_auc:.4f}')
"

# Run concept drift check
uv run python -c "
from signals.training.concept_drift import ConceptDriftDetector
from data.store import get_engine
from signals.features import FEATURE_COLUMNS
import pandas as pd

detector = ConceptDriftDetector()
engine = get_engine()
df = pd.read_sql('SELECT * FROM ohlcv WHERE time >= NOW() - INTERVAL 30 days', engine)
drift = detector.check(df[FEATURE_COLUMNS])
print({f: p for f, p in drift.items() if p < 0.05})
"

# Check A/B test stats
uv run python -c "
from orchestrator.ab_tester import ABTestOrchestrator
ab = ABTestOrchestrator()
stats = ab.get_ab_test_stats()
print(f'Champion Sharpe: {stats[\"champion\"].get(\"avg_sharpe\", 0):.2f}')
print(f'Challenger Sharpe: {stats[\"challenger\"].get(\"avg_sharpe\", 0):.2f}')
"

# Run tests
uv run pytest tests/test_orchestrator_ml_evolution.py -v
uv run pytest tests/test_ab_testing.py -v

# Start orchestrator with scheduler
uv run python -m orchestrator.main --market equity

# Full lint + test
uv run ruff format .
uv run ruff check . --fix
uv run mypy . --strict
uv run pytest tests/ -q
```

---

## 14. Backward Compatibility

**Phase 8 is 100% backward compatible with Phase 1-7.**

- All existing tests pass unchanged.
- Old signal processing logic untouched.
- Risk, execution, and broker adapters unchanged.
- Ensemble is **optional** (can disable with `ab_test_enabled=false`).

When `ab_test_enabled=false`:
- System falls back to single champion model
- A/B routing disabled
- No ensemble overhead

---

## Next Steps

1. **Deploy Phase 8:** Run full test suite, deploy to staging
2. **Monitor:** Watch for concept drift, A/B test metrics
3. **Optimize:** Use quarterly HPO to tune hyperparams
4. **Iterate:** Add features, retrain, promote better models

---

**Questions?** See `CLAUDE.md` for Phase 8 commands, or reach out to the trading system maintainers.
