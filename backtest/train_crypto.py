#!/usr/bin/env python
"""
Crypto signal model training script.

Fetches 2-year daily OHLCV from Binance public REST API for BTC, ETH, SOL,
builds features using the *same* pipeline as equity models (no schema drift),
and trains via WalkForwardTrainer, registering the best fold's model as
the CRYPTO segment in MLflow.

Usage
-----
    uv run python backtest/train_crypto.py
    uv run python backtest/train_crypto.py --symbols BTCUSDT ETHUSDT SOLUSDT
    uv run python backtest/train_crypto.py --min-auc 0.58
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests
import structlog

# Ensure project root is importable when running directly
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from signals.features import FEATURE_COLUMNS, build_features
from signals.model import ModelRegistry
from signals.train import WalkForwardTrainer

log = structlog.get_logger(__name__)

_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
_TWO_YEARS_DAYS = 730
_DEFAULT_MIN_AUC = 0.58


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_binance_ohlcv(symbol: str, days: int = _TWO_YEARS_DAYS) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Binance public API for the last *days* days.

    Returns a DataFrame with columns: open, high, low, close, volume
    indexed by UTC date.
    """
    limit = min(days, 1000)   # Binance max per request
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": limit,
    }
    resp = requests.get(_BINANCE_KLINES, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "n_trades",
        "taker_buy_base", "taker_buy_quote", "_ignore",
    ])
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df = df.set_index("time")
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    log.info("binance_ohlcv_fetched", symbol=symbol, rows=len(df))
    return df[["open", "high", "low", "close", "volume"]]


def _build_labelled_df(symbol: str, days: int = _TWO_YEARS_DAYS) -> pd.DataFrame:
    """
    Fetch OHLCV, compute features, and create a binary label.

    Label: 1 if next-bar return > 0 (long-only signal).
    Only rows where all FEATURE_COLUMNS are present are kept.
    """
    raw = _fetch_binance_ohlcv(symbol, days=days)
    df = build_features(raw)

    # Forward return label — 1 if next close > current close
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # Drop last row (no forward return available) and any NaN features
    df = df.iloc[:-1]
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = set(FEATURE_COLUMNS) - set(available)
    if missing:
        log.warning("crypto_features_missing", symbol=symbol, missing=sorted(missing))

    df = df.dropna(subset=available + ["label"])
    log.info("crypto_dataset_built", symbol=symbol, rows=len(df), features=len(available))
    return df


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_crypto(
    symbols: list[str] | None = None,
    min_auc: float = _DEFAULT_MIN_AUC,
    dry_run: bool = False,
) -> dict[str, object]:
    """
    Train walk-forward crypto signal model and optionally register in MLflow.

    Parameters
    ----------
    symbols  : Binance pair symbols. Defaults to BTC/ETH/SOL (USDT).
    min_auc  : Minimum mean AUC to promote model to MLflow Staging.
    dry_run  : If True, train but do not register in MLflow.

    Returns
    -------
    Walk-forward results dict from ``WalkForwardTrainer.run()``.
    """
    symbols = symbols or _DEFAULT_SYMBOLS

    # Combine all symbols into one multi-asset training frame
    frames = []
    for sym in symbols:
        try:
            df = _build_labelled_df(sym)
            df["symbol"] = sym
            frames.append(df)
        except Exception as exc:
            log.error("crypto_fetch_failed", symbol=sym, error=str(exc))

    if not frames:
        raise RuntimeError("No crypto data fetched — check network or symbol list.")

    combined = pd.concat(frames).sort_index()
    log.info("crypto_combined_dataset", rows=len(combined), symbols=symbols)

    if len(combined) < 300:
        raise RuntimeError(f"Insufficient data ({len(combined)} rows) for walk-forward training.")

    # Use same FEATURE_COLUMNS as equity — no schema mismatch at inference
    available_features = [c for c in FEATURE_COLUMNS if c in combined.columns]

    trainer = WalkForwardTrainer(train_months=18, test_months=3)
    results = trainer.run(
        combined,
        features=available_features,
        label="label",
        experiment_name="crypto_signals",
    )
    trainer.save_drift_reference(combined, available_features)

    mean_auc = results.get("mean_auc", 0.0)
    log.info(
        "crypto_training_complete",
        mean_auc=round(mean_auc, 4),
        n_folds=results.get("n_folds"),
        symbols=symbols,
    )

    if dry_run:
        print(f"Dry run — model NOT registered. mean_auc={mean_auc:.4f}")
        return results

    if mean_auc < min_auc:
        print(
            f"mean_auc={mean_auc:.4f} < threshold={min_auc}. Model NOT registered.\n"
            "Increase training data or tune hyperparameters."
        )
        return results

    # Register best fold model as CRYPTO segment → Staging (manual review before Production)
    registry = ModelRegistry()
    try:
        best = max(results["folds"], key=lambda r: r.get("auc", 0.0))
        run_id = best["run_id"]
        version = registry.register_model(
            run_id=run_id,
            segment="CRYPTO",
            model_path=f"runs:/{run_id}/model",
        )

        import mlflow
        mlflow.MlflowClient().transition_model_version_stage(
            name="trading_signal_crypto",
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        print(
            f"✅  Registered crypto model v{version} → Staging  "
            f"(AUC={mean_auc:.4f}). Promote to Production manually after review."
        )
    except Exception as exc:
        log.error("crypto_registration_failed", error=str(exc))
        print(f"❌  Registration failed: {exc}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train crypto signal models on Binance OHLCV.")
    p.add_argument(
        "--symbols", nargs="+", default=_DEFAULT_SYMBOLS, metavar="SYM",
        help="Binance symbols to train on (default: BTCUSDT ETHUSDT SOLUSDT)",
    )
    p.add_argument(
        "--min-auc", type=float, default=_DEFAULT_MIN_AUC, metavar="F",
        help=f"Minimum mean AUC to register model (default: {_DEFAULT_MIN_AUC})",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Train but skip MLflow registration",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = train_crypto(
        symbols=args.symbols,
        min_auc=args.min_auc,
        dry_run=args.dry_run,
    )
    print(
        f"Walk-forward results: "
        f"mean_auc={results.get('mean_auc', 0):.4f} "
        f"(±{results.get('std_auc', 0):.4f}), "
        f"folds={results.get('n_folds')}"
    )
