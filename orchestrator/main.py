"""
TradingSystem — provider-agnostic orchestrator.

Supports three market modes controlled by ``MARKET_TYPE`` in ``.env``:

    equity  — NSE equities via Kite or Upstox, IST market hours
    crypto  — Top-N coins via Binance (24/7, no market hours)
    both    — Runs equity and crypto loops concurrently

Broker selection is fully automatic:
    PAPER_TRADE_MODE=true  →  PaperBrokerAdapter  (default — safe for dev)
    DATA_PROVIDER=kite     →  KiteBrokerAdapter
    DATA_PROVIDER=upstox   →  UpstoxBrokerAdapter
    DATA_PROVIDER=binance  →  PaperBrokerAdapter  (live crypto orders: future)

Entry point:
    python -m orchestrator.main
    python -m orchestrator.main --market equity
    python -m orchestrator.main --market crypto
    python -m orchestrator.main --market both
"""
from __future__ import annotations

import signal
import sys
import time
from typing import Any

import pandas as pd
import structlog

from config.settings import settings
from data.providers import get_crypto_provider, get_provider
from data.redis_keys import RedisKeys
from data.store import get_ohlcv, get_redis, get_universe
from execution.broker import get_broker_adapter
from execution.logger import TradeLogger
from execution.orders import OrderExecutor
from monitoring.health import HealthMonitor
from risk.breakers import CircuitBreaker
from risk.monitor import PortfolioMonitor
from risk.sizer import PositionSizer
from signals.features import FEATURE_COLUMNS, build_features
from signals.model import ModelRegistry, SignalModel
from signals.vcp_scanner import scan_vcp_universe

log = structlog.get_logger(__name__)

_TOP_N_EQUITY = 50
_TOP_N_CRYPTO = 20
_MIN_RETRAIN_WIN_RATE = 0.50


class TradingSystem:
    """
    Provider-agnostic orchestrator: data → features → sentiment → XGBoost → risk → execution.

    Args:
        market_type: ``"equity"`` | ``"crypto"`` | ``"both"``.
                     Defaults to ``settings.market_type``.
    """

    def __init__(self, market_type: str | None = None) -> None:
        self._market_type = (market_type or settings.market_type).lower()
        log.info(
            "trading_system_init",
            market=self._market_type,
            paper_mode=settings.paper_trade_mode,
            provider=settings.data_provider,
        )

        # ------------------------------------------------------------------
        # Data providers
        # ------------------------------------------------------------------
        if self._market_type == "crypto":
            self._equity_provider = None
            self._crypto_provider = get_crypto_provider()
        elif self._market_type == "both":
            self._equity_provider = get_provider()
            self._crypto_provider = get_crypto_provider()
        else:   # equity (default)
            self._equity_provider = get_provider()
            self._crypto_provider = None

        # ------------------------------------------------------------------
        # Broker adapter (paper, Kite, Upstox, or Binance paper)
        # ------------------------------------------------------------------
        self._broker = get_broker_adapter()
        capital = self._broker.get_balance() or settings.initial_capital

        # ------------------------------------------------------------------
        # Risk layer
        # ------------------------------------------------------------------
        position_pct = (
            settings.crypto_max_position_pct
            if self._market_type == "crypto"
            else settings.max_position_pct
        )
        self._circuit_breaker = CircuitBreaker(
            daily_limit=settings.daily_dd_limit,
            weekly_limit=settings.weekly_dd_limit,
        )
        self._sizer = PositionSizer(
            total_capital=capital,
            max_position_pct=position_pct,
        )
        self._monitor = PortfolioMonitor(initial_capital=capital)

        # ------------------------------------------------------------------
        # Execution
        # ------------------------------------------------------------------
        self._executor = OrderExecutor(
            broker=self._broker,
            circuit_breaker=self._circuit_breaker,
        )
        self._logger = TradeLogger()

        # ------------------------------------------------------------------
        # Signals
        # ------------------------------------------------------------------
        self._registry = ModelRegistry(tracking_uri=settings.mlflow_tracking_uri)
        self._model: SignalModel | None = None
        self._challenger_model: SignalModel | None = None
        self._crypto_model: SignalModel | None = None

        # A/B routing — challenger model loaded lazily on first use
        from orchestrator.ab_router import SignalRouter
        self._ab_router = SignalRouter(challenger_pct=settings.ab_test_pct)

        # Sentiment — equity only (crypto uses sources_crypto separately)
        self._sentiment = None
        if self._market_type != "crypto":
            from llm.pipeline import SentimentPipeline
            self._sentiment = SentimentPipeline()

        # ------------------------------------------------------------------
        # Universe cache
        # ------------------------------------------------------------------
        self._equity_universe: list[str] = []
        self._crypto_universe: list[str] = []
        self._vcp_candidates: list[dict] = []
        self._open_positions: set[str] = set()

        # Regime detector — used in trading_loop to gate signal emission
        from signals.regime import RegimeDetector
        self._regime_detector = RegimeDetector()
        self._current_regime = None

    # ------------------------------------------------------------------
    # Pre-market  (equity: 08:45 IST | crypto: called once at startup)
    # ------------------------------------------------------------------

    def pre_market_setup(self) -> None:
        """Prepare the system for the trading session."""
        log.info("pre_market_setup_start", market=self._market_type)

        # 1. Refresh auth tokens (no-op for paper / Binance)
        try:
            self._broker.refresh_auth()
        except Exception as exc:
            log.warning("auth_refresh_failed", error=str(exc))

        # 2. Load universe
        if self._market_type in ("equity", "both"):
            self._load_equity_universe()
        if self._market_type in ("crypto", "both"):
            self._load_crypto_universe()

        # 3. Run VCP scan across equity universe (CPU-bound; results cached for the session)
        if self._market_type in ("equity", "both") and self._equity_universe:
            try:
                self._vcp_candidates = scan_vcp_universe(self._equity_universe)
                log.info("vcp_scan_complete", candidates=len(self._vcp_candidates))
            except Exception as exc:
                log.error("vcp_scan_failed", error=str(exc))
                self._vcp_candidates = []

        # 4. Run sentiment for equity universe
        if self._sentiment and self._equity_universe:
            try:
                self._sentiment.run_daily(self._equity_universe)
            except Exception as exc:
                log.error("sentiment_pre_market_failed", error=str(exc))

        # 5. Load latest ML model from MLflow
        self._load_model()

        # 6. Reset daily circuit breaker
        capital = self._broker.get_balance() or settings.initial_capital
        self._circuit_breaker.reset_daily(current_capital=capital)

        # 7. Telegram health check (prepend yesterday's macro briefing if available)
        briefing_prefix = ""
        try:
            from llm.pipeline import LLMSentimentEngine
            cached = LLMSentimentEngine.get_cached_briefing()
            if cached:
                briefing_prefix = cached + "\n\n"
        except Exception:
            pass

        self._send_alert(
            f"{briefing_prefix}Pre-market OK | market={self._market_type} | "
            f"paper={settings.paper_trade_mode} | "
            f"capital={capital:,.0f} | "
            f"model={'OK' if self._model else 'MISSING'} | "
            f"vcp_candidates={len(self._vcp_candidates)}"
        )
        log.info("pre_market_setup_complete")

    # ------------------------------------------------------------------
    # Trading loop  (equity: every 5 min 09:15–15:25 | crypto: every 5 min 24/7)
    # ------------------------------------------------------------------

    def trading_loop(self) -> None:
        """One scan-signal-execute cycle for the active universe(s)."""
        try:
            # Redis kill switch — checked first so a running process can be halted dynamically
            try:
                if get_redis().get(RedisKeys.TRADING_KILL_SWITCH) == b"1":
                    log.warning("trading_loop_skipped_kill_switch_active")
                    return
            except Exception:
                pass

            if self._circuit_breaker.is_halted():
                log.warning("trading_loop_skipped_circuit_halted")
                return

            dd = self._monitor.get_drawdown()
            if dd["daily_dd"] > settings.daily_dd_limit:
                self._circuit_breaker.halt(
                    f"daily drawdown {dd['daily_dd']:.2%} exceeded {settings.daily_dd_limit:.2%}"
                )
                return

            # Equity cycle — regime-gated; suppression does NOT block crypto
            if self._market_type in ("equity", "both"):
                self._update_regime()
                if self._current_regime is not None and self._regime_detector.should_suppress_new_entries(
                    self._current_regime.state
                ):
                    log.info(
                        "equity_cycle_skipped_choppy_regime",
                        regime=self._current_regime.state.value,
                    )
                else:
                    self._run_equity_cycle()

            # Crypto cycle — always runs regardless of equity regime
            if self._market_type in ("crypto", "both"):
                self._run_crypto_cycle()

            self._monitor.get_drawdown()

        finally:
            # Heartbeat fires even on early returns so the health monitor stays informed
            try:
                HealthMonitor().write_heartbeat()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Post-market  (equity: 15:35 IST | crypto: daily midnight UTC)
    # ------------------------------------------------------------------

    def post_market_summary(self) -> None:
        log.info("post_market_summary_start")
        try:
            self._logger.daily_summary()
        except Exception as exc:
            log.error("daily_summary_failed", error=str(exc))
        try:
            self._check_model_drift()
        except Exception as exc:
            log.error("drift_check_failed", error=str(exc))
        try:
            self._check_feature_drift()
        except Exception as exc:
            log.error("feature_drift_check_failed", error=str(exc))
        try:
            from monitoring.reconciliation import DailyReconciler
            DailyReconciler(self._broker).reconcile(self._open_positions)
        except Exception as exc:
            log.error("reconciliation_failed", error=str(exc))
        try:
            from llm.pipeline import LLMSentimentEngine
            LLMSentimentEngine().generate_macro_briefing()
        except Exception as exc:
            log.error("macro_briefing_generation_failed", error=str(exc))
        self._open_positions.clear()
        log.info("post_market_summary_complete")

    # ------------------------------------------------------------------
    # Weekly / retrain
    # ------------------------------------------------------------------

    def reset_weekly(self) -> None:
        capital = self._broker.get_balance() or settings.initial_capital
        self._circuit_breaker.reset_weekly(current_capital=capital)
        log.info("weekly_reset_complete", capital=capital)

    def retrain_check(self) -> None:
        try:
            from monitoring.mlflow_tracker import ModelDriftMonitor
            score = ModelDriftMonitor().compare_live_vs_backtest(window_trades=20)
            if score > 0.3:
                log.warning("model_drift_detected", drift_score=score)
                self._send_alert(f"⚠️ Model drift detected — score={score:.3f}. Starting retrain…")
                self._auto_retrain_and_promote()
        except Exception as exc:
            log.error("retrain_check_failed", error=str(exc))

    def _auto_retrain_and_promote(self) -> None:
        """
        Run walk-forward retraining and auto-promote to Production if AUC ≥ 0.60.

        Intentionally conservative threshold: only promotes a genuinely better model.
        Runs synchronously — expected to complete within a few minutes on Sunday
        overnight when no trading is active.
        """
        try:
            import pandas as pd
            from signals.train import WalkForwardTrainer
            from signals.features import FEATURE_COLUMNS
            from signals.model import ModelRegistry

            log.info("auto_retrain_started")

            # Fetch recent equity OHLCV from TimescaleDB for retraining
            from data.store import get_engine
            engine = get_engine()
            df = pd.read_sql(
                "SELECT * FROM ohlcv WHERE time >= NOW() - INTERVAL '30 months' ORDER BY time",
                engine,
                parse_dates=["time"],
                index_col="time",
            )

            if len(df) < 500:
                log.warning("auto_retrain_skipped_insufficient_data", rows=len(df))
                self._send_alert("⚠️ Auto-retrain skipped — insufficient data (< 500 rows).")
                return

            trainer = WalkForwardTrainer(train_months=24, test_months=3)
            results = trainer.run(df, features=FEATURE_COLUMNS, label="label",
                                  experiment_name="nse_equity_signals_auto")
            trainer.save_drift_reference(df, FEATURE_COLUMNS)

            mean_auc = results.get("mean_auc", 0.0)
            log.info("auto_retrain_complete", mean_auc=round(mean_auc, 4), n_folds=results.get("n_folds"))

            if mean_auc < 0.60:
                log.info("auto_retrain_no_promote", mean_auc=mean_auc)
                self._send_alert(
                    f"🔁 Retrain complete — AUC={mean_auc:.4f} (below 0.60 threshold, not promoted)."
                )
                return

            # Promote best fold to Production in MLflow
            import mlflow
            folds = results.get("folds", [])
            if not folds:
                log.warning("auto_retrain_no_folds")
                self._send_alert("⚠️ Retrain complete but no fold data returned — not promoting.")
                return

            best_fold = max(folds, key=lambda f: f.get("auc", 0.0))
            best_run_id = best_fold.get("run_id", "")
            if not best_run_id:
                log.warning("auto_retrain_no_run_id")
                self._send_alert("⚠️ Retrain complete but best fold has no run_id — not promoting.")
                return

            version = self._registry.register_model(
                run_id=best_run_id,
                segment="equity",
                model_path=f"runs:/{best_run_id}/model",
            )

            client = mlflow.MlflowClient()
            client.transition_model_version_stage(
                name="trading_signal_equity",
                version=version,
                stage="Production",
                archive_existing_versions=True,
            )

            log.info("auto_promote_success", version=version, auc=round(mean_auc, 4))
            self._send_alert(
                f"✅ Auto-retrain promoted v{version} to Production — AUC={mean_auc:.4f}"
            )

        except Exception as exc:
            log.error("auto_retrain_failed", error=str(exc))
            self._send_alert(f"❌ Auto-retrain failed: {exc}")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        log.info("graceful_shutdown_initiated")
        for symbol in list(self._open_positions):
            try:
                self._executor.cancel_order(symbol)
            except Exception:
                pass
        log.info("graceful_shutdown_complete")

    # ------------------------------------------------------------------
    # Internal — equity cycle
    # ------------------------------------------------------------------

    def _run_equity_cycle(self) -> None:
        if not self._model or not self._model.is_healthy():
            log.warning("equity_cycle_skipped_no_model")
            return

        try:
            features_map = self._fetch_equity_features()
        except Exception as exc:
            log.error("equity_feature_fetch_failed", error=str(exc))
            return

        for symbol in self._equity_universe:
            if symbol in self._open_positions:
                continue
            feature_df = features_map.get(symbol)
            if feature_df is None:
                continue
            self._execute_signal(symbol, feature_df, asset_class="equity")

    # ------------------------------------------------------------------
    # Internal — crypto cycle
    # ------------------------------------------------------------------

    def _run_crypto_cycle(self) -> None:
        if not self._crypto_universe:
            log.debug("crypto_cycle_skipped_empty_universe")
            return

        for symbol in self._crypto_universe:
            if symbol in self._open_positions:
                continue
            try:
                self._execute_crypto_signal(symbol)
            except Exception as exc:
                log.error("crypto_signal_error", symbol=symbol, error=str(exc))

    def _execute_crypto_signal(self, symbol: str) -> None:
        """Full crypto signal pipeline: fetch → features → model → risk → order."""
        if not self._crypto_model:
            log.debug("crypto_signal_skipped_no_model", symbol=symbol)
            return

        try:
            from datetime import datetime, timedelta
            to_date = datetime.utcnow()
            from_date = to_date - timedelta(days=120)

            df = self._crypto_provider.fetch_historical(
                symbol, from_date, to_date, interval="day"
            )
            if df is None or len(df) < 60:
                log.debug("crypto_signal_insufficient_data", symbol=symbol, rows=len(df) if df is not None else 0)
                return

            features_df = build_features(df, include_labels=False)
            if features_df.empty:
                return

            last_row = features_df.iloc[[-1]].copy()
            probs = self._crypto_model.predict(last_row[FEATURE_COLUMNS])
            signal_prob = float(probs.iloc[0])

            self._logger.log_signal(
                symbol=symbol,
                features_dict=last_row[FEATURE_COLUMNS].iloc[0].to_dict(),
                signal_prob=signal_prob,
                action_taken="BUY" if signal_prob >= settings.crypto_signal_threshold else "SKIP",
            )

            if signal_prob < settings.crypto_signal_threshold or self._circuit_breaker.is_halted():
                return

            # Cross-asset penalty: reduce exposure when many positions are open
            n_open = len(self._open_positions)
            cross_asset_penalty = min(0.5, n_open * 0.1)  # 0.0 → 0.5, never eliminate

            current_price = float(last_row["close"].iloc[0])
            vol = float(last_row.get("realized_vol_20", pd.Series([0.30])).iloc[0])
            capital = self._broker.get_balance() or settings.initial_capital

            crypto_sizer = PositionSizer(
                total_capital=capital,
                max_position_pct=settings.crypto_max_position_pct,
            )
            size_inr = crypto_sizer.size(signal_prob, vol, capital, correlation_penalty=cross_asset_penalty)
            qty = crypto_sizer.shares(size_inr, current_price) if current_price > 0 else 0

            if qty <= 0:
                return

            order_id = self._executor.place_market_order(
                symbol=symbol,
                transaction_type="BUY",
                quantity=qty,
                tag=f"crypto_xgb_{signal_prob:.2f}",
            )
            self._open_positions.add(symbol)
            log.info("crypto_order_placed", symbol=symbol, qty=qty, prob=round(signal_prob, 3), order_id=order_id)
            self._send_alert(
                f"🪙 Crypto BUY {symbol} | prob={signal_prob:.2%} | "
                f"qty={qty} | price={current_price:,.4f} | penalty={cross_asset_penalty:.2f}"
            )

        except Exception as exc:
            log.error("crypto_signal_execution_error", symbol=symbol, error=str(exc))

    # ------------------------------------------------------------------
    # Internal — shared signal execution
    # ------------------------------------------------------------------

    def _execute_signal(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        asset_class: str = "equity",
    ) -> None:
        try:
            sentiment_score = (
                self._sentiment.get_latest_score(symbol)
                if self._sentiment else 0.0
            )
            last_row = feature_df.iloc[[-1]].copy()

            # A/B routing: route signal to champion (Production) or challenger (Staging) model
            date_str = last_row.index[-1].strftime("%Y-%m-%d") if hasattr(last_row.index[-1], "strftime") else str(last_row.index[-1])
            ab_slot = self._ab_router.route(symbol=symbol, date=date_str)
            model_to_use = self._model
            if ab_slot == "challenger":
                if self._challenger_model is None:
                    try:
                        self._challenger_model = self._registry.get_latest_model(segment="EQ", stage="Staging")
                    except RuntimeError:
                        ab_slot = "champion"   # fall back gracefully
                else:
                    model_to_use = self._challenger_model

            probs = model_to_use.predict(last_row[FEATURE_COLUMNS])
            signal_prob = float(probs.iloc[0])

            threshold = (
                settings.crypto_signal_threshold
                if asset_class == "crypto"
                else settings.signal_threshold
            )
            action = "BUY" if signal_prob >= threshold else "SKIP"

            self._logger.log_signal(
                symbol=symbol,
                features_dict=last_row[FEATURE_COLUMNS].iloc[0].to_dict(),
                signal_prob=signal_prob,
                action_taken=action,
            )

            if action == "SKIP" or self._circuit_breaker.is_halted():
                return

            # Earnings blackout — skip BUY during earnings window
            try:
                from signals.filters import EarningsFilter
                if EarningsFilter().is_blackout(symbol):
                    log.info("signal_skipped_earnings_blackout", symbol=symbol)
                    return
            except Exception:
                pass

            current_price = float(last_row["close"].iloc[0])
            vol = float(last_row.get("realized_vol_20", pd.Series([0.20])).iloc[0])
            capital = self._broker.get_balance() or settings.initial_capital
            size_inr = self._sizer.size(signal_prob, vol, capital)

            # Apply regime size multiplier (only for equity; crypto ignores regime for now)
            if asset_class == "equity" and self._current_regime is not None:
                mult = self._regime_detector.get_position_size_multiplier(self._current_regime.state)
                size_inr *= mult

            qty = self._sizer.shares(size_inr, current_price) if current_price > 0 else 0

            if qty <= 0:
                return

            order_id = self._executor.place_market_order(
                symbol=symbol,
                transaction_type="BUY",
                quantity=qty,
                tag=f"xgb_{signal_prob:.2f}",
            )
            self._open_positions.add(symbol)
            log.info("order_placed", symbol=symbol, qty=qty, prob=round(signal_prob, 3), order_id=order_id)

            # Buy alert with SHAP attribution
            try:
                shap_dict = self._model.explain(last_row[FEATURE_COLUMNS].iloc[0].to_dict())
                top_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                shap_str = ", ".join(f"{k}={v:+.3f}" for k, v in top_features)
                self._send_alert(
                    f"📈 BUY {symbol} | prob={signal_prob:.2%} | "
                    f"qty={qty} | price={current_price:,.2f}\nSHAP: {shap_str}"
                )
            except Exception:
                pass

        except Exception as exc:
            log.error("signal_execution_error", symbol=symbol, error=str(exc))

    # ------------------------------------------------------------------
    # Internal — universe loaders
    # ------------------------------------------------------------------

    def _load_equity_universe(self) -> None:
        try:
            instruments = get_universe(segment="EQ")
            self._equity_universe = [i["symbol"] for i in instruments[:_TOP_N_EQUITY]]
            log.info("equity_universe_loaded", count=len(self._equity_universe))
        except Exception as exc:
            log.error("equity_universe_load_failed", error=str(exc))

    def _load_crypto_universe(self) -> None:
        try:
            from data.universe_crypto import CryptoUniverse
            universe = CryptoUniverse(api_key=settings.coingecko_api_key)
            instruments = universe.get_tradeable(
                top_n=_TOP_N_CRYPTO,
                min_volume_usd=settings.crypto_min_volume_usd,
            )
            instrument_map = {i["symbol"]: i["pair"] for i in instruments}
            self._crypto_universe = list(instrument_map.keys())

            if self._crypto_provider:
                self._crypto_provider.register_instruments(instrument_map)

            log.info("crypto_universe_loaded", count=len(self._crypto_universe))
        except Exception as exc:
            log.error("crypto_universe_load_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Internal — helpers
    # ------------------------------------------------------------------

    def _update_regime(self) -> None:
        """Detect current market regime using Nifty 50 data + India VIX."""
        try:
            from datetime import datetime, timedelta
            from data.ingest import NSEDataScraper
            nifty_token = settings.nifty50_token  # set in .env / settings
            to_date = datetime.utcnow()
            nifty_df = get_ohlcv(nifty_token, to_date - timedelta(days=400), to_date, interval="day")
            if len(nifty_df) < 252:
                log.warning("regime_update_insufficient_data", rows=len(nifty_df))
                return
            india_vix = NSEDataScraper().get_india_vix()
            self._current_regime = self._regime_detector.detect(nifty_df, india_vix)
            log.info(
                "regime_updated",
                state=self._current_regime.state.value,
                adx=round(self._current_regime.adx_14, 1),
                vix=round(self._current_regime.vix, 2),
                score=round(self._current_regime.score, 2),
            )
        except Exception as exc:
            log.warning("regime_update_failed", error=str(exc))

    def _load_model(self) -> None:
        try:
            self._model = self._registry.get_latest_model(segment="EQ")
            log.info("model_loaded")
        except Exception as exc:
            log.warning("model_load_failed", error=str(exc))
            self._model = None

        if self._market_type in ("crypto", "both"):
            try:
                self._crypto_model = self._registry.get_latest_model(segment="CRYPTO")
                log.info("crypto_model_loaded")
            except Exception as exc:
                log.warning("crypto_model_load_failed", error=str(exc))
                self._crypto_model = None

    def _fetch_equity_features(self) -> dict[str, pd.DataFrame]:
        from datetime import datetime, timedelta
        feature_map: dict[str, pd.DataFrame] = {}
        to_date = datetime.utcnow()
        # 400 calendar days: ~280 trading days — enough for 252-bar warmup + signal rows
        from_date = to_date - timedelta(days=400)

        instruments = get_universe(segment="EQ")
        for inst in instruments[:_TOP_N_EQUITY]:
            token, symbol = inst["token"], inst["symbol"]
            try:
                df = get_ohlcv(token, from_date, to_date, interval="day")
                if len(df) < 60:
                    continue
                features = build_features(df, include_labels=False)
                if not features.empty:
                    feature_map[symbol] = features
            except Exception as exc:
                log.debug("equity_feature_error", symbol=symbol, error=str(exc))
        return feature_map

    def _check_model_drift(self) -> None:
        try:
            from monitoring.mlflow_tracker import ModelDriftMonitor
            score = ModelDriftMonitor().compare_live_vs_backtest(window_trades=20)
            if score > 0.3:
                log.warning("model_drift_above_threshold", score=score)
        except Exception as exc:
            log.debug("drift_check_skipped", error=str(exc))

    def _check_feature_drift(self) -> None:
        """Run KS-based feature drift check against training reference distribution."""
        if self._market_type not in ("equity", "both"):
            return
        try:
            from monitoring.drift_detector import ConceptDriftDetector
            feature_map = self._fetch_equity_features()
            if not feature_map:
                return
            sample_symbol = next(iter(feature_map))
            live_df = feature_map[sample_symbol]
            detector = ConceptDriftDetector()
            results = detector.check(live_df)
            if detector.is_drifting(live_df):
                drifted = [f for f, p in results.items() if p < 0.05]
                msg = f"⚠️ Feature drift detected in {len(drifted)} features: {', '.join(drifted[:5])}"
                log.warning("feature_drift_detected", drifted=drifted)
                self._send_alert(msg)
        except Exception as exc:
            log.debug("feature_drift_check_skipped", error=str(exc))

    def _send_alert(self, message: str) -> None:
        try:
            from monitoring.alerts import TelegramAlerter
            TelegramAlerter().send(message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NSE + Crypto Trading System")
    parser.add_argument(
        "--market",
        choices=["equity", "crypto", "both"],
        default=settings.market_type,
        help="Market to trade (default: from MARKET_TYPE env var)",
    )
    args = parser.parse_args()

    system = TradingSystem(market_type=args.market)
    scheduler = None

    def _handle_shutdown(signum, frame):
        log.info("shutdown_signal_received", signum=signum)
        system.shutdown()
        if scheduler:
            scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    from orchestrator.scheduler import TradingScheduler
    scheduler = TradingScheduler(system, market_type=args.market)
    scheduler.start()

    log.info("trading_system_running", market=args.market, paper_mode=settings.paper_trade_mode)
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        _handle_shutdown(None, None)


if __name__ == "__main__":
    main()
