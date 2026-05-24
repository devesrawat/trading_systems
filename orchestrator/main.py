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

import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import date, datetime

import numpy as np
import pandas as pd
import structlog

from config.settings import settings
from data.providers import get_crypto_provider, get_provider
from data.redis_keys import RedisKeys
from data.store import get_ohlcv, get_ohlcv_batch, get_redis, get_universe
from execution.broker import get_broker_adapter
from execution.logger import TradeLogger
from execution.orders import OrderExecutor
from monitoring.health import HealthMonitor
from orchestrator.feature_engineer import FeatureEngineer
from portfolio.risk_manager import PreExecutionRiskCheck
from portfolio.schema import PortfolioPosition, PortfolioState
from risk.breakers import CircuitBreaker
from risk.monitor import PortfolioMonitor
from risk.sizer import PositionSizer
from signals.alpha_composite import AlphaEngine
from signals.contracts import Signal
from signals.features import FEATURE_COLUMNS, build_features
from signals.model import ModelRegistry, SignalModel
from signals.registry import StrategyRegistry
from signals.scanner_engine import ScannerEngine
from signals.signal_router import normalize_strategy_result
from signals.training.ensemble_models import EnsembleStrategy

log = structlog.get_logger(__name__)

_TOP_N_EQUITY = 50
_TOP_N_CRYPTO = 20
_MIN_RETRAIN_WIN_RATE = 0.50


# ---------------------------------------------------------------------------
# Mode gating
# ---------------------------------------------------------------------------


def _mode_gate(
    signal_mode: str,
    paper_trade_mode: bool,
    circuit_breaker_halted: bool,
) -> tuple[bool, str | None]:
    """
    Determine if a signal should be executed based on mode and system state.

    Parameters
    ----------
    signal_mode : str
        Signal mode (research, watchlist, paper, live)
    paper_trade_mode : bool
        Whether the system is in paper trading mode
    circuit_breaker_halted : bool
        Whether the circuit breaker is halted

    Returns
    -------
    tuple[bool, str | None]
        (allowed, reason_rejected) where reason_rejected is None if allowed,
        or a string explaining why the signal was rejected
    """
    if signal_mode == "research":
        return False, "research mode signals not executed"

    if signal_mode == "watchlist":
        return False, "watchlist mode signals not executed"

    if signal_mode == "paper":
        if not paper_trade_mode:
            return False, "paper mode signal attempted in live trading"
        if circuit_breaker_halted:
            return False, "circuit breaker halted"
        return True, None

    if signal_mode == "live":
        if paper_trade_mode:
            return False, "live mode signal attempted in paper trading"
        if circuit_breaker_halted:
            return False, "circuit breaker halted"
        return True, None

    return False, f"unknown signal mode: {signal_mode}"


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
        else:  # equity (default)
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

        # Phase 9 Quick Win 3: Portfolio risk checker (pre-execution validation)
        self._risk_checker = PreExecutionRiskCheck()

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

        # A/B testing orchestrator — champion vs. challenger comparison
        from orchestrator.ab_tester import ABTestOrchestrator
        from orchestrator.model_registry import ModelRegistry as ABModelRegistry

        self._ab_tester = ABTestOrchestrator(registry=ABModelRegistry())

        # Legacy A/B routing — challenger model loaded lazily on first use
        from orchestrator.ab_router import SignalRouter

        self._ab_router = SignalRouter(challenger_pct=settings.ab_test_pct)

        # ------------------------------------------------------------------
        # Phase 10: Multi-Strategy Alpha Engine
        # ------------------------------------------------------------------
        self._alpha_engine = AlphaEngine(kite=getattr(self._equity_provider, "kite", None))

        # Phase 8: Feature engineering and ensemble models
        self._feature_engineer = FeatureEngineer()
        self._ensemble_model: EnsembleStrategy | None = None

        # Sentiment — equity only (crypto uses sources_crypto separately)
        self._sentiment = None
        if self._market_type != "crypto":
            from llm.pipeline import SentimentPipeline

            self._sentiment = SentimentPipeline()

        # ------------------------------------------------------------------
        # Universe cache
        # ------------------------------------------------------------------
        self._equity_universe: list[str] = []
        self._equity_instruments: list[dict] = []  # full instrument dicts (token + symbol)
        self._crypto_universe: list[str] = []
        self._scan_candidates: list[dict] = []

        # Load persisted open positions from Redis so restarts don't lose tracking
        self._open_positions: set[str] = set()
        try:
            members = get_redis().smembers(RedisKeys.OPEN_POSITIONS)
            self._open_positions = set(members)
            if self._open_positions:
                log.info("positions_restored_from_redis", count=len(self._open_positions))
        except Exception as _exc:
            log.warning("position_restore_failed", error=str(_exc))

        # Cached capital — refreshed once per trading_loop to avoid per-symbol broker calls
        self._cached_capital: float = settings.initial_capital

        # Regime detector — used in trading_loop to gate signal emission
        from signals.regime import RegimeDetector

        self._regime_detector = RegimeDetector()
        self._current_regime = None
        self._regime_last_updated: datetime | None = None

    # ------------------------------------------------------------------
    # Pre-market  (equity: 08:45 IST | crypto: called once at startup)
    # ------------------------------------------------------------------

    def pre_market_setup(self) -> None:
        """Prepare the system for the trading session."""
        log.info("pre_market_setup_start", market=self._market_type)

        # QUICK WIN 1: Clear prior session state to prevent memory accumulation
        self._scan_candidates.clear()
        log.debug("state_cleared", cleared_scan_candidates=len(self._scan_candidates))

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

        # 3. Run enabled strategies against equity universe using ScannerEngine
        if self._market_type in ("equity", "both") and self._equity_universe:
            try:
                registry = StrategyRegistry()
                enabled = registry.group_by_interval_asset_class()

                # Collect all strategies for all intervals
                all_strategies = []
                for (_asset_class, _interval), strategies in enabled.items():
                    all_strategies.extend(strategies)

                if all_strategies:
                    engine = ScannerEngine(strategies=all_strategies)
                    raw_results = engine.run(self._equity_universe)

                    # Normalize all strategy results to Signal objects
                    for strategy_name, result_list in raw_results.items():
                        for result_dict in result_list:
                            try:
                                signal = normalize_strategy_result(
                                    strategy_result=result_dict,
                                    symbol=result_dict.get("symbol", ""),
                                    strategy_name=strategy_name,
                                    confidence=result_dict.get("confidence", 0.5),
                                    mode="research",  # Strategy scanner hits are research-mode by default
                                )
                                # Store as raw dicts for backward compatibility (to be phased out)
                                self._scan_candidates.append(result_dict)
                                log.info(
                                    "strategy_signal_generated",
                                    signal_id=signal.signal_id,
                                    symbol=signal.symbol,
                                    strategy=strategy_name,
                                    confidence=signal.confidence,
                                )
                            except Exception as e:
                                log.warning(
                                    "strategy_result_normalization_failed",
                                    strategy=strategy_name,
                                    error=str(e),
                                )
                    log.info(
                        "strategies_scan_complete",
                        strategies=len(all_strategies),
                        results={k: len(v) for k, v in raw_results.items()},
                    )
                else:
                    log.warning("no_enabled_strategies_found")
                    self._scan_candidates = []
            except Exception as exc:
                log.error("strategies_scan_failed", error=str(exc))
                self._scan_candidates = []

        # 4. Run sentiment for equity universe
        if self._sentiment and self._equity_universe:
            try:
                self._sentiment.run_daily(self._equity_universe)
            except Exception as exc:
                log.error("sentiment_pre_market_failed", error=str(exc))

        # 5. Load latest ML model from MLflow
        self._load_model()

        # 5a. Phase 8: Load champion and challenger ensemble models (if A/B testing enabled)
        self._load_ensemble_models()

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
            f"scan_candidates={len(self._scan_candidates)}"
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

            # Refresh capital once per loop — avoids per-symbol broker API round-trips
            with suppress(Exception):  # keep stale value from previous iteration on error
                self._cached_capital = self._broker.get_balance() or settings.initial_capital

            # Process time-based exits first (before scanning new entries)
            if self._market_type in ("equity", "both"):
                with suppress(Exception):
                    self._process_exits()

            # Equity cycle — regime-gated; suppression does NOT block crypto
            if self._market_type in ("equity", "both"):
                self._update_regime()
                if (
                    self._current_regime is not None
                    and self._regime_detector.should_suppress_new_entries(
                        self._current_regime.state
                    )
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

            # Phase 9 Quick Win 1: Batch flush circuit breaker state at end of cycle
            with suppress(Exception):
                self._circuit_breaker.batch_flush()

        finally:
            # Heartbeat fires even on early returns so the health monitor stays informed
            with suppress(Exception):
                HealthMonitor().write_heartbeat()

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
        try:
            r = get_redis()
            for symbol in list(self._open_positions):
                r.srem(RedisKeys.OPEN_POSITIONS, symbol)
                r.delete(RedisKeys.position_meta(symbol))
        except Exception as exc:
            log.warning("positions_redis_clear_failed", error=str(exc))
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

            from signals.features import FEATURE_COLUMNS
            from signals.train import WalkForwardTrainer

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
            results = trainer.run(
                df,
                features=FEATURE_COLUMNS,
                label="label",
                experiment_name="nse_equity_signals_auto",
            )
            trainer.save_drift_reference(df, FEATURE_COLUMNS)

            mean_auc = results.get("mean_auc", 0.0)
            log.info(
                "auto_retrain_complete", mean_auc=round(mean_auc, 4), n_folds=results.get("n_folds")
            )

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
            with suppress(Exception):
                self._executor.cancel_order(symbol)
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

        # Batch model predictions: collect all features and predict once
        # instead of predicting per-symbol (eliminates per-call validation overhead)
        batch_predictions = self._batch_predict_equity(features_map)

        for symbol in self._equity_universe:
            if symbol in self._open_positions:
                continue
            feature_df = features_map.get(symbol)
            if feature_df is None:
                continue

            # Retrieve pre-computed prediction instead of calling model.predict()
            signal_prob = batch_predictions.get(symbol)
            self._execute_signal(
                symbol, feature_df, asset_class="equity", precomputed_signal_prob=signal_prob
            )

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

            df = self._crypto_provider.fetch_historical(symbol, from_date, to_date, interval="day")
            if df is None or len(df) < 60:
                log.debug(
                    "crypto_signal_insufficient_data",
                    symbol=symbol,
                    rows=len(df) if df is not None else 0,
                )
                return

            df._symbol = symbol
            features_df = build_features(df, include_labels=False)
            if features_df.empty:
                return

            last_row = features_df.iloc[[-1]]
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
            capital = self._cached_capital

            crypto_sizer = PositionSizer(
                total_capital=capital,
                max_position_pct=settings.crypto_max_position_pct,
            )
            size_inr = crypto_sizer.size(
                signal_prob, vol, capital, correlation_penalty=cross_asset_penalty
            )
            qty = crypto_sizer.shares(size_inr, current_price) if current_price > 0 else 0

            if qty <= 0:
                return

            order_id = self._executor.place_market_order(
                symbol=symbol,
                transaction_type="BUY",
                quantity=qty,
                tag=f"crypto_xgb_{signal_prob:.2f}",
                price=current_price,
            )
            self._add_position(symbol, current_price, signal_prob, qty)
            log.info(
                "crypto_order_placed",
                symbol=symbol,
                qty=qty,
                prob=round(signal_prob, 3),
                order_id=order_id,
            )
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
        signal_or_symbol: str | Signal,
        feature_df: pd.DataFrame | None = None,
        asset_class: str = "equity",
        precomputed_signal_prob: float | None = None,
    ) -> None:
        """
        Execute a signal with mode gating.

        Supports both old interface (symbol, feature_df) and new (Signal).

        Parameters
        ----------
        precomputed_signal_prob : float, optional
            If provided, use this probability instead of calling model.predict().
            Used for batch prediction optimization.
        """
        # Handle both old and new interfaces
        if isinstance(signal_or_symbol, Signal):
            signal = signal_or_symbol
            symbol = signal.symbol
        else:
            # Legacy interface: (symbol, feature_df) tuple
            symbol = signal_or_symbol
            signal = None

        try:
            # Mode gating: reject research/watchlist, check paper/live conditions
            if signal:
                allowed, reason_rejected = _mode_gate(
                    signal_mode=signal.mode,
                    paper_trade_mode=settings.paper_trade_mode,
                    circuit_breaker_halted=self._circuit_breaker.is_halted(),
                )

                log.info(
                    "signal_mode_check",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    strategy=signal.strategy_name,
                    mode=signal.mode,
                    allowed=allowed,
                    reason=reason_rejected,
                )

                if not allowed:
                    # Log rejected signal but don't execute
                    self._logger.log_signal(
                        symbol=symbol,
                        features_dict=signal.features,
                        signal_prob=signal.confidence,
                        action_taken="REJECTED",
                    )
                    return

            # For legacy interface (feature_df provided), run XGBoost model
            if feature_df is not None:
                self._execute_legacy_signal(
                    symbol, feature_df, asset_class, precomputed_signal_prob
                )
            elif signal:
                # Signal-based execution: use signal's confidence and entry specs
                self._execute_signal_based(signal, asset_class)

        except Exception as exc:
            log.error("signal_execution_error", symbol=symbol, error=str(exc))

    def _get_portfolio_snapshot(self) -> PortfolioState:
        """
        Build current portfolio state for risk checks.

        Phase 9 Quick Win 3: Used by PreExecutionRiskCheck before every order.
        """
        positions = {}
        for symbol, pos_data in self._monitor._positions.items():
            positions[symbol] = PortfolioPosition(
                symbol=symbol,
                qty=pos_data["qty"],
                entry_price=pos_data["avg_price"],
                current_price=pos_data["current_price"],
            )

        return PortfolioState(
            positions=positions,
            total_capital=self._monitor._initial_capital,
            cash_available=self._cached_capital - sum(p.market_value for p in positions.values()),
        )

    # ------------------------------------------------------------------
    # Position lifecycle helpers (Redis-backed, restart-safe)
    # ------------------------------------------------------------------

    def _add_position(
        self, symbol: str, entry_price: float, entry_prob: float, quantity: int
    ) -> None:
        self._open_positions.add(symbol)
        try:
            r = get_redis()
            r.sadd(RedisKeys.OPEN_POSITIONS, symbol)
            r.hset(
                RedisKeys.position_meta(symbol),
                mapping={
                    "entry_price": str(entry_price),
                    "entry_date": date.today().isoformat(),
                    "entry_prob": str(entry_prob),
                    "quantity": str(quantity),
                },
            )
        except Exception as exc:
            log.warning("position_persist_failed", symbol=symbol, error=str(exc))

    def _remove_position(self, symbol: str) -> None:
        self._open_positions.discard(symbol)
        try:
            r = get_redis()
            r.srem(RedisKeys.OPEN_POSITIONS, symbol)
            r.delete(RedisKeys.position_meta(symbol))
        except Exception as exc:
            log.warning("position_remove_failed", symbol=symbol, error=str(exc))

    def _process_exits(self) -> None:
        """Exit positions held >= 5 business days (matches forward_days: 5 label)."""
        today = date.today()
        r = get_redis()
        for symbol in list(self._open_positions):
            try:
                meta = r.hgetall(RedisKeys.position_meta(symbol))
                if not meta:
                    continue
                entry_date_str = meta.get("entry_date")
                if not entry_date_str:
                    continue
                entry_date = date.fromisoformat(entry_date_str)
                elapsed = int(np.busday_count(entry_date.isoformat(), today.isoformat()))
                if elapsed < 5:
                    continue

                qty = int(meta.get("quantity") or "0")

                # Try to get current close from live day bar; fall back to 0.0
                exit_price = 0.0
                try:
                    bar_raw = r.get(f"bar:day:{symbol}")
                    if bar_raw:
                        exit_price = float(json.loads(bar_raw).get("close") or 0.0)
                except Exception:
                    pass

                if qty > 0:
                    self._executor.place_market_order(
                        symbol=symbol,
                        transaction_type="SELL",
                        quantity=qty,
                        tag=f"exit_t5_{elapsed}d",
                        price=exit_price,
                    )
                    log.info(
                        "time_exit_executed",
                        symbol=symbol,
                        elapsed_bdays=elapsed,
                        exit_price=exit_price,
                    )
                    self._send_alert(
                        f"📤 EXIT {symbol} | held={elapsed} bdays | price={exit_price:,.2f}"
                    )
                self._remove_position(symbol)
            except Exception as exc:
                log.error("position_exit_error", symbol=symbol, error=str(exc))

    def _execute_legacy_signal(
        self,
        symbol: str,
        feature_df: pd.DataFrame,
        asset_class: str = "equity",
        precomputed_signal_prob: float | None = None,
    ) -> None:
        """Legacy XGBoost-based execution (for backward compatibility)."""
        try:
            sentiment_score = self._sentiment.get_latest_score(symbol) if self._sentiment else 0.0  # noqa: F841 — logged in signal metadata (future)
            last_row = feature_df.iloc[[-1]]

            # Use precomputed probability if available (from batch predict),
            # otherwise call model.predict() (backward compatible with old code)
            if precomputed_signal_prob is not None:
                signal_prob = precomputed_signal_prob
            else:
                # A/B routing: route signal to champion (Production) or challenger (Staging) model
                date_str = (
                    last_row.index[-1].strftime("%Y-%m-%d")
                    if hasattr(last_row.index[-1], "strftime")
                    else str(last_row.index[-1])
                )
                ab_slot = self._ab_router.route(symbol=symbol, date=date_str)
                model_to_use = self._model
                if ab_slot == "challenger":
                    if self._challenger_model is None:
                        try:
                            self._challenger_model = self._registry.get_latest_model(
                                segment="EQ", stage="Staging"
                            )
                        except RuntimeError:
                            ab_slot = "champion"  # fall back gracefully
                    else:
                        model_to_use = self._challenger_model

                probs = model_to_use.predict(last_row[FEATURE_COLUMNS])
                signal_prob = float(probs.iloc[0])

            # Apply Multi-Strategy Alpha Multiplier
            if asset_class == "equity":
                from portfolio.exposure import get_sector_for_symbol

                sector = get_sector_for_symbol(symbol)
                regime_state = (
                    self._current_regime.state.value if self._current_regime else "normal"
                )

                alpha_multiplier = self._alpha_engine.calculate_multiplier(
                    symbol=symbol, sector=sector, current_regime=regime_state, side="BUY"
                )
                signal_prob *= alpha_multiplier
                log.info(
                    "alpha_multiplier_applied",
                    symbol=symbol,
                    multiplier=round(alpha_multiplier, 2),
                    final_prob=round(signal_prob, 3),
                )

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
            capital = self._cached_capital
            size_inr = self._sizer.size(signal_prob, vol, capital)

            # Apply regime size multiplier (only for equity; crypto ignores regime for now)
            if asset_class == "equity" and self._current_regime is not None:
                mult = self._regime_detector.get_position_size_multiplier(
                    self._current_regime.state
                )
                size_inr *= mult

            qty = self._sizer.shares(size_inr, current_price) if current_price > 0 else 0

            if qty <= 0:
                return

            # Phase 9 Quick Win 3: Portfolio risk check before execution
            # Note: For legacy signal path, minimal risk check is done in the portfolio
            # check; full checks only apply to new Signal-based execution
            try:
                portfolio = self._get_portfolio_snapshot()
                # Quick sector concentration check for legacy path
                if portfolio.positions.get(symbol) is None:  # New position
                    sector_exposure = (size_inr / portfolio.total_capital) * 100
                    # Allow if under 5% single-position limit (loose check)
                    if sector_exposure > 5.0:
                        log.info(
                            "signal_rejected_portfolio_risk",
                            symbol=symbol,
                            reason=f"single_position_limit_exceeded: {sector_exposure:.1f}%",
                            prob=round(signal_prob, 3),
                        )
                        return
            except Exception as exc:
                log.warning("portfolio_risk_check_failed", symbol=symbol, error=str(exc))
                # Continue on risk check error (fail-open to avoid blocking trades)

            order_id = self._executor.place_market_order(
                symbol=symbol,
                transaction_type="BUY",
                quantity=qty,
                tag=f"xgb_{signal_prob:.2f}",
                price=current_price,
            )
            self._add_position(symbol, current_price, signal_prob, qty)
            log.info(
                "order_placed",
                symbol=symbol,
                qty=qty,
                prob=round(signal_prob, 3),
                order_id=order_id,
            )

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
            log.error("legacy_signal_execution_error", symbol=symbol, error=str(exc))

    def _execute_signal_based(
        self,
        signal: Signal,
        asset_class: str = "equity",
    ) -> None:
        """Execute a Signal object with unified contract."""
        try:
            symbol = signal.symbol

            # Apply Multi-Strategy Alpha Multiplier
            if asset_class == "equity":
                from portfolio.exposure import get_sector_for_symbol

                sector = get_sector_for_symbol(symbol)
                regime_state = (
                    self._current_regime.state.value if self._current_regime else "normal"
                )

                alpha_multiplier = self._alpha_engine.calculate_multiplier(
                    symbol=symbol, sector=sector, current_regime=regime_state, side="BUY"
                )
                signal.confidence *= alpha_multiplier
                log.info(
                    "alpha_multiplier_applied_signal",
                    symbol=symbol,
                    multiplier=round(alpha_multiplier, 2),
                    final_conf=round(signal.confidence, 3),
                )

            # Log the signal
            self._logger.log_signal(
                symbol=symbol,
                features_dict=signal.features,
                signal_prob=signal.confidence,
                action_taken="BUY" if signal.confidence >= settings.signal_threshold else "SKIP",
            )

            if signal.confidence < settings.signal_threshold:
                return

            # Earnings blackout for equity
            if asset_class == "equity":
                try:
                    from signals.filters import EarningsFilter

                    if EarningsFilter().is_blackout(symbol):
                        log.info("signal_skipped_earnings_blackout", symbol=symbol)
                        return
                except Exception:
                    pass

            # Extract entry specifications
            if signal.entry is None:
                log.warning("signal_missing_entry_spec", signal_id=signal.signal_id, symbol=symbol)
                return

            entry_price = signal.entry.entry_price
            current_price = entry_price  # Use signal's entry price

            # Risk sizing from signal
            size_pct = signal.risk.size_hint_pct if signal.risk else 0.01
            capital = self._cached_capital
            size_inr = capital * size_pct

            qty = self._sizer.shares(size_inr, current_price) if current_price > 0 else 0

            if qty <= 0:
                return

            # Phase 9 Quick Win 3: Portfolio risk check before execution
            try:
                portfolio = self._get_portfolio_snapshot()
                risk_decision = self._risk_checker.check_signal_execution(signal, portfolio)
                if not risk_decision.allowed:
                    log.info(
                        "signal_rejected_portfolio_risk",
                        signal_id=signal.signal_id,
                        symbol=symbol,
                        reason=risk_decision.reason,
                        confidence=round(signal.confidence, 3),
                    )
                    return
            except Exception as exc:
                log.warning(
                    "portfolio_risk_check_failed",
                    signal_id=signal.signal_id,
                    symbol=symbol,
                    error=str(exc),
                )
                # Continue on risk check error (fail-open to avoid blocking trades)

            order_id = self._executor.place_market_order(
                symbol=symbol,
                transaction_type="BUY",
                quantity=qty,
                tag=f"signal_{signal.strategy_name}_{signal.confidence:.2f}",
                price=current_price,
            )
            self._add_position(symbol, current_price, signal.confidence, qty)
            log.info(
                "signal_order_placed",
                signal_id=signal.signal_id,
                symbol=symbol,
                strategy=signal.strategy_name,
                qty=qty,
                confidence=round(signal.confidence, 3),
                order_id=order_id,
            )

            # Buy alert
            self._send_alert(
                f"📈 {signal.strategy_name.upper()} BUY {symbol} | "
                f"confidence={signal.confidence:.2%} | qty={qty} | price={current_price:,.2f}"
            )

        except Exception as exc:
            log.error("signal_based_execution_error", symbol=signal.symbol, error=str(exc))

    # ------------------------------------------------------------------
    # Internal — universe loaders
    # ------------------------------------------------------------------

    def _load_equity_universe(self) -> None:
        try:
            instruments = get_universe(segment="EQ")
            self._equity_instruments = instruments[:_TOP_N_EQUITY]
            self._equity_universe = [i["symbol"] for i in self._equity_instruments]
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
        """Detect current market regime using Nifty 50 data + India VIX.

        Results are cached for 60 minutes — regime state doesn't change bar-by-bar.
        """
        now = datetime.utcnow()
        if (
            self._current_regime is not None
            and self._regime_last_updated is not None
            and (now - self._regime_last_updated).total_seconds() < 3600
        ):
            return
        try:
            from datetime import timedelta

            from data.ingest import NSEDataScraper

            nifty_token = settings.nifty50_token  # set in .env / settings
            nifty_df = get_ohlcv(nifty_token, now - timedelta(days=400), now, interval="day")
            if len(nifty_df) < 252:
                log.warning("regime_update_insufficient_data", rows=len(nifty_df))
                return
            india_vix = NSEDataScraper().get_india_vix()
            self._current_regime = self._regime_detector.detect(nifty_df, india_vix)
            self._regime_last_updated = now
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

    def _load_ensemble_models(self) -> None:
        """Phase 8: Load champion and challenger ensemble models for A/B testing."""
        try:
            if not settings.ab_test_enabled:
                log.info("ensemble_loading_skipped_ab_test_disabled")
                return

            # Try to load champion ensemble from model registry
            try:
                champion = self._ab_tester._registry.get_champion()
                if champion:
                    log.info("ensemble_champion_loaded", version=champion.get("version"))
            except Exception as exc:
                log.warning("ensemble_champion_load_failed", error=str(exc))

            # Try to load challenger ensemble from model registry
            try:
                challenger = self._ab_tester._registry.get_challenger()
                if challenger:
                    log.info("ensemble_challenger_loaded", version=challenger.get("version"))
            except Exception as exc:
                log.warning("ensemble_challenger_load_failed", error=str(exc))

        except Exception as exc:
            log.error("ensemble_loading_failed", error=str(exc))

    def _batch_predict_equity(self, features_map: dict[str, pd.DataFrame]) -> dict[str, float]:
        """
        Batch predict signal probabilities for all symbols at once.

        Instead of calling model.predict() per-symbol (which re-validates features
        each time), concatenate all features and predict once. This eliminates
        O(N) validation overhead and leverages vectorized inference.

        Returns
        -------
        dict[str, float]
            Mapping of symbol → signal_probability
        """
        if not features_map or not self._model:
            return {}

        try:
            # Collect all last rows and symbols in order
            symbols: list[str] = []
            feature_list: list[pd.DataFrame] = []

            for symbol, feature_df in features_map.items():
                if feature_df is not None and not feature_df.empty:
                    symbols.append(symbol)
                    # Extract last row (most recent bar)
                    feature_list.append(feature_df.iloc[[-1]][FEATURE_COLUMNS])

            if not symbols:
                return {}

            # Concatenate all last rows into single batch DataFrame
            batch_features = pd.concat(feature_list, ignore_index=False)

            # Single model.predict() call on entire batch (validates once)
            probs = self._model.predict(batch_features)

            # Map probabilities back to symbols
            result = {symbol: float(prob) for symbol, prob in zip(symbols, probs, strict=True)}
            log.debug("batch_predict_complete", n_symbols=len(symbols))
            return result

        except Exception as exc:
            log.error("batch_predict_failed", error=str(exc))
            return {}

    def _fetch_equity_features(self) -> dict[str, pd.DataFrame]:
        from datetime import timedelta

        to_date = datetime.utcnow()
        # 400 calendar days: ~280 trading days — enough for 252-bar warmup + signal rows
        from_date = to_date - timedelta(days=400)

        instruments = self._equity_instruments or get_universe(segment="EQ")[:_TOP_N_EQUITY]
        tokens = [i["token"] for i in instruments]
        token_to_symbol = {i["token"]: i["symbol"] for i in instruments}

        # Single round-trip to TimescaleDB for all tokens
        ohlcv_map = get_ohlcv_batch(tokens, from_date, to_date, interval="day")

        def _build_one(token: int) -> tuple[str, pd.DataFrame | None]:
            symbol = token_to_symbol[token]
            df = ohlcv_map.get(token)
            if df is None or len(df) < 60:
                return symbol, None
            try:
                # Set symbol as attribute for cache key generation
                df._symbol = symbol
                features = build_features(df, include_labels=False)
                return symbol, features if not features.empty else None
            except Exception as exc:
                log.debug("equity_feature_error", symbol=symbol, error=str(exc))
                return symbol, None

        feature_map: dict[str, pd.DataFrame] = {}
        # Parallel feature building — CPU-bound pandas operations
        with ThreadPoolExecutor(max_workers=8) as pool:
            for symbol, features in pool.map(_build_one, tokens):
                if features is not None:
                    feature_map[symbol] = features
        return feature_map

    # ------------------------------------------------------------------
    # Hermes Wealth Architect — weekly scan
    # ------------------------------------------------------------------

    def run_wealth_scan(self) -> None:
        """
        Hermes Wealth Architect: screen the Nifty 50 universe for blue-chip
        compounding candidates and send results to Telegram.

        Criteria: PE < sector average AND ROE > 15%.
        Reads fundamentals from Redis cache (key ``FUND:{symbol}``).
        Symbols without cached data are skipped silently.

        Scheduled: Saturday 09:00 IST via TradingScheduler.
        """
        try:
            from signals.wealth_architect_scanner import WealthArchitectScanner

            universe = self._equity_instruments or get_universe(segment="EQ")[:_TOP_N_EQUITY]
            symbols = [i["symbol"] for i in universe]

            scanner = WealthArchitectScanner()
            candidates = scanner.run(symbols)

            if not candidates:
                msg = "Hermes Wealth Architect: no candidates passed PE/ROE filter this week."
            else:
                top3 = candidates[:3]
                lines = ["Hermes Wealth Architect — Weekly SIP Candidates\n"]
                for rank, c in enumerate(top3, start=1):
                    lines.append(
                        f"{rank}. {c['symbol']}  ROE={c['roe']}%  "
                        f"PE={c['pe']} vs sector avg {c['sector_avg_pe']} "
                        f"({c['pe_discount_pct']}% discount)  [{c['sector']}]"
                    )
                if len(candidates) > 3:
                    lines.append(f"\n+{len(candidates) - 3} more passed the filter.")
                msg = "\n".join(lines)

            log.info(
                "wealth_architect_scan_done",
                total_candidates=len(candidates),
                top3=[c["symbol"] for c in candidates[:3]],
            )
            self._send_alert(msg)

        except Exception as exc:
            log.error("wealth_architect_scan_failed", error=str(exc))

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
                msg = (
                    f"⚠️ Feature drift detected in {len(drifted)} features: {', '.join(drifted[:5])}"
                )
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
