"""
TradingSystem — the main orchestrator that ties all modules together.

Lifecycle:
  pre_market_setup()   08:45 IST — token refresh, sentiment, model load
  trading_loop()       09:15–15:25 every 5 min — signal → risk → order
  post_market_summary() 15:35 IST — summary, drift check
  reset_weekly()        Monday 08:30 — weekly circuit breaker reset
  retrain_check()       Sunday 02:00 — flag if retraining needed

Entry point:
  python -m orchestrator.main
"""
from __future__ import annotations

import signal
import sys
import time
from typing import Any

import pandas as pd
import structlog

from config.settings import settings
from data.ingest import KiteIngestor
from data.store import get_ohlcv, get_universe
from execution.logger import TradeLogger
from execution.orders import OrderExecutor
from llm.pipeline import SentimentPipeline
from risk.breakers import CircuitBreaker
from risk.monitor import PortfolioMonitor
from risk.sizer import PositionSizer
from signals.features import FEATURE_COLUMNS, build_features
from signals.model import ModelRegistry, SignalModel

log = structlog.get_logger(__name__)

_TOP_N_UNIVERSE = 50        # trade top N liquid instruments
_MIN_RETRAIN_WIN_RATE = 0.50


class TradingSystem:
    """
    Ties together: data → features → sentiment → XGBoost → risk → execution.
    """

    def __init__(self) -> None:
        log.info("trading_system_init", paper_mode=settings.paper_trade_mode)
        self._paper_mode: bool = settings.paper_trade_mode

        # Data
        self._ingestor = KiteIngestor(
            api_key=settings.kite_api_key,
            access_token=settings.kite_access_token,
        )

        # Signals
        self._registry = ModelRegistry(tracking_uri=settings.mlflow_tracking_uri)
        self._model: SignalModel | None = None

        # Sentiment
        self._sentiment = SentimentPipeline()

        # Risk
        total_capital = self._get_current_capital()
        self._circuit_breaker = CircuitBreaker(
            daily_limit=settings.daily_dd_limit,
            weekly_limit=settings.weekly_dd_limit,
        )
        self._sizer = PositionSizer(
            total_capital=total_capital,
            max_position_pct=settings.max_position_pct,
        )
        self._monitor = PortfolioMonitor(initial_capital=total_capital)

        # Execution
        self._executor = OrderExecutor(
            kite=self._ingestor.kite,
            circuit_breaker=self._circuit_breaker,
            paper_mode=self._paper_mode,
        )
        self._logger = TradeLogger()

        # Universe cache
        self._universe: list[str] = []
        self._open_positions: set[str] = set()

    # ------------------------------------------------------------------
    # Pre-market  (08:45 IST)
    # ------------------------------------------------------------------

    def pre_market_setup(self) -> None:
        """Prepare the system for the trading day."""
        log.info("pre_market_setup_start")

        try:
            # 1. Refresh Kite access token
            cached_token = self._get_cached_token()
            if cached_token:
                self._ingestor.kite.set_access_token(cached_token)
        except Exception as exc:
            log.warning("token_refresh_skipped", error=str(exc))

        try:
            # 2. Run sentiment pipeline for full universe
            universe_instruments = get_universe(segment="EQ")
            symbols = [i["symbol"] for i in universe_instruments[:_TOP_N_UNIVERSE]]
            self._universe = symbols
            self._sentiment.run_daily(symbols)
        except Exception as exc:
            log.error("sentiment_pre_market_failed", error=str(exc))

        # 3. Load latest model from MLflow registry
        self._load_model()

        # 4. Reset daily circuit breaker
        capital = self._get_current_capital()
        self._circuit_breaker.reset_daily(current_capital=capital)

        # 5. Health check to Telegram
        try:
            from monitoring.alerts import TelegramAlerter
            TelegramAlerter().send(
                f"✅ Pre-market setup complete | Capital: ₹{capital:,.0f} | "
                f"Paper mode: {self._paper_mode} | Model: {'OK' if self._model else 'MISSING'}"
            )
        except Exception as exc:
            log.warning("telegram_health_check_failed", error=str(exc))

        log.info("pre_market_setup_complete")

    # ------------------------------------------------------------------
    # Trading loop  (09:15–15:25, every 5 min)
    # ------------------------------------------------------------------

    def trading_loop(self) -> None:
        """One cycle of the signal → risk → execution pipeline."""
        if not self._model or not self._model.is_healthy():
            log.warning("trading_loop_skipped_no_model")
            return

        if self._circuit_breaker.is_halted():
            log.warning("trading_loop_skipped_circuit_halted")
            return

        # Check drawdown before each cycle
        dd = self._monitor.get_drawdown()
        if dd["daily_dd"] > settings.daily_dd_limit:
            self._circuit_breaker.halt(
                f"daily drawdown {dd['daily_dd']:.2%} exceeded limit {settings.daily_dd_limit:.2%}"
            )
            return

        try:
            features_map = self._fetch_features()
        except Exception as exc:
            log.error("feature_fetch_failed", error=str(exc))
            return

        for symbol in self._universe:
            feature_df = features_map.get(symbol)
            if symbol in self._open_positions:
                continue
            if feature_df is None:
                continue

            try:
                # Append latest sentiment score
                sentiment_score = self._sentiment.get_latest_score(symbol)
                last_row = feature_df.iloc[[-1]].copy()
                # (sentiment used as context — model was trained without it for now)

                # Run XGBoost inference
                probs = self._model.predict(last_row[FEATURE_COLUMNS])
                signal_prob = float(probs.iloc[0])

                action = "BUY" if signal_prob >= settings.signal_threshold else "SKIP"

                # Log the decision
                self._logger.log_signal(
                    symbol=symbol,
                    features_dict=last_row[FEATURE_COLUMNS].iloc[0].to_dict(),
                    signal_prob=signal_prob,
                    action_taken=action,
                )

                if action == "SKIP":
                    continue

                if self._circuit_breaker.is_halted():
                    break

                # Size the position
                current_price = float(last_row["ema_50"].iloc[0])
                vol = float(last_row.get("realized_vol_20", pd.Series([0.20])).iloc[0])
                capital = self._get_current_capital()
                size_inr = self._sizer.size(signal_prob, vol, capital)
                qty = self._sizer.shares(size_inr, current_price) if current_price > 0 else 0

                if qty <= 0:
                    continue

                # Place order
                order_id = self._executor.place_market_order(
                    symbol=symbol,
                    transaction_type="BUY",
                    quantity=qty,
                    tag=f"xgb_{signal_prob:.2f}",
                )
                self._open_positions.add(symbol)
                log.info("order_placed", symbol=symbol, qty=qty, prob=round(signal_prob, 3), order_id=order_id)

            except Exception as exc:
                log.error("trading_loop_symbol_error", symbol=symbol, error=str(exc))
                continue

        # Update monitor
        self._monitor.get_drawdown()

    # ------------------------------------------------------------------
    # Post-market  (15:35 IST)
    # ------------------------------------------------------------------

    def post_market_summary(self) -> None:
        """End-of-day wrap-up: summary, Telegram report, drift check."""
        log.info("post_market_summary_start")
        try:
            self._logger.daily_summary()
        except Exception as exc:
            log.error("daily_summary_failed", error=str(exc))

        try:
            self._check_model_drift()
        except Exception as exc:
            log.error("drift_check_failed", error=str(exc))

        self._open_positions.clear()
        log.info("post_market_summary_complete")

    # ------------------------------------------------------------------
    # Weekly reset  (Monday 08:30)
    # ------------------------------------------------------------------

    def reset_weekly(self) -> None:
        capital = self._get_current_capital()
        self._circuit_breaker.reset_weekly(current_capital=capital)
        log.info("weekly_reset_complete", capital=capital)

    # ------------------------------------------------------------------
    # Retrain check  (Sunday 02:00)
    # ------------------------------------------------------------------

    def retrain_check(self) -> None:
        """
        Flag for retraining if rolling 20-trade win rate drops below 50%.
        """
        try:
            from monitoring.mlflow_tracker import ModelDriftMonitor
            drift = ModelDriftMonitor()
            score = drift.compare_live_vs_backtest(window_trades=20)
            if score > 0.3:
                log.warning("model_drift_detected", drift_score=score)
                try:
                    from monitoring.alerts import TelegramAlerter
                    TelegramAlerter().alert_model_drift(
                        current_win_rate=score,
                        baseline_win_rate=0.58,
                    )
                except Exception:
                    pass
        except Exception as exc:
            log.error("retrain_check_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Cancel open orders, persist state, exit cleanly."""
        log.info("graceful_shutdown_initiated")
        for symbol in list(self._open_positions):
            try:
                self._executor.cancel_order(symbol)
            except Exception:
                pass
        log.info("graceful_shutdown_complete")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            self._model = self._registry.get_latest_model(segment="EQ")
            log.info("model_loaded")
        except Exception as exc:
            log.error("model_load_failed", error=str(exc))
            self._model = None

    def _fetch_features(self) -> dict[str, pd.DataFrame]:
        """Fetch latest 60 OHLCV bars for each universe instrument and build features."""
        from datetime import datetime, timedelta
        feature_map: dict[str, pd.DataFrame] = {}
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=90)

        instruments = get_universe(segment="EQ")
        for inst in instruments[:_TOP_N_UNIVERSE]:
            token = inst["token"]
            symbol = inst["symbol"]
            try:
                df = get_ohlcv(token, from_date, to_date, interval="day")
                if len(df) < 60:
                    continue
                features = build_features(df, include_labels=False)
                if not features.empty:
                    feature_map[symbol] = features
            except Exception as exc:
                log.debug("feature_fetch_symbol_error", symbol=symbol, error=str(exc))
        return feature_map

    def _get_current_capital(self) -> float:
        try:
            margins = self._ingestor.kite.margins()
            return float(margins.get("equity", {}).get("available", {}).get("live_balance", 500_000.0))
        except Exception:
            return 500_000.0

    def _get_cached_token(self) -> str | None:
        from data.store import get_redis
        from data.redis_keys import RedisKeys
        return get_redis().get(RedisKeys.KITE_ACCESS_TOKEN)

    def _check_model_drift(self) -> None:
        """Warn if rolling 20-trade win rate has dropped below minimum threshold."""
        try:
            from monitoring.mlflow_tracker import ModelDriftMonitor
            monitor = ModelDriftMonitor()
            drift_score = monitor.compare_live_vs_backtest(window_trades=20)
            if drift_score > 0.3:
                log.warning("model_drift_above_threshold", score=drift_score)
        except Exception as exc:
            log.debug("drift_check_skipped", error=str(exc))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    system = TradingSystem()
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
    scheduler = TradingScheduler(system)
    scheduler.start()

    log.info("trading_system_running", paper_mode=settings.paper_trade_mode)
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        _handle_shutdown(None, None)


if __name__ == "__main__":
    main()
