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
from data.store import get_ohlcv, get_universe
from execution.broker import get_broker_adapter
from execution.logger import TradeLogger
from execution.orders import OrderExecutor
from risk.breakers import CircuitBreaker
from risk.monitor import PortfolioMonitor
from risk.sizer import PositionSizer
from signals.features import FEATURE_COLUMNS, build_features
from signals.model import ModelRegistry, SignalModel

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
        self._open_positions: set[str] = set()

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

        # 3. Run sentiment for equity universe
        if self._sentiment and self._equity_universe:
            try:
                self._sentiment.run_daily(self._equity_universe)
            except Exception as exc:
                log.error("sentiment_pre_market_failed", error=str(exc))

        # 4. Load latest ML model from MLflow
        self._load_model()

        # 5. Reset daily circuit breaker
        capital = self._broker.get_balance() or settings.initial_capital
        self._circuit_breaker.reset_daily(current_capital=capital)

        # 6. Telegram health check
        self._send_alert(
            f"Pre-market OK | market={self._market_type} | "
            f"paper={settings.paper_trade_mode} | "
            f"capital={capital:,.0f} | "
            f"model={'OK' if self._model else 'MISSING'}"
        )
        log.info("pre_market_setup_complete")

    # ------------------------------------------------------------------
    # Trading loop  (equity: every 5 min 09:15–15:25 | crypto: every 5 min 24/7)
    # ------------------------------------------------------------------

    def trading_loop(self) -> None:
        """One scan-signal-execute cycle for the active universe(s)."""
        if self._circuit_breaker.is_halted():
            log.warning("trading_loop_skipped_circuit_halted")
            return

        dd = self._monitor.get_drawdown()
        if dd["daily_dd"] > settings.daily_dd_limit:
            self._circuit_breaker.halt(
                f"daily drawdown {dd['daily_dd']:.2%} exceeded {settings.daily_dd_limit:.2%}"
            )
            return

        if self._market_type in ("equity", "both"):
            self._run_equity_cycle()
        if self._market_type in ("crypto", "both"):
            self._run_crypto_cycle()

        self._monitor.get_drawdown()

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
                self._send_alert(f"Model drift detected — score={score:.3f}. Retrain recommended.")
        except Exception as exc:
            log.error("retrain_check_failed", error=str(exc))

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
        """Placeholder crypto signal — extend with FinBERT + technical features."""
        from data.store import get_latest_crypto_tick
        tick = get_latest_crypto_tick(symbol)
        if tick is None:
            return
        # TODO: build crypto features, run XGBoost, execute
        log.debug("crypto_tick_received", symbol=symbol, price=tick.get("last_price"))

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
            probs = self._model.predict(last_row[FEATURE_COLUMNS])
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

            current_price = float(last_row["ema_50"].iloc[0])
            vol = float(last_row.get("realized_vol_20", pd.Series([0.20])).iloc[0])
            capital = self._broker.get_balance() or settings.initial_capital
            size_inr = self._sizer.size(signal_prob, vol, capital)
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

    def _load_model(self) -> None:
        try:
            self._model = self._registry.get_latest_model(segment="EQ")
            log.info("model_loaded")
        except Exception as exc:
            log.warning("model_load_failed", error=str(exc))
            self._model = None

    def _fetch_equity_features(self) -> dict[str, pd.DataFrame]:
        from datetime import datetime, timedelta
        feature_map: dict[str, pd.DataFrame] = {}
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=90)

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
