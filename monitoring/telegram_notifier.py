"""
Phase 6: Telegram notifier with templates and rate limiting.

Sends formatted alerts to Telegram with templates for pre-market scans,
signals, trades, risk alerts, and daily/weekly summaries.

Rate limiting: max 1 alert per symbol per 5 minutes, batch updates every 10 minutes.
"""

from __future__ import annotations

import time
from datetime import datetime
from zoneinfo import ZoneInfo

import structlog

from signals.contracts import Signal

log = structlog.get_logger(__name__)

_TZ_UTC = ZoneInfo("UTC")


class RateLimiter:
    """Rate limiter for Telegram alerts — max 1 per symbol per 5 minutes."""

    def __init__(self, window_seconds: int = 300):
        """
        Initialize rate limiter.

        Parameters
        ----------
        window_seconds : int
            Time window in seconds (default 300 = 5 minutes)
        """
        self._window = window_seconds
        self._last_alert_time: dict[str, float] = {}

    def should_send(self, key: str) -> bool:
        """
        Check if alert for key should be sent.

        Parameters
        ----------
        key : str
            Alert key (e.g., symbol or "system")

        Returns
        -------
        bool
            True if enough time has passed since last alert for this key
        """
        now = time.time()
        last = self._last_alert_time.get(key, 0)
        if now - last >= self._window:
            self._last_alert_time[key] = now
            return True
        return False

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit for a key or all keys."""
        if key is None:
            self._last_alert_time.clear()
        else:
            self._last_alert_time.pop(key, None)


class TelegramNotifier:
    """
    Telegram notifier with templates and rate limiting.

    Uses the existing TelegramAlerter for actual message sending.
    """

    def __init__(self, rate_limit_window_sec: int = 300, batch_window_sec: int = 600):
        """
        Initialize notifier.

        Parameters
        ----------
        rate_limit_window_sec : int
            Rate limit window per symbol in seconds (default 300)
        batch_window_sec : int
            Batch update window in seconds (default 600 = 10 minutes)
        """
        self._rate_limiter = RateLimiter(window_seconds=rate_limit_window_sec)
        self._batch_window = batch_window_sec
        self._pending_batch: list[str] = []
        self._last_batch_send = 0.0

    # ------------------------------------------------------------------
    # Pre-market alerts
    # ------------------------------------------------------------------

    def alert_pre_market_scan(
        self,
        strategies_enabled: list[str],
        universe_size: int,
        expected_runtime_min: int,
    ) -> None:
        """
        Send pre-market scan summary alert.

        Parameters
        ----------
        strategies_enabled : list[str]
            List of enabled strategies
        universe_size : int
            Number of symbols to scan
        expected_runtime_min : int
            Expected scan runtime in minutes
        """
        strategies_str = ", ".join(strategies_enabled) if strategies_enabled else "none"
        msg = (
            "🌅 <b>PRE-MARKET SCAN</b>\n"
            f"Strategies: {strategies_str}\n"
            f"Universe: {universe_size} symbols\n"
            f"Estimated runtime: {expected_runtime_min} min"
        )
        self._send_with_rate_limit("premarket", msg)

    # ------------------------------------------------------------------
    # Signal alerts
    # ------------------------------------------------------------------

    def alert_signal(self, signal: Signal) -> None:
        """
        Send signal alert.

        Parameters
        ----------
        signal : Signal
            Signal object
        """
        if not self._rate_limiter.should_send(f"signal_{signal.symbol}"):
            log.debug("signal_alert_rate_limited", symbol=signal.symbol)
            return

        emoji = "🟢" if signal.direction.value == "long" else "🔴"
        msg = (
            f"{emoji} <b>SIGNAL: {signal.symbol}</b>\n"
            f"Strategy: {signal.strategy_name}\n"
            f"Direction: {signal.direction.value.upper()}\n"
            f"Confidence: {signal.confidence:.0%}\n"
            f"Score: {signal.score:.2f}"
        )

        if signal.entry:
            msg += (
                f"\nEntry: ₹{signal.entry.entry_price:.2f}\n"
                f"Stop: ₹{signal.entry.stop_price:.2f}\n"
                f"Target: ₹{signal.entry.target_price:.2f}"
            )

        if signal.risk:
            msg += f"\nSize hint: {signal.risk.size_hint_pct:.2%}"

        msg += f"\nTime: {signal.timestamp.strftime('%H:%M UTC')}"

        self._send_with_rate_limit(f"signal_{signal.symbol}", msg)

    # ------------------------------------------------------------------
    # Trade alerts
    # ------------------------------------------------------------------

    def alert_trade_entry(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        risk_reward: float,
    ) -> None:
        """
        Send trade entry alert.

        Parameters
        ----------
        symbol : str
            Trading symbol
        direction : str
            "long" or "short"
        quantity : int
            Position size
        entry_price : float
            Entry price
        stop_price : float
            Stop-loss price
        target_price : float
            Profit target price
        risk_reward : float
            Risk/reward ratio
        """
        emoji = "🟢" if direction == "long" else "🔴"
        risk_pct = abs((entry_price - stop_price) / entry_price)
        reward_pct = abs((target_price - entry_price) / entry_price)

        msg = (
            f"{emoji} <b>TRADE ENTRY</b>\n"
            f"Symbol: {symbol}\n"
            f"Direction: {direction.upper()}\n"
            f"Quantity: {quantity}\n"
            f"Entry: ₹{entry_price:.2f}\n"
            f"Stop: ₹{stop_price:.2f}\n"
            f"Target: ₹{target_price:.2f}\n"
            f"Risk/Reward: {risk_reward:.2f}x\n"
            f"Risk %: {risk_pct:.2%} | Reward %: {reward_pct:.2%}\n"
            f"Time: {datetime.now(tz=_TZ_UTC).strftime('%H:%M UTC')}"
        )
        self._send_with_rate_limit(f"entry_{symbol}", msg)

    def alert_trade_exit(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        reason: str = "target",
    ) -> None:
        """
        Send trade exit alert.

        Parameters
        ----------
        symbol : str
            Trading symbol
        quantity : int
            Position size
        entry_price : float
            Entry price
        exit_price : float
            Exit price
        pnl : float
            Absolute P&L
        pnl_pct : float
            Percentage P&L
        reason : str
            Exit reason (target, stoploss, breakeven, time, etc.)
        """
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} <b>TRADE EXIT</b>\n"
            f"Symbol: {symbol}\n"
            f"Quantity: {quantity}\n"
            f"Entry: ₹{entry_price:.2f}\n"
            f"Exit: ₹{exit_price:.2f}\n"
            f"P&L: ₹{pnl:+.0f} ({pnl_pct:+.2%})\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now(tz=_TZ_UTC).strftime('%H:%M UTC')}"
        )
        self._send_with_rate_limit(f"exit_{symbol}", msg)

    # ------------------------------------------------------------------
    # Risk alerts
    # ------------------------------------------------------------------

    def alert_sector_concentration(
        self,
        sector: str,
        current_pct: float,
        limit_pct: float,
    ) -> None:
        """
        Alert when sector exposure exceeds limit.

        Parameters
        ----------
        sector : str
            Sector name
        current_pct : float
            Current sector exposure %
        limit_pct : float
            Sector limit %
        """
        msg = (
            f"⚠️ <b>SECTOR CONCENTRATION</b>\n"
            f"Sector: {sector}\n"
            f"Current: {current_pct:.1%}\n"
            f"Limit: {limit_pct:.1%}\n"
            f"Exposure exceeds threshold!"
        )
        self._send_with_rate_limit(f"sector_{sector}", msg)

    def alert_correlation_warning(self, symbols: list[str], correlation: float) -> None:
        """
        Alert when portfolio correlation exceeds threshold.

        Parameters
        ----------
        symbols : list[str]
            Related symbols
        correlation : float
            Correlation value
        """
        symbols_str = ", ".join(symbols)
        msg = (
            f"⚠️ <b>HIGH CORRELATION WARNING</b>\n"
            f"Symbols: {symbols_str}\n"
            f"Correlation: {correlation:.2f}\n"
            "Portfolio risk may be concentrated."
        )
        self._send_with_rate_limit("correlation", msg)

    def alert_liquidity_warning(self, symbol: str, bid_ask_spread: float) -> None:
        """
        Alert when liquidity is poor.

        Parameters
        ----------
        symbol : str
            Trading symbol
        bid_ask_spread : float
            Bid-ask spread %
        """
        msg = (
            f"⚠️ <b>LIQUIDITY WARNING</b>\n"
            f"Symbol: {symbol}\n"
            f"Bid-ask spread: {bid_ask_spread:.2%}\n"
            "Execution may be slipped."
        )
        self._send_with_rate_limit(f"liquidity_{symbol}", msg)

    # ------------------------------------------------------------------
    # Summary alerts
    # ------------------------------------------------------------------

    def alert_daily_summary(
        self,
        date: str,
        scans: int,
        signals: int,
        trades: int,
        pnl: float,
        pnl_pct: float,
        win_rate: float,
        daily_dd: float,
    ) -> None:
        """
        Send daily summary alert.

        Parameters
        ----------
        date : str
            Date string (YYYY-MM-DD)
        scans : int
            Number of scans
        signals : int
            Number of signals generated
        trades : int
            Number of trades
        pnl : float
            Daily P&L
        pnl_pct : float
            Daily P&L %
        win_rate : float
            Win rate
        daily_dd : float
            Daily drawdown
        """
        emoji = "🟢" if pnl >= 0 else "🔴"
        msg = (
            f"{emoji} <b>DAILY SUMMARY</b> — {date}\n"
            f"Scans: {scans} | Signals: {signals} | Trades: {trades}\n"
            f"P&L: {pnl:+.0f} ({pnl_pct:+.2%})\n"
            f"Win rate: {win_rate:.1%}\n"
            f"Daily DD: {daily_dd:.2%}"
        )
        self._send(msg)

    def alert_weekly_summary(
        self,
        week_start: str,
        week_end: str,
        pnl: float,
        pnl_pct: float,
        win_rate: float,
        sharpe: float,
        max_dd: float,
    ) -> None:
        """
        Send weekly summary alert.

        Parameters
        ----------
        week_start : str
            Week start date
        week_end : str
            Week end date
        pnl : float
            Weekly P&L
        pnl_pct : float
            Weekly P&L %
        win_rate : float
            Win rate
        sharpe : float
            Sharpe ratio
        max_dd : float
            Maximum drawdown
        """
        emoji = "📈" if pnl >= 0 else "📉"
        msg = (
            f"{emoji} <b>WEEKLY SUMMARY</b> — {week_start} to {week_end}\n"
            f"P&L: {pnl:+.0f} ({pnl_pct:+.2%})\n"
            f"Win rate: {win_rate:.1%}\n"
            f"Sharpe: {sharpe:.2f}\n"
            f"Max DD: {max_dd:.2%}"
        )
        self._send(msg)

    # ------------------------------------------------------------------
    # Error alerts
    # ------------------------------------------------------------------

    def alert_error(self, module: str, error_msg: str, is_critical: bool = False) -> None:
        """
        Send error alert.

        Parameters
        ----------
        module : str
            Module where error occurred
        error_msg : str
            Error message
        is_critical : bool
            Whether error is critical
        """
        emoji = "🔴" if is_critical else "⚠️"
        msg = f"{emoji} <b>ERROR</b>\nModule: {module}\nMessage: {error_msg[:200]}"

        # Critical errors bypass rate limit
        if is_critical:
            self._send(msg)
        else:
            self._send_with_rate_limit(f"error_{module}", msg)

    def send_message(self, message: str) -> None:
        """
        Send generic message to Telegram.

        Parameters
        ----------
        message : str
            Message text to send.
        """
        self._send(message)

    # ------------------------------------------------------------------
    # Batch updates
    # ------------------------------------------------------------------

    def add_to_batch(self, message: str) -> None:
        """
        Add message to batch queue for later sending.

        Parameters
        ----------
        message : str
            Message to batch
        """
        self._pending_batch.append(message)

    def flush_batch(self, force: bool = False) -> None:
        """
        Send batched messages if batch window elapsed or forced.

        Parameters
        ----------
        force : bool
            Force send immediately
        """
        now = time.time()
        if not force and (now - self._last_batch_send) < self._batch_window:
            return

        if self._pending_batch:
            msg = "\n\n".join(self._pending_batch)
            self._send(msg)
            self._pending_batch.clear()
            self._last_batch_send = now

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _send_with_rate_limit(self, key: str, message: str) -> None:
        """Send message if rate limit allows."""
        if self._rate_limiter.should_send(key):
            self._send(message)

    def _send(self, message: str) -> None:
        """Send message via TelegramAlerter."""
        try:
            from monitoring.alerts import TelegramAlerter

            TelegramAlerter().send(message)
        except Exception as exc:
            log.warning("telegram_send_failed", error=str(exc))
