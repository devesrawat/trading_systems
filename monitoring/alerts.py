"""
Telegram alerter for circuit breakers, daily summaries, and system errors.

All methods are synchronous wrappers — internally runs async Telegram Bot
in a temporary event loop so callers don't need to be async.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

try:
    from telegram import Bot
except ImportError:
    Bot = None  # type: ignore[assignment,misc]


class TelegramAlerter:
    """
    Sends formatted messages to a Telegram chat via the Bot API.

    If bot_token or chat_id is None, all methods are silent no-ops.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        if bot_token is None or chat_id is None:
            from config.settings import settings
            bot_token = bot_token or settings.telegram_bot_token
            chat_id = chat_id or settings.telegram_chat_id

        self._token = bot_token
        self._chat_id = chat_id
        self._max_retries = max_retries
        self._bot = None

        if self._token and Bot is not None:
            self._bot = Bot(token=self._token)

    # ------------------------------------------------------------------
    # Core send
    # ------------------------------------------------------------------

    def send(self, message: str) -> None:
        """Send *message* to the configured chat. Silently no-ops if no token."""
        if not self._bot or not self._chat_id:
            return

        for attempt in range(self._max_retries):
            try:
                asyncio.run(
                    self._bot.send_message(
                        chat_id=self._chat_id,
                        text=message,
                        parse_mode="HTML",
                    )
                )
                return
            except Exception as exc:
                log.warning("telegram_send_failed", attempt=attempt + 1, error=str(exc))
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)

    # ------------------------------------------------------------------
    # Formatted alert helpers
    # ------------------------------------------------------------------

    def alert_circuit_breaker(
        self,
        reason: str,
        dd_pct: float,
        capital: float,
    ) -> None:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(tz=ZoneInfo("Asia/Kolkata")).strftime("%H:%M IST")
        msg = (
            "🔴 <b>CIRCUIT BREAKER TRIGGERED</b>\n"
            f"Reason: {reason}\n"
            f"DD: {dd_pct:.2%} | Capital: ₹{capital:,.0f}\n"
            "All trading halted. Manual reset required.\n"
            f"Time: {now}"
        )
        self.send(msg)

    def alert_daily_summary(
        self,
        trades: int,
        pnl_pct: float,
        sharpe: float,
        win_rate: float,
    ) -> None:
        emoji = "🟢" if pnl_pct >= 0 else "🔴"
        msg = (
            f"{emoji} <b>Daily Summary</b>\n"
            f"Trades: {trades} | Win rate: {win_rate:.1%}\n"
            f"P&L: {pnl_pct:+.2%} | Sharpe: {sharpe:.2f}"
        )
        self.send(msg)

    def alert_model_drift(
        self,
        current_win_rate: float,
        baseline_win_rate: float,
    ) -> None:
        msg = (
            "⚠️ <b>MODEL DRIFT WARNING</b>\n"
            f"Current win rate: {current_win_rate:.1%}\n"
            f"Baseline win rate: {baseline_win_rate:.1%}\n"
            "Consider retraining the signal model."
        )
        self.send(msg)

    def alert_system_error(self, module: str, error_msg: str) -> None:
        msg = (
            "🔧 <b>SYSTEM ERROR</b>\n"
            f"Module: <code>{module}</code>\n"
            f"Error: {error_msg[:200]}"
        )
        self.send(msg)
