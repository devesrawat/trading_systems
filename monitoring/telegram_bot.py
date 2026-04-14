"""
Telegram bot for live trading commands and signal notifications.

Commands:
  /status    — system state, daily drawdown, halt reason
  /portfolio — open positions with unrealized P&L
  /pause     — operator soft-pause (does NOT clear risk state)
  /resume    — lift operator pause (only clears operator pause, not risk halts)

To run the bot as a long-lived process alongside the main orchestrator::

    python -m monitoring.telegram_bot

Required env vars: TELEGRAM_BOT_TOKEN, TELEGRAM_CHANNEL_ID
"""
from __future__ import annotations

import os

import structlog
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes

log = structlog.get_logger(__name__)


class TelegramSignalBot:
    """
    Interactive Telegram bot wired to CircuitBreaker and PortfolioMonitor.

    Parameters
    ----------
    circuit_breaker:
        ``risk.breakers.CircuitBreaker`` instance.
    portfolio_monitor:
        ``risk.monitor.PortfolioMonitor`` instance.
    """

    def __init__(self, circuit_breaker, portfolio_monitor) -> None:  # type: ignore[type-arg]
        token = os.environ["TELEGRAM_BOT_TOKEN"]
        self._bot = Bot(token=token)
        self._channel_id = os.environ["TELEGRAM_CHANNEL_ID"]
        self._cb = circuit_breaker
        self._pm = portfolio_monitor

    # ------------------------------------------------------------------
    # Outbound: signal notifications
    # ------------------------------------------------------------------

    async def send_signal_alert(self, symbol: str, prob: float, qty: int, price: float) -> None:
        """Send a signal notification to the configured Telegram channel."""
        text = (
            f"🟢 BUY {symbol}\n"
            f"Conf: {prob:.0%} | Qty: {qty} | Price: ₹{price:,.2f}"
        )
        await self._bot.send_message(chat_id=self._channel_id, text=text)

    async def send_halt_alert(self, reason: str) -> None:
        await self._bot.send_message(
            chat_id=self._channel_id,
            text=f"⛔ TRADING HALTED\nReason: {reason}\nUse /resume to lift operator pause.",
        )

    # ------------------------------------------------------------------
    # Application builder
    # ------------------------------------------------------------------

    def build_application(self) -> Application:
        """Return a configured python-telegram-bot Application with all handlers."""
        app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("portfolio", self._cmd_portfolio))
        app.add_handler(CommandHandler("pause", self._cmd_pause))
        app.add_handler(CommandHandler("resume", self._cmd_resume))
        return app

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        halted = self._cb.is_halted()
        dd = self._pm.get_daily_drawdown()
        heat = self._pm.get_current_heat()
        status = "HALTED ⛔" if halted else "ACTIVE ✅"
        reason = self._cb.halt_reason() or "N/A"
        await update.message.reply_text(  # type: ignore[union-attr]
            f"System: {status}\n"
            f"Daily DD: {dd:.2%}\n"
            f"Portfolio heat: {heat:.2%}\n"
            f"Halt reason: {reason}"
        )

    async def _cmd_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        positions = self._pm.get_open_positions_detail()
        if not positions:
            await update.message.reply_text("No open positions.")  # type: ignore[union-attr]
            return
        lines = ["📊 Open Positions:"]
        for sym, pos in positions.items():
            pnl_pct = pos["unrealized_pnl_pct"]
            pnl_inr = pos["unrealized_pnl_inr"]
            lines.append(
                f"  {sym}: {pos['qty']} shares @ ₹{pos['avg_price']:,.2f}"
                f" | P&L: {pnl_pct:+.2%} (₹{pnl_inr:+,.0f})"
            )
        await update.message.reply_text("\n".join(lines))  # type: ignore[union-attr]

    async def _cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Operator soft-pause — does NOT clear risk state."""
        self._cb.operator_pause(reason="telegram_operator_pause")
        await update.message.reply_text(  # type: ignore[union-attr]
            "⏸ Trading paused by operator.\n"
            "Risk halt state is preserved — use /resume only for this operator pause.\n"
            "A genuine risk halt must be cleared via the admin CLI."
        )
        log.warning("telegram_operator_pause_issued")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Lift operator pause only — cannot clear a risk-triggered halt."""
        if self._cb._halted:  # genuine risk halt
            await update.message.reply_text(  # type: ignore[union-attr]
                "❌ Cannot resume: system is halted by a risk rule.\n"
                f"Halt reason: {self._cb.halt_reason()}\n"
                "Use the admin CLI to clear risk halts after manual review."
            )
            return
        self._cb.operator_resume()
        await update.message.reply_text("▶ Operator pause lifted. Trading resumed.")  # type: ignore[union-attr]
        log.warning("telegram_operator_resume_issued")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the Telegram bot as a standalone polling process."""
    from config.settings import settings
    from risk.breakers import CircuitBreaker
    from risk.monitor import PortfolioMonitor

    cb = CircuitBreaker(
        daily_limit=settings.daily_dd_limit,
        weekly_limit=settings.weekly_dd_limit,
    )
    pm = PortfolioMonitor(initial_capital=settings.initial_capital)
    bot = TelegramSignalBot(circuit_breaker=cb, portfolio_monitor=pm)
    app = bot.build_application()
    log.info("telegram_bot_starting")
    app.run_polling()


if __name__ == "__main__":
    main()
