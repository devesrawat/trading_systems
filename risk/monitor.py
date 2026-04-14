"""
Real-time portfolio P&L and drawdown tracker.

State stored in Redis — survives process restarts and is available
to all modules (circuit breaker, Prometheus exporter, Telegram alerts).
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from prometheus_client import Gauge

from data.redis_keys import RedisKeys
from data.store import get_redis

log = structlog.get_logger(__name__)

_REDIS_KEY = RedisKeys.PORTFOLIO_STATE

# Prometheus gauges (module-level singletons)
_G_UNREALIZED = Gauge("trading_unrealized_pnl_inr", "Unrealized P&L in INR")
_G_REALIZED = Gauge("trading_realized_pnl_inr", "Realized P&L in INR")
_G_DAILY_DD = Gauge("trading_daily_drawdown_pct", "Daily drawdown as fraction")
_G_WEEKLY_DD = Gauge("trading_weekly_drawdown_pct", "Weekly drawdown as fraction")
_G_MAX_DD = Gauge("trading_max_drawdown_pct", "Max drawdown from peak as fraction")
_G_POSITIONS = Gauge("trading_open_positions", "Number of open positions")


class PortfolioMonitor:
    """
    Tracks open positions, P&L, drawdown, and exposure in real-time.

    All state is mirrored to Redis so the circuit breaker and monitoring
    stack always have current data.
    """

    def __init__(self, initial_capital: float) -> None:
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        self._daily_start_capital = initial_capital
        self._weekly_start_capital = initial_capital
        self._peak_capital = initial_capital
        self._realized_pnl: float = 0.0
        self._positions: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def update_position(
        self,
        symbol: str,
        qty: int,
        avg_price: float,
        current_price: float,
    ) -> None:
        """Add, update, or remove a position. qty=0 closes the position."""
        if qty == 0:
            self._positions.pop(symbol, None)
        else:
            self._positions[symbol] = {
                "qty": qty,
                "avg_price": avg_price,
                "current_price": current_price,
            }
        self._persist()

    def record_realized(self, symbol: str, pnl: float) -> None:
        """Record a realized P&L event (on position close)."""
        self._realized_pnl += pnl
        self._persist()

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    def get_pnl(self) -> dict[str, float]:
        unrealized = sum(
            pos["qty"] * (pos["current_price"] - pos["avg_price"])
            for pos in self._positions.values()
        )
        total = self._realized_pnl + unrealized
        pct_return = total / self._initial_capital if self._initial_capital > 0 else 0.0

        _G_UNREALIZED.set(unrealized)
        _G_REALIZED.set(self._realized_pnl)

        return {
            "unrealized": round(unrealized, 2),
            "realized": round(self._realized_pnl, 2),
            "total": round(total, 2),
            "pct_return": round(pct_return, 6),
        }

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def get_drawdown(self) -> dict[str, float]:
        capital = self._current_capital + self.get_pnl()["unrealized"]

        daily_dd = (
            (self._daily_start_capital - capital) / self._daily_start_capital
            if self._daily_start_capital > 0
            else 0.0
        )
        weekly_dd = (
            (self._weekly_start_capital - capital) / self._weekly_start_capital
            if self._weekly_start_capital > 0
            else 0.0
        )
        self._peak_capital = max(self._peak_capital, capital)
        max_dd = (
            (self._peak_capital - capital) / self._peak_capital if self._peak_capital > 0 else 0.0
        )
        current_dd = max(0.0, daily_dd)

        _G_DAILY_DD.set(max(0.0, daily_dd))
        _G_WEEKLY_DD.set(max(0.0, weekly_dd))
        _G_MAX_DD.set(max(0.0, max_dd))

        return {
            "daily_dd": round(max(0.0, daily_dd), 6),
            "weekly_dd": round(max(0.0, weekly_dd), 6),
            "max_dd": round(max(0.0, max_dd), 8),
            "current_dd": round(current_dd, 6),
        }

    # ------------------------------------------------------------------
    # Exposure
    # ------------------------------------------------------------------

    def get_exposure(self) -> dict[str, float]:
        if not self._positions:
            return {"gross_exposure": 0.0, "net_exposure": 0.0, "largest_position_pct": 0.0}

        position_values = {
            sym: pos["qty"] * pos["current_price"] for sym, pos in self._positions.items()
        }
        gross = sum(abs(v) for v in position_values.values())
        net = sum(position_values.values())
        largest = max(abs(v) for v in position_values.values()) if position_values else 0.0
        largest_pct = largest / self._initial_capital if self._initial_capital > 0 else 0.0

        _G_POSITIONS.set(len(self._positions))

        return {
            "gross_exposure": round(gross, 2),
            "net_exposure": round(net, 2),
            "largest_position_pct": round(largest_pct, 6),
        }

    # ------------------------------------------------------------------
    # Convenience helpers (used by TelegramSignalBot and RiskGateway)
    # ------------------------------------------------------------------

    def get_current_heat(self) -> float:
        """Gross exposure as a fraction of initial capital (0.0–1.0+)."""
        gross = self.get_exposure()["gross_exposure"]
        return gross / self._initial_capital if self._initial_capital > 0 else 0.0

    def get_daily_drawdown(self) -> float:
        """Daily drawdown fraction — convenience wrapper around get_drawdown()."""
        return self.get_drawdown()["daily_dd"]

    def get_open_positions(self) -> set[str]:
        """Return the set of symbols with open positions."""
        return set(self._positions.keys())

    def get_open_positions_detail(self) -> dict[str, dict]:
        """
        Return per-symbol detail for the /portfolio Telegram command.

        Returns
        -------
        dict[symbol, {qty, avg_price, current_price, unrealized_pnl_pct}]
        """
        result: dict[str, dict] = {}
        for sym, pos in self._positions.items():
            cost = pos["avg_price"] * pos["qty"]
            pnl_pct = (
                (pos["current_price"] - pos["avg_price"]) / pos["avg_price"]
                if pos["avg_price"] > 0
                else 0.0
            )
            result[sym] = {
                "qty": pos["qty"],
                "avg_price": pos["avg_price"],
                "current_price": pos["current_price"],
                "unrealized_pnl_pct": round(pnl_pct, 6),
                "unrealized_pnl_inr": round(cost * pnl_pct, 2),
            }
        return result

    # ------------------------------------------------------------------
    # Prometheus
    # ------------------------------------------------------------------

    def export_prometheus_metrics(self) -> str:
        """Generate Prometheus text-format metrics string."""
        pnl = self.get_pnl()
        dd = self.get_drawdown()
        exp = self.get_exposure()

        lines = [
            "# HELP trading_unrealized_pnl_inr Unrealized P&L in INR",
            f"trading_unrealized_pnl_inr {pnl['unrealized']}",
            "# HELP trading_realized_pnl_inr Realized P&L in INR",
            f"trading_realized_pnl_inr {pnl['realized']}",
            "# HELP trading_daily_drawdown_pct Daily drawdown fraction",
            f"trading_daily_drawdown_pct {dd['daily_dd']}",
            "# HELP trading_weekly_drawdown_pct Weekly drawdown fraction",
            f"trading_weekly_drawdown_pct {dd['weekly_dd']}",
            "# HELP trading_max_drawdown_pct Max drawdown from peak",
            f"trading_max_drawdown_pct {dd['max_dd']}",
            "# HELP trading_open_positions Number of open positions",
            f"trading_open_positions {len(self._positions)}",
            "# HELP trading_gross_exposure_inr Gross exposure in INR",
            f"trading_gross_exposure_inr {exp['gross_exposure']}",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        state = {
            "current_capital": self._current_capital,
            "daily_start_capital": self._daily_start_capital,
            "weekly_start_capital": self._weekly_start_capital,
            "peak_capital": self._peak_capital,
            "realized_pnl": self._realized_pnl,
            "positions": self._positions,
        }
        get_redis().hset(_REDIS_KEY, mapping={"data": json.dumps(state)})

    def _load(self) -> None:
        raw = get_redis().hgetall(_REDIS_KEY)
        if not raw or "data" not in raw:
            return
        try:
            state = json.loads(raw["data"])
            self._current_capital = float(state.get("current_capital", self._initial_capital))
            self._daily_start_capital = float(
                state.get("daily_start_capital", self._initial_capital)
            )
            self._weekly_start_capital = float(
                state.get("weekly_start_capital", self._initial_capital)
            )
            self._peak_capital = float(state.get("peak_capital", self._initial_capital))
            self._realized_pnl = float(state.get("realized_pnl", 0.0))
            self._positions = state.get("positions", {})
        except Exception as exc:
            log.error("portfolio_state_load_failed", error=str(exc))
