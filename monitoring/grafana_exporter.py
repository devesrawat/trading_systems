"""
Prometheus metrics HTTP server for Grafana dashboards.

Exposes /metrics endpoint on port 8000 (default).
Grafana scrapes this endpoint every 15 seconds.

Metrics exported:
  trading_unrealized_pnl_inr
  trading_realized_pnl_inr
  trading_daily_drawdown_pct
  trading_weekly_drawdown_pct
  trading_max_drawdown_pct
  trading_open_positions
  trading_gross_exposure_inr
  trading_signal_count_total   (counter)
  trading_orders_total         (counter, labels: side, paper)
"""
from __future__ import annotations

import threading

import structlog
from prometheus_client import Counter, Gauge, start_http_server

log = structlog.get_logger(__name__)

# Gauges — already defined in risk/monitor.py (shared via prometheus_client registry)
# Additional counters defined here

_signal_counter = Counter(
    "trading_signal_count_total",
    "Total number of signals generated",
    ["action"],
)

_order_counter = Counter(
    "trading_orders_total",
    "Total number of orders placed",
    ["side", "paper"],
)


class GrafanaExporter:
    """
    Starts a Prometheus HTTP metrics server in a background thread.

    Usage
    -----
    exporter = GrafanaExporter(port=8000)
    exporter.start()
    # … trading runs …
    exporter.record_signal("BUY")
    exporter.record_order("BUY", paper=True)
    """

    def __init__(self, port: int = 8000) -> None:
        self._port = port
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        start_http_server(self._port)
        self._started = True
        log.info("prometheus_exporter_started", port=self._port)

    def record_signal(self, action: str) -> None:
        """Increment signal counter (action: BUY | SKIP)."""
        _signal_counter.labels(action=action).inc()

    def record_order(self, side: str, paper: bool) -> None:
        """Increment order counter."""
        _order_counter.labels(side=side, paper=str(paper)).inc()
