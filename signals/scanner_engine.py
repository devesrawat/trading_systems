"""
ScannerEngine — runs any number of strategies against 500+ symbols in parallel.

Design
------
- One DB fetch per unique (interval, lookback_days) pair across all strategies.
  Three strategies that all use daily bars share a single SQL query.
- All strategy × symbol tasks are submitted to one ProcessPoolExecutor,
  so CPU cores are fully utilised without strategy-level serialisation.
- Strategies are re-imported by fully-qualified name inside each worker —
  safe for both fork (Linux) and spawn (macOS / Windows) process starts.
- Results are streamed back as futures complete; nothing is held in memory
  until the full run is done.

Usage
-----
    from signals.scanner_engine import ScannerEngine
    from signals.strategies.vcp import VCPStrategy
    from signals.strategies.rs_breakout import RSBreakoutStrategy
    from signals.strategies.tight_closes import TightClosesStrategy

    engine = ScannerEngine([VCPStrategy(), RSBreakoutStrategy(), TightClosesStrategy()])
    results = engine.run(symbols)   # {"vcp": [...], "rs_breakout": [...], ...}
"""
from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from datetime import date, timedelta
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import text

from data.store import get_engine
from signals.base_strategy import BaseStrategy

log = structlog.get_logger(__name__)

_WORKERS = min(os.cpu_count() or 4, 8)


# ---------------------------------------------------------------------------
# Worker entry point — runs inside each subprocess
# ---------------------------------------------------------------------------

def _worker(
    strategy_fqname: str,
    symbol: str,
    records: list[dict],
) -> tuple[str, dict[str, Any] | None]:
    """
    Instantiate strategy by name, clean data, run scan.

    Returns (strategy_name, result_or_None).
    Running inside a worker process — no shared state with parent.
    """
    import pandas as pd
    from signals.base_strategy import BaseStrategy

    strategy = BaseStrategy.from_fqname(strategy_fqname)
    df = pd.DataFrame(records)
    if df.empty:
        return strategy.name, None

    clean = strategy.prepare(df)
    if clean is None:
        return strategy.name, None

    result = strategy.scan(symbol, clean)
    return strategy.name, result


# ---------------------------------------------------------------------------
# Data fetch — one query per unique (interval, lookback_days)
# ---------------------------------------------------------------------------

def _fetch_group(
    symbols: list[str],
    interval: str,
    lookback_days: int,
) -> pd.DataFrame:
    """Single SQL query for all symbols at a given interval + lookback."""
    from_date = date.today() - timedelta(days=lookback_days)
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text("""
                SELECT symbol, time, open, high, low, close, volume
                FROM ohlcv
                WHERE symbol   = ANY(:symbols)
                  AND interval = :interval
                  AND time    >= :from_date
                ORDER BY symbol, time ASC
            """),
            conn,
            params={"symbols": symbols, "interval": interval, "from_date": from_date},
            parse_dates=["time"],
        )
    log.info(
        "ohlcv_group_fetched",
        interval=interval,
        lookback_days=lookback_days,
        symbols=len(symbols),
        rows=len(df),
    )
    return df


# ---------------------------------------------------------------------------
# ScannerEngine
# ---------------------------------------------------------------------------

class ScannerEngine:
    """
    Runs multiple strategies against a universe of symbols in one pass.

    Parameters
    ----------
    strategies : list of instantiated BaseStrategy subclasses
    workers    : number of worker processes (default = CPU count, capped at 8)
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        workers: int = _WORKERS,
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required.")
        self._strategies = strategies
        self._workers    = workers

    def run(self, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Scan *symbols* with every registered strategy.

        Returns
        -------
        dict keyed by strategy.name, each value a list of passing result dicts,
        sorted by the strategy's own sort key (default: symbol alphabetically).

        Example output::

            {
                "vcp":         [{"symbol": "DIXON", "pivot_buy": 17240.0, ...}],
                "rs_breakout": [{"symbol": "IREDA", "rs_score": 94.2,    ...}],
                "tight_closes":[{"symbol": "KPIL",  "tight_range_pct": 0.4,...}],
            }
        """
        # ----------------------------------------------------------------
        # 1. Group strategies by interval to minimise DB round-trips.
        #    All daily strategies share one SQL query using the maximum
        #    lookback_days across the group.
        # ----------------------------------------------------------------
        groups: dict[str, list[BaseStrategy]] = defaultdict(list)
        for s in self._strategies:
            groups[s.interval].append(s)

        # ----------------------------------------------------------------
        # 2. Fetch data for each interval group in parallel (I/O bound).
        #    symbol_data[interval] = {symbol: [records]}
        # ----------------------------------------------------------------
        symbol_data: dict[str, dict[str, list[dict]]] = {}

        def _fetch_and_index(interval: str, strats: list[BaseStrategy]) -> tuple[str, dict[str, list[dict]]]:
            lookback = max(s.lookback_days for s in strats)
            df = _fetch_group(symbols, interval, lookback)
            if df.empty:
                return interval, {}
            return interval, {
                sym: grp.to_dict(orient="records")
                for sym, grp in df.groupby("symbol", sort=False)
            }

        with ThreadPoolExecutor(max_workers=len(groups) or 1) as io_pool:
            fetch_futures = {
                io_pool.submit(_fetch_and_index, interval, strats): interval
                for interval, strats in groups.items()
            }
            for f in as_completed(fetch_futures):
                interval, sym_map = f.result()
                symbol_data[interval] = sym_map

        # ----------------------------------------------------------------
        # 3. Submit all (strategy × symbol) tasks to the process pool.
        #    All strategies and all symbols run concurrently — no
        #    strategy waits for another to finish.
        # ----------------------------------------------------------------
        results: dict[str, list[dict]] = defaultdict(list)
        total_tasks = sum(
            len(sym_map)
            for interval, sym_map in symbol_data.items()
            for _ in groups[interval]
        )

        log.info(
            "scan_start",
            strategies=[s.name for s in self._strategies],
            symbols=len(symbols),
            total_tasks=total_tasks,
            workers=self._workers,
        )

        with ProcessPoolExecutor(max_workers=self._workers) as pool:
            futures = {}
            for interval, strategies in groups.items():
                sym_map = symbol_data.get(interval, {})
                for strategy in strategies:
                    fqname = strategy.fqname()
                    for sym, records in sym_map.items():
                        f = pool.submit(_worker, fqname, sym, records)
                        futures[f] = strategy.name

            done = 0
            for future in as_completed(futures):
                done += 1
                strategy_name, result = future.result()
                if result is not None:
                    results[strategy_name].append(result)
                if done % 200 == 0:
                    log.info(
                        "scan_progress",
                        done=done,
                        total=total_tasks,
                        found={k: len(v) for k, v in results.items()},
                    )

        # ----------------------------------------------------------------
        # 4. Sort each strategy's results by its preferred sort key
        # ----------------------------------------------------------------
        strategy_map = {s.name: s for s in self._strategies}
        final: dict[str, list[dict]] = {}
        for s in self._strategies:
            hits = results.get(s.name, [])
            hits.sort(key=lambda r: s.sort_key(r))
            final[s.name] = hits

        log.info(
            "scan_complete",
            scanned=len(symbols),
            results={k: len(v) for k, v in final.items()},
        )
        return final
