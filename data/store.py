"""
TimescaleDB and Redis read/write helpers.

Schema is managed via Alembic migrations (see migrations/).
Run `alembic upgrade head` or call `init_schema()` to apply all pending migrations.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import redis as redis_lib
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from config.settings import settings

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine singleton (shared across the process)
# ---------------------------------------------------------------------------

_engine: Engine | None = None
_redis_client: redis_lib.Redis | None = None

_INSTRUMENTS_PATH = Path(__file__).parent.parent / "config" / "instruments.json"
_ALEMBIC_INI = Path(__file__).parent.parent / "alembic.ini"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.timescale_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
        log.info("db_engine_created", url=settings.timescale_url)
    return _engine


def get_redis() -> redis_lib.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis_lib.from_url(settings.redis_url, decode_responses=True)
        log.info("redis_client_created", url=settings.redis_url)
    return _redis_client


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_schema() -> None:
    """Apply all pending Alembic migrations. Safe to call multiple times."""
    from alembic import command as alembic_command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig(str(_ALEMBIC_INI))
    alembic_command.upgrade(cfg, "head")
    log.info("schema_migrations_applied")


# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------

_OHLCV_COLS = ["time", "token", "symbol", "open", "high", "low", "close", "volume", "interval"]


def get_ohlcv(
    token: int,
    from_date: date | datetime,
    to_date: date | datetime,
    interval: str,
) -> pd.DataFrame:
    """Fetch OHLCV bars from TimescaleDB, sorted ascending by time."""
    engine = get_engine()
    query = text("""
        SELECT time, token, symbol, open, high, low, close, volume, interval
        FROM ohlcv
        WHERE token = :token
          AND interval = :interval
          AND time >= :from_date
          AND time <= :to_date
        ORDER BY time ASC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"token": token, "interval": interval, "from_date": from_date, "to_date": to_date},
            parse_dates=["time"],
            index_col="time",
        )
    log.debug("ohlcv_fetched", token=token, interval=interval, rows=len(df))
    return df


def write_ohlcv(df: pd.DataFrame) -> None:
    """Bulk upsert OHLCV rows — conflict on (time, token, interval)."""
    if df.empty:
        return

    engine = get_engine()
    records = _df_to_records(df)

    upsert_sql = text("""
        INSERT INTO ohlcv (time, token, symbol, open, high, low, close, volume, interval)
        VALUES (:time, :token, :symbol, :open, :high, :low, :close, :volume, :interval)
        ON CONFLICT DO NOTHING
    """)

    with engine.connect() as conn:
        conn.execute(upsert_sql, records)
        conn.commit()
    log.debug("ohlcv_written", rows=len(records))


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame (with or without DatetimeIndex) to list of dicts."""
    reset = df.reset_index() if df.index.name == "time" else df.copy()
    return reset[_OHLCV_COLS].to_dict(orient="records")


# ---------------------------------------------------------------------------
# Redis tick cache
# ---------------------------------------------------------------------------

_TICK_TTL_SECONDS = 5


def get_latest_tick(token: int) -> dict[str, Any] | None:
    """Return the most recent tick for a token from Redis, or None."""
    r = get_redis()
    raw = r.get(f"tick:{token}")
    if raw is None:
        return None
    return json.loads(raw)


def write_tick(token: int, tick: dict[str, Any]) -> None:
    """Cache a live tick in Redis with a short TTL."""
    r = get_redis()
    r.setex(f"tick:{token}", _TICK_TTL_SECONDS, json.dumps(tick))


# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

def get_universe(segment: str = "EQ") -> list[dict[str, Any]]:
    """Load instruments from instruments.json, filtered by segment."""
    with open(_INSTRUMENTS_PATH) as f:
        data = json.load(f)
    instruments: list[dict[str, Any]] = data.get("instruments", [])
    if segment:
        instruments = [i for i in instruments if i.get("segment") == segment]
    return instruments


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--init-schema" in sys.argv:
        init_schema()
        print("Schema initialised successfully.")
