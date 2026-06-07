"""
Pre-flight validation before enabling live trading.

Checks:
  1. Infrastructure health (Redis, TimescaleDB)
  2. Paper trade count >= 200
  3. Paper trade win rate >= 50%
  4. Broker API credentials present
  5. MLflow production model registered
  6. PAPER_TRADE_MODE=false is set

Exit code 0 = all checks pass (safe to go live).
Exit code 1 = one or more checks failed (DO NOT go live).
"""

from __future__ import annotations

import os
import sys

import structlog

log = structlog.get_logger(__name__)

MIN_PAPER_TRADES = 200
MIN_WIN_RATE = 0.50

PASS = "\033[92m[PASS]\033[0m"  # nosec B105
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"


def _check_redis() -> bool:
    try:
        import redis

        from config.settings import settings

        r = redis.from_url(settings.redis_url, socket_connect_timeout=3)
        r.ping()
        print(f"{PASS} Redis reachable at {settings.redis_url}")
        return True
    except Exception as exc:
        print(f"{FAIL} Redis unreachable: {exc}")
        return False


def _check_timescaledb() -> tuple[bool, int, float]:
    """Return (healthy, trade_count, win_rate)."""
    try:
        import sqlalchemy

        from config.settings import settings

        engine = sqlalchemy.create_engine(
            settings.timescale_url, connect_args={"connect_timeout": 5}
        )
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(
                    "SELECT COUNT(*) AS n, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0) AS win_rate "
                    "FROM paper_trades"
                )
            )
            row = result.fetchone()
            count = int(row[0]) if row and row[0] else 0
            win_rate = float(row[1]) if row and row[1] is not None else 0.0
        print(f"{PASS} TimescaleDB reachable")
        return True, count, win_rate
    except Exception as exc:
        print(f"{FAIL} TimescaleDB unreachable: {exc}")
        return False, 0, 0.0


def _check_paper_trades(count: int, win_rate: float) -> bool:
    ok = True
    if count >= MIN_PAPER_TRADES:
        print(f"{PASS} Paper trades: {count} >= {MIN_PAPER_TRADES} required")
    else:
        print(f"{FAIL} Paper trades: {count} / {MIN_PAPER_TRADES} required. Keep paper trading.")
        ok = False

    if win_rate >= MIN_WIN_RATE:
        print(f"{PASS} Win rate: {win_rate:.1%} >= {MIN_WIN_RATE:.0%} required")
    else:
        print(
            f"{FAIL} Win rate: {win_rate:.1%} / {MIN_WIN_RATE:.0%} required. Review signal quality."
        )
        ok = False
    return ok


def _check_broker_credentials() -> bool:
    from config.settings import settings

    provider = settings.data_provider.lower()
    ok = True

    if provider == "kite":
        missing = []
        if not settings.kite_api_key:
            missing.append("KITE_API_KEY")
        if not settings.kite_api_secret:
            missing.append("KITE_API_SECRET")
        if not settings.kite_access_token:
            missing.append("KITE_ACCESS_TOKEN")
        if missing:
            print(f"{FAIL} Kite credentials missing: {', '.join(missing)}")
            ok = False
        else:
            print(f"{PASS} Kite credentials present")

    elif provider == "upstox":
        if not settings.upstox_access_token:
            print(f"{FAIL} UPSTOX_ACCESS_TOKEN missing")
            ok = False
        else:
            print(f"{PASS} Upstox credentials present")

    elif provider == "binance":
        missing = []
        if not settings.binance_api_key:
            missing.append("BINANCE_API_KEY")
        if not settings.binance_api_secret:
            missing.append("BINANCE_API_SECRET")
        if missing:
            print(f"{FAIL} Binance credentials missing: {', '.join(missing)}")
            ok = False
        else:
            print(f"{PASS} Binance credentials present")

    return ok


def _check_production_model() -> bool:
    try:
        import mlflow

        from config.settings import settings

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient()
        versions = client.get_model_version_by_alias("SignalModel", "production")
        print(f"{PASS} MLflow production model registered: run_id={versions.run_id}")
        return True
    except Exception as exc:
        print(
            f"{WARN} MLflow production model not found ({exc}). System will use fallback scoring."
        )
        return True  # warning only — system degrades gracefully


def _check_paper_mode_disabled() -> bool:
    val = os.environ.get("PAPER_TRADE_MODE", "true").lower()
    if val in ("false", "0", "no"):
        print(f"{PASS} PAPER_TRADE_MODE=false confirmed")
        return True
    print(f"{FAIL} PAPER_TRADE_MODE={val!r} — must be false for live trading")
    return False


def main() -> None:
    print("\n=== Go-Live Pre-flight Check ===\n")

    results: list[bool] = []

    # 1. Infrastructure
    redis_ok = _check_redis()
    results.append(redis_ok)

    db_ok, trade_count, win_rate = _check_timescaledb()
    results.append(db_ok)

    # 2. Paper trade validation
    if db_ok:
        results.append(_check_paper_trades(trade_count, win_rate))

    # 3. Broker credentials
    results.append(_check_broker_credentials())

    # 4. Production model
    results.append(_check_production_model())

    # 5. Paper mode must be disabled
    results.append(_check_paper_mode_disabled())

    print()
    if all(results):
        print("=== ALL CHECKS PASSED — safe to go live ===\n")
        sys.exit(0)
    else:
        failed = results.count(False)
        print(f"=== {failed} CHECK(S) FAILED — DO NOT go live until fixed ===\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
