"""Extended schema: live_trades, iv_snapshots, circuit_events.

live_trades  — SEBI-compliant audit log for real (non-paper) order execution.
               Currently missing: live orders are placed via Kite but never
               written to the database (compliance gap).

iv_snapshots — Persistent store for F&O options IV feature snapshots.
               Currently the F&O module computes IV features but discards them.

circuit_events — DB-level audit trail for circuit breaker halts and resets,
                 supplementing the MLflow log so events are queryable via SQL.

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-22
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # live_trades — real order audit log (SEBI 5-year retention)
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS live_trades (
            id                  SERIAL           PRIMARY KEY,
            time                TIMESTAMPTZ      DEFAULT NOW(),
            symbol              TEXT             NOT NULL,
            side                TEXT             NOT NULL,
            quantity            INT              NOT NULL,
            price               DOUBLE PRECISION NOT NULL,
            order_id            TEXT,
            signal_prob         DOUBLE PRECISION,
            position_size_inr   DOUBLE PRECISION,
            slippage_pct        DOUBLE PRECISION,
            tag                 TEXT,
            strategy_version    TEXT             DEFAULT '1.0'
        )
    """)
    op.execute(
        "CREATE INDEX IF NOT EXISTS live_trades_symbol_time_idx ON live_trades (symbol, time DESC)"
    )

    # ------------------------------------------------------------------
    # iv_snapshots — F&O options IV feature snapshots (hypertable)
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS iv_snapshots (
            time                TIMESTAMPTZ      NOT NULL,
            symbol              TEXT             NOT NULL,
            expiry_date         DATE             NOT NULL,
            iv_rank             DOUBLE PRECISION,
            iv_percentile       DOUBLE PRECISION,
            iv_premium          DOUBLE PRECISION,
            put_call_ratio      DOUBLE PRECISION,
            max_pain            INT,
            days_to_expiry      INT,
            current_iv          DOUBLE PRECISION,
            realized_vol        DOUBLE PRECISION,
            signal_type         TEXT
        )
    """)
    op.execute("SELECT create_hypertable('iv_snapshots', 'time', if_not_exists => TRUE)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS iv_snapshots_symbol_time_idx "
        "ON iv_snapshots (symbol, time DESC)"
    )

    # ------------------------------------------------------------------
    # circuit_events — circuit breaker halt/reset audit trail
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS circuit_events (
            id                  SERIAL           PRIMARY KEY,
            time                TIMESTAMPTZ      DEFAULT NOW(),
            event_type          TEXT             NOT NULL,
            reason              TEXT,
            capital_at_event    DOUBLE PRECISION,
            daily_dd_pct        DOUBLE PRECISION,
            weekly_dd_pct       DOUBLE PRECISION
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS circuit_events")
    op.execute("DROP TABLE IF EXISTS iv_snapshots")
    op.execute("DROP TABLE IF EXISTS live_trades")
