"""Initial schema: ohlcv, sentiment_scores, paper_trades hypertables.

Revision ID: 0001
Revises:
Create Date: 2026-03-22
"""

from __future__ import annotations

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # ohlcv — TimescaleDB hypertable for OHLCV bars
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            time        TIMESTAMPTZ      NOT NULL,
            token       INTEGER          NOT NULL,
            symbol      TEXT             NOT NULL,
            open        DOUBLE PRECISION,
            high        DOUBLE PRECISION,
            low         DOUBLE PRECISION,
            close       DOUBLE PRECISION,
            volume      BIGINT,
            interval    TEXT             NOT NULL
        )
    """)
    op.execute("SELECT create_hypertable('ohlcv', 'time', if_not_exists => TRUE)")
    op.execute("CREATE INDEX IF NOT EXISTS ohlcv_token_time_idx ON ohlcv (token, time DESC)")

    # ------------------------------------------------------------------
    # sentiment_scores — TimescaleDB hypertable for FinBERT scores
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_scores (
            time            TIMESTAMPTZ      NOT NULL,
            symbol          TEXT             NOT NULL,
            score           DOUBLE PRECISION,
            headline_count  INT
        )
    """)
    op.execute("SELECT create_hypertable('sentiment_scores', 'time', if_not_exists => TRUE)")

    # ------------------------------------------------------------------
    # paper_trades — audit log (SEBI compliance)
    # ------------------------------------------------------------------
    op.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            id                  SERIAL           PRIMARY KEY,
            time                TIMESTAMPTZ      DEFAULT NOW(),
            symbol              TEXT,
            side                TEXT,
            quantity            INT,
            price               DOUBLE PRECISION,
            signal_prob         DOUBLE PRECISION,
            position_size_inr   DOUBLE PRECISION,
            tag                 TEXT
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS paper_trades")
    op.execute("DROP TABLE IF EXISTS sentiment_scores")
    op.execute("DROP TABLE IF EXISTS ohlcv")
