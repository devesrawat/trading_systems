"""timescaledb_optimization

Revision ID: d8a5a1cb2f6d
Revises: 0003
Create Date: 2026-06-07 16:49:47.360231
"""

from __future__ import annotations

from alembic import op

# revision identifiers
revision = "d8a5a1cb2f6d"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Enable compression on ohlcv hypertable
    # Segment by token and interval for optimal compression ratio
    op.execute("""
        ALTER TABLE ohlcv SET (
            timescaledb.compress,
            timescaledb.compress_segmentby = 'token, interval',
            timescaledb.compress_orderby = 'time DESC'
        )
    """)
    # Add compression policy: compress chunks older than 7 days
    op.execute("SELECT add_compression_policy('ohlcv', INTERVAL '7 days')")

    # 2. Convert paper_trades to hypertable to support retention policy
    # Drop PK first because hypertables need time in the PK
    op.execute("ALTER TABLE paper_trades DROP CONSTRAINT IF EXISTS paper_trades_pkey")
    op.execute("SELECT create_hypertable('paper_trades', 'time', if_not_exists => TRUE)")

    # 3. Set data retention policies
    op.execute("SELECT add_retention_policy('ohlcv', INTERVAL '5 years')")
    op.execute("SELECT add_retention_policy('sentiment_scores', INTERVAL '90 days')")
    op.execute("SELECT add_retention_policy('paper_trades', INTERVAL '5 years')")


def downgrade() -> None:
    # Remove policies first
    op.execute("SELECT remove_retention_policy('paper_trades', if_exists => TRUE)")
    op.execute("SELECT remove_retention_policy('sentiment_scores', if_exists => TRUE)")
    op.execute("SELECT remove_retention_policy('ohlcv', if_exists => TRUE)")
    op.execute("SELECT remove_compression_policy('ohlcv', if_exists => TRUE)")
    # Disable compression
    op.execute("ALTER TABLE ohlcv SET (timescaledb.compress = false)")
    # Note: paper_trades remains a hypertable (standard practice in downgrades)
