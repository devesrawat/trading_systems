"""Add composite index on ohlcv for faster batch queries.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-25
"""

from __future__ import annotations

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old single-column index (optional, but cleaner)
    op.execute("DROP INDEX IF EXISTS ohlcv_token_time_idx")

    # Create new composite index: (token, interval, time DESC)
    # Using BRIN compression for time-series data (compact & fast)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ohlcv_token_interval_time_idx
        ON ohlcv (token, interval, time DESC)
        USING brin
    """)

    # Also add interval-time index for filtered queries
    op.execute("""
        CREATE INDEX IF NOT EXISTS ohlcv_interval_time_idx
        ON ohlcv (interval, time DESC)
        USING brin
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ohlcv_token_interval_time_idx")
    op.execute("DROP INDEX IF EXISTS ohlcv_interval_time_idx")
    # Restore old index
    op.execute("CREATE INDEX IF NOT EXISTS ohlcv_token_time_idx ON ohlcv (token, time DESC)")
