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
    # Drop old single-column index if it exists
    op.execute("DROP INDEX IF EXISTS ohlcv_token_time_idx")

    # Create composite indexes optimized for batch queries
    # Removed DESC ordering for compatibility with PostgreSQL 13+
    # PostgreSQL optimizes reverse scans automatically
    op.execute(
        "CREATE INDEX IF NOT EXISTS ohlcv_token_interval_time_idx ON ohlcv (token, interval, time)"
    )

    op.execute("CREATE INDEX IF NOT EXISTS ohlcv_interval_time_idx ON ohlcv (interval, time)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ohlcv_token_interval_time_idx")
    op.execute("DROP INDEX IF EXISTS ohlcv_interval_time_idx")
    # Restore old index without DESC
    op.execute("CREATE INDEX IF NOT EXISTS ohlcv_token_time_idx ON ohlcv (token, time)")
