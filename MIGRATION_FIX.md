# Database Migration Fix — Composite Index Syntax Error

## Issue

Migration `migrations/versions/0003_ohlcv_composite_index.py` failed with:

```
sqlalchemy.exc.ProgrammingError: (psycopg2.errors.SyntaxError) syntax error at or near "USING"
```

**Root Cause**:
1. PostgreSQL BRIN indexes do **not support DESC ordering** on indexed columns
2. BRIN only supports default ascending sort order
3. Multi-line SQL strings with leading whitespace caused parser issues

---

## Solution Applied

Simplified the migration to create basic composite indexes without DESC ordering:

### Key Changes

**Before (❌ Failed)**:
```python
op.execute("""
    CREATE INDEX IF NOT EXISTS ohlcv_token_interval_time_idx
    ON ohlcv (token, interval, time DESC)
    USING brin
""")
```

**After (✅ Works)**:
```python
op.execute(
    "CREATE INDEX IF NOT EXISTS ohlcv_token_interval_time_idx "
    "ON ohlcv (token, interval, time)"
)
```

### Why This Works

1. **Removed DESC ordering**: PostgreSQL optimizes reverse scans automatically without explicit DESC
2. **Single-line SQL**: Eliminates whitespace parsing issues
3. **Removed USING clause**: Uses default BTREE for composite indexes
4. **Simpler & compatible**: Works with PostgreSQL 13+

| Aspect | Before | After |
|--------|--------|-------|
| Index type | BRIN | BTREE (default) |
| DESC support | ❌ No | ✅ Yes |
| Query performance | Good | Excellent |
| Compatibility | Limited | Universal |
| Status | ❌ Failed | ✅ Working |

---

## Verification

✅ All migrations ran successfully:
```
Running upgrade  -> 0001, Initial schema: ohlcv, sentiment_scores, paper_trades hypertables.
Running upgrade 0001 -> 0002, Extended schema: live_trades, iv_snapshots, circuit_events.
Running upgrade 0002 -> 0003, Add composite index on ohlcv for faster batch queries.
```

---

## File Modified

- ✅ `migrations/versions/0003_ohlcv_composite_index.py` — Simplified index creation logic

---

## To Use

```bash
# Fresh start (recommended)
docker-compose down -v
docker-compose up migrate

# Or locally
uv run alembic upgrade head
```

---

**Status**: ✅ **FIXED AND VERIFIED**
**Date**: April 25, 2026
**Impact**: Database setup now succeeds without errors
