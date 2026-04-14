<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

# Project: AI Trading System

## Architecture

- Signal engine: Python 3.13, async, FastAPI for internal APIs
- Data: Redis (cache) + TimescaleDB (OHLCV) + SQLite (signals log)
- Broker API: Zerodha Kite (Indian equities + F&O), Binance WS (crypto)
- ML: LightGBM primary, PatchTST for experimental. Models in ./models/
- Notifications: Telegram bot (token in .env, never hardcode)

## Commands to run before committing

```bash
uv run pytest tests/ -x -q --no-cov
uv run ruff format .
uv run ruff check . --fix
uv run mypy data/ signals/ orchestrator/ execution/ risk/ monitoring/ llm/ backtest/ options/
uv run bandit -c pyproject.toml -r .
uv run pip-audit
```

Pre-commit hooks run ruff + bandit on every commit automatically (after `uv run pre-commit install`).

## Key conventions

- All signals must pass through SignalValidator before output
- Never call broker API without rate limiter wrapper
- Feature names must match schema in ./docs/feature_registry.md
- Backtest results always saved to ./results/ with timestamp

## Do not touch

- ./models/production/ — read-only, never overwrite
- .env — never echo or log

## Gotchas discovered

- Zerodha Kite WebSocket disconnects silently — always wrap with reconnect loop
- NSE bhavcopy URL format changes on expiry days — use the adaptive fetcher
- TimescaleDB hypertable requires explicit time dimension in all queries
