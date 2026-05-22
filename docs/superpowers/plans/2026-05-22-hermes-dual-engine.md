# Hermes Dual-Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Setup an autonomous Hermes agent system (Momentum Sentinel + Wealth Architect) for Indian Stock Market research and long-term portfolio management.

**Architecture:** A unified Hermes profile (`india-alpha`) using Composio MCP for brokerage (Zerodha/Angel One) and market data integration, with automated Telegram alerting.

**Tech Stack:** Hermes Agent, Composio CLI/MCP, Claude 3.5 Sonnet, Telegram Bot API.

---

### Task 1: Foundation & Profile Setup

**Files:**
- Create: `docs/superpowers/hermes/config.json` (Configuration tracking)
- Modify: `~/.hermes/profiles/india-alpha/settings.json` (via CLI)

- [ ] **Step 1: Create the isolated Hermes profile**

Run: `hermes profile create india-alpha --clone`
Expected: Success message from Hermes CLI.

- [ ] **Step 2: Set the primary model to Claude 3.5 Sonnet**

Run: `hermes config set model claude-3.5-sonnet --profile india-alpha`
Expected: Model updated to Claude 3.5 Sonnet.

- [ ] **Step 3: Verify profile isolation**

Run: `hermes profile list`
Expected: `india-alpha` appears in the list.

- [ ] **Step 4: Commit configuration record**

```bash
git add docs/superpowers/hermes/config.json
git commit -m "chore: initialize hermes india-alpha profile"
```

### Task 2: Composio & Broker Integration

**Files:**
- Modify: `docs/superpowers/hermes/tools.md` (Tool inventory)

- [ ] **Step 1: Install Composio CLI if missing**

Run: `pip install composio-core`
Expected: Composio installed.

- [ ] **Step 2: Add Indian Broker (Zerodha/Angel One)**

Run: `composio add zerodha` (or `angelone`)
Note: This will open a browser for OAuth. Ensure it's set to "Read-Only" or "Paper" if available.

- [ ] **Step 3: Connect Composio MCP to Hermes**

Run: `hermes tool add composio --mcp-url https://connect.composio.dev/mcp --profile india-alpha`
Expected: Composio tools registered in Hermes.

- [ ] **Step 4: Verify connectivity**

Run: `hermes tool list --profile india-alpha`
Expected: `zerodha` or `angelone` actions appear.

### Task 3: Telegram Gateway Setup

**Files:**
- Modify: `docs/superpowers/hermes/gateway.md`

- [ ] **Step 1: Configure Telegram Gateway**

Run: `hermes gateway setup telegram`
Note: Follow prompts to provide Bot Token and Chat ID.

- [ ] **Step 2: Test notification**

Run: `hermes chat "Send a test alert to Telegram: 'India Sentinel Active'" --profile india-alpha`
Expected: Message received in Telegram.

### Task 4: Momentum Sentinel Skill Development

**Files:**
- Create: `docs/superpowers/hermes/skills/momentum-scanner.md` (Definition)

- [ ] **Step 1: Define the Momentum Scanner logic**

Write to `docs/superpowers/hermes/skills/momentum-scanner.md`:
```markdown
# Skill: Momentum Scanner
Description: Scans Nifty 500 for volume breakouts and volatility spikes.
Logic:
1. Fetch Top 500 NSE gainers.
2. Filter for stocks where Volume > 2x 20-day Average.
3. Check RSI (14) > 60.
4. Alert Telegram with Stock Symbol, Price, and Volume % change.
```

- [ ] **Step 2: Teach the skill to Hermes**

Run: `hermes skill create momentum-scanner --file docs/superpowers/hermes/skills/momentum-scanner.md --profile india-alpha`
Expected: Skill learned by Hermes.

- [ ] **Step 3: Run a manual test scan**

Run: `hermes chat "Execute momentum-scanner for today's market" --profile india-alpha`
Expected: Hermes pulls data and sends a summary to Telegram.

### Task 5: Wealth Architect Skill Development

**Files:**
- Create: `docs/superpowers/hermes/skills/wealth-architect.md`

- [ ] **Step 1: Define Fundamental Screening logic**

Write to `docs/superpowers/hermes/skills/wealth-architect.md`:
```markdown
# Skill: Wealth Architect
Description: Identifies undervalued blue-chip stocks for long-term compounding.
Logic:
1. Fetch Nifty 50 constituents.
2. Filter for PE < Sector Average and ROE > 15%.
3. Suggest top 3 stocks for SIP adjustment.
```

- [ ] **Step 2: Teach the skill to Hermes**

Run: `hermes skill create wealth-architect --file docs/superpowers/hermes/skills/wealth-architect.md --profile india-alpha`
Expected: Skill learned by Hermes.

### Task 6: Automation & Scheduling

- [ ] **Step 1: Schedule the Momentum Scanner**

Run: `hermes schedule "Every weekday at 10:00 AM, 1:00 PM, and 3:00 PM run momentum-scanner" --profile india-alpha`
Expected: Job scheduled in Hermes cron.

- [ ] **Step 2: Schedule the Wealth Briefing**

Run: `hermes schedule "Every Saturday at 9:00 AM run wealth-architect and send a weekly summary to Telegram" --profile india-alpha`
Expected: Weekly job scheduled.

- [ ] **Step 3: Final Verification**

Run: `hermes schedule list --profile india-alpha`
Expected: Both jobs active.
