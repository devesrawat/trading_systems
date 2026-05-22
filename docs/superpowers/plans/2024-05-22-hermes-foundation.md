# Hermes Foundation & Profile Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize an isolated Hermes CLI profile for the India Alpha trading engine and record configuration.

**Architecture:** Isolated profile creation using Hermes CLI with local tracking in project documentation for version control.

**Tech Stack:** Hermes CLI, Git

---

### Task 1: Hermes Profile Initialization

**Files:**
- Create: `docs/superpowers/hermes/config.json`
- Modify: `~/.hermes/profiles/india-alpha/settings.json` (managed by CLI)

- [x] **Step 1: Create the isolated Hermes profile**

Run: `hermes profile create india-alpha --clone`
Expected: `Profile 'india-alpha' created successfully.`

- [x] **Step 2: Set the primary model to Claude 3.5 Sonnet**

Run: `hermes config set model claude-3.5-sonnet --profile india-alpha`
Expected: `Model updated to claude-3.5-sonnet for profile 'india-alpha'.`

- [x] **Step 3: Verify profile isolation**

Run: `hermes profile list`
Expected: `india-alpha` appears in the list output.

- [x] **Step 4: Create configuration tracking record**

Create `docs/superpowers/hermes/config.json`:
```json
{
  "profile": "india-alpha",
  "model": "claude-3.5-sonnet",
  "initialized_at": "2024-05-22T00:00:00Z"
}
```

- [ ] **Step 5: Commit configuration record**

```bash
git add docs/superpowers/hermes/config.json
git commit -m "chore: initialize hermes india-alpha profile"
```
