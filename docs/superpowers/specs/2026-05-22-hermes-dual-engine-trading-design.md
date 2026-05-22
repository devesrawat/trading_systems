# Design Specification: Hermes Dual-Engine Trading & Wealth System

**Date:** 2026-05-22
**Status:** Approved
**Topic:** Indian Stock Market Research & Long-Term Portfolio Management

## 1. Executive Summary
The "Hermes Dual-Engine" is an autonomous agentic system designed to achieve aggressive short-term research gains (Momentum Sentinel) while building sustainable long-term wealth (Wealth Architect) in the Indian Stock Market. It leverages the Hermes Agent (Composio) for persistence and the Model Context Protocol (MCP) for deep brokerage and data integration.

## 2. Architecture Overview
The system consists of two primary logic engines running within a unified Hermes profile.

### 2.1 The Momentum Sentinel (Aggressive Engine)
- **Primary Goal:** Identify high-yield (targeting 50% monthly return potential) opportunities.
- **Scope:** Nifty/Bank Nifty Options, Small/Mid-cap breakouts.
- **Workflow:**
    1. **Scan:** Continuous monitoring of Nifty 500 volume and volatility spikes.
    2. **Filter:** Applies "Breakout Momentum" and "Relative Strength" filters.
    3. **Alert:** Sends real-time, actionable trade ideas to Telegram via Hermes Gateway.

### 2.2 The Wealth Architect (Conservative Engine)
- **Primary Goal:** Long-term compounding and capital preservation.
- **Scope:** Blue-chip stocks, fundamentally strong Mid-caps, ETFs.
- **Workflow:**
    1. **Screen:** Weekly fundamental scans (PE, PEG, Debt-to-Equity, Cash Flow).
    2. **Rebalance:** Monthly evaluation of sector exposure and SIP adjustments.
    3. **Portfolio Health:** Monitors overall drawdown and diversification.

## 3. Technology Stack & Integrations
- **Agent Framework:** Hermes Agent (via Composio).
- **Inference Model:** Claude 3.5 Sonnet (Logic & Analysis).
- **Tooling Layer:** Composio MCP.
    - **Brokers:** Zerodha (Kite Connect) or Angel One (SmartAPI).
    - **Search/News:** Tavily / Exa Search (Indian Market Sentiment).
    - **Data:** CoinMarketCap (for Crypto-diversification) and Finage (Live Price Feeds).
- **Communication:** Telegram Bot (Hermes Gateway).

## 4. Implementation Phases

### Phase 1: Core Foundation
- Setup Hermes `alpha-portfolio` profile.
- Configure Telegram Gateway for notifications.
- Integrate Brokerage (Zerodha/Angel One) in "Read-Only/Paper" mode.

### Phase 2: Momentum Engine Development
- Create "Momentum Scanner" skill for Hermes.
- Define triggers for volume spikes and Nifty breakouts.
- Setup automated daily research summaries.

### Phase 3: Wealth Architect Development
- Create "Fundamental Screener" skill.
- Establish monthly rebalancing logic.
- Connect Google Sheets/Notion for unified P/L tracking.

### Phase 4: Live Deployment & Scaling
- Transition from Paper Trading to Live execution (with manual approval).
- Optimize risk guardrails and stop-loss automation.

## 5. Risk Management & Safety
- **SEBI Awareness:** The system explicitly notes the "90% loss rule" in F&O and implements strict stop-loss logic.
- **Read-Only First:** Initial deployment will be research-only to validate performance against the "50% monthly" goal.
- **Diversification:** The Wealth Architect enforces a minimum 40% allocation to low-volatility assets to hedge aggressive momentum plays.
