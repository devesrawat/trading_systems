#!/bin/bash
# ==============================================================================
# Project Bootstrap Script
# Handles environment checks, dependencies, and initial data seeding.
# ==============================================================================

set -e

echo "🚀 Starting Trading System Bootstrap..."

# 1. Check for prerequisites
echo "🔍 Checking prerequisites..."
command -v uv >/dev/null 2>&1 || { echo "❌ Error: 'uv' is not installed. Visit https://github.com/astral-sh/uv"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ Error: 'docker' is not installed."; exit 1; }

# 2. Setup Environment
echo "📂 Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Created .env from .env.example. Please edit it with your API keys!"
else
    echo "✅ .env file exists."
fi

# 3. Install Dependencies
echo "📦 Installing Python dependencies..."
uv sync

# 4. Start Infrastructure
echo "🐳 Starting Docker infrastructure..."
docker compose up -d timescaledb redis mlflow
echo "⏳ Waiting for database to be ready..."
sleep 5

# 5. Initialize Database
echo "🗄️  Initializing database schema..."
uv run python -m data.store --init-schema

# 6. Seed Data (if possible)
echo "🌱 Checking for instrument universe..."
if [ ! -f config/instruments.json ]; then
    echo "⬇️  Instruments not found. Attempting refresh (requires valid .env)..."
    uv run python -m data.universe --refresh || echo "⚠️  Refresh failed. Ensure KITE_API_KEY is set in .env."
else
    echo "✅ Instrument universe exists."
fi

echo "=============================================================================="
echo "✅ BOOTSTRAP COMPLETE!"
echo "=============================================================================="
echo "Next steps:"
echo "1. Ensure your .env has KITE_API_KEY and KITE_API_SECRET."
echo "2. Run 'make login' to authorize Zerodha."
echo "3. Run 'make backfill-ohlcv' to seed historical data."
echo "4. Run 'make run-paper' to start trading."
echo "=============================================================================="
