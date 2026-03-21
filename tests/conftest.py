"""
Pytest configuration — injects minimal env vars so pydantic Settings
can be imported in all tests without a real .env file.
"""
import os

# Set required fields before any module imports trigger Settings()
os.environ.setdefault("KITE_API_KEY", "test_api_key")
os.environ.setdefault("KITE_API_SECRET", "test_api_secret")
os.environ.setdefault("KITE_ACCESS_TOKEN", "test_access_token")
os.environ.setdefault("TIMESCALE_URL", "postgresql://trader:password@localhost:5432/nse_trading")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("PAPER_TRADE_MODE", "true")
