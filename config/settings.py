from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Data provider selection: "kite" (default) | "upstox"
    data_provider: str = "kite"

    # Zerodha Kite
    kite_api_key: str = ""
    kite_api_secret: str = ""
    kite_access_token: Optional[str] = None

    # Upstox v2
    upstox_api_key: Optional[str] = None
    upstox_api_secret: Optional[str] = None
    upstox_access_token: Optional[str] = None
    upstox_redirect_uri: str = "http://localhost:8080"

    # Database
    timescale_url: str = "postgresql://trader:password@localhost:5432/nse_trading"
    redis_url: str = "redis://localhost:6379/0"

    # News APIs — equities
    finnhub_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None

    # Crypto data providers (all free-tier)
    binance_api_key: Optional[str] = None     # optional — only needed for order placement
    binance_api_secret: Optional[str] = None  # optional — public market data needs no key
    coingecko_api_key: Optional[str] = None   # optional free key → 50 req/min (vs 30 anon)
    cryptopanic_api_key: Optional[str] = None # free key at cryptopanic.com/developers/api/

    # Crypto universe & strategy
    crypto_enabled: bool = False
    crypto_universe_size: int = 30            # top-N coins by market cap to scan
    crypto_min_volume_usd: float = 5_000_000  # 24 h volume floor (USD)
    crypto_signal_threshold: float = 0.65    # same default as equities
    crypto_max_position_pct: float = 0.01    # tighter cap — crypto is more volatile

    # Monitoring
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    mlflow_tracking_uri: str = "http://localhost:5001"

    # Performance tuning — override in .env to tune for your hardware/tier
    kite_rps: int = 3                  # Kite API: max historical requests/second
    bulk_ingest_batch_size: int = 500  # OHLCV rows to accumulate before a DB flush
    bulk_ingest_max_workers: int = 8   # Threads fetching from Kite API
    bulk_ingest_db_workers: int = 2    # Threads writing to DB concurrently

    # Strategy
    signal_threshold: float = 0.65
    max_position_pct: float = 0.02
    daily_dd_limit: float = 0.03
    weekly_dd_limit: float = 0.07
    paper_trade_mode: bool = True

    @field_validator("max_position_pct")
    @classmethod
    def cap_position_pct(cls, v: float) -> float:
        if v > 0.02:
            raise ValueError("max_position_pct cannot exceed 2% — non-negotiable risk constraint")
        return v

    @field_validator("daily_dd_limit")
    @classmethod
    def cap_daily_dd(cls, v: float) -> float:
        if v > 0.05:
            raise ValueError("daily_dd_limit cannot exceed 5%")
        return v


settings = Settings()  # type: ignore[call-arg]
