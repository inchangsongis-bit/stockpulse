from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    polygon_api_key: str = ""
    finnhub_api_key: str = ""
    resend_api_key: str = ""
    # Defaults to Resend's shared sandbox sender, which works without
    # verifying a domain but can only deliver to the account owner's own
    # verified address(es). Set to a from-address on a verified domain in
    # Resend to send to arbitrary subscribers.
    resend_from_address: str = "StockPulse <onboarding@resend.dev>"
    database_url: str = "sqlite+aiosqlite:///./stockpulse.db"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
