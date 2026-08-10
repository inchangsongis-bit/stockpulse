from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    polygon_api_key: str = ""
    finnhub_api_key: str = ""
    database_url: str = "sqlite+aiosqlite:///./stockpulse.db"

    model_config = {"env_file": "../.env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
