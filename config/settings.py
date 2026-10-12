"""Pydantic settings, loaded from .env at the repo root."""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    alpaca_key: str = ""
    alpaca_secret: str = ""
    alpaca_paper: bool = True
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"

    polygon_key: str = ""

    initial_equity: float = 25_000.0
    risk_pct_per_trade: float = 0.01
    max_leverage: float = 4.0
    daily_loss_limit: float = 0.025
    weekly_loss_limit: float = 0.05

    log_level: str = "INFO"

    data_dir: Path = Field(default_factory=lambda: REPO_ROOT / "data")
    reports_dir: Path = Field(default_factory=lambda: REPO_ROOT / "reports")
    logs_dir: Path = Field(default_factory=lambda: REPO_ROOT / "logs")


def get_settings() -> Settings:
    return Settings()
