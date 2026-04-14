# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: Settings, settings.
import json

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "safe-driving-api"
    app_env: str = "local"
    app_version: str = "0.1.0"
    debug: bool = False

    host: str = "127.0.0.1"
    port: int = 8000

    log_level: str = "INFO"
    cors_origins_raw: str = (
        "http://localhost:8081,"
        "http://127.0.0.1:8081,"
        "http://localhost:19006,"
        "http://127.0.0.1:19006,"
        "http://localhost:3000,"
        "http://127.0.0.1:3000"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )
    # Auth / Security
    secret_key: str = "WkJKHgQU7u_FxeQpG2HYv95v-LUCE1rwayUomwq75vrXii-bOV1zB-9A07rsu0E0WQoFS3r7TdRukx9Kl5yI2w"
    access_token_expire_minutes: int = 60
    admin_email: str = "admin@sdb.com"
    admin_password: str = "admin123"

    database_url: str = "sqlite:///./sdbackend.db"
    auto_retrain_enabled: bool = False
    auto_retrain_trip_interval: int = 100
    auto_retrain_skip_tests: bool = True

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, value):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"release", "prod", "production"}:
                return False
            if lowered in {"debug", "dev", "development"}:
                return True
        return value

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("postgres://"):
                return "postgresql://" + stripped[len("postgres://") :]
            return stripped
        return value

    @field_validator("cors_origins_raw", mode="before")
    @classmethod
    def normalize_cors_origins(cls, value):
        if isinstance(value, str):
            return value.strip()
        return value

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        stripped = self.cors_origins_raw.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in stripped.split(",") if item.strip()]

settings = Settings()
