# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: Settings, settings.
import hashlib
import json
import uuid

from pydantic import computed_field, field_validator, model_validator
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
        extra="ignore",
    )

    # Auth / Security — NO HARDCODED DEFAULTS. Must be set via .env or environment variables.
    # Set SECRET_KEY to a strong random value in production.
    # Set ADMIN_PASSWORD to a strong password.
    secret_key: str = ""
    access_token_expire_minutes: int = 60
    admin_email: str = "admin@sdb.com"
    admin_password: str = ""

    database_url: str = "sqlite:///./sdbackend.db"
    auto_retrain_enabled: bool = False
    auto_retrain_trip_interval: int = 100
    auto_retrain_skip_tests: bool = True
    route_snap_enabled: bool = True
    route_snap_base_url: str = "https://router.project-osrm.org"

    @model_validator(mode="after")
    def _ensure_secrets_in_production(self) -> "Settings":
        env = self.app_env.strip().lower()
        # Only require secrets in non-local, non-test environments
        is_development = env in {"", "local", "dev", "development", "test"}
        if is_development:
            # Auto-generate a secret key for local development if not provided
            if not self.secret_key:
                self.secret_key = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
            return self

        if not self.secret_key:
            raise ValueError(
                "SECRET_KEY is required in production. "
                "Set it in the .env file or as an environment variable."
            )
        if not self.admin_password:
            raise ValueError(
                "ADMIN_PASSWORD is required in production. "
                "Set it in the .env file or as an environment variable."
            )
        return self

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
