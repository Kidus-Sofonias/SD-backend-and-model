# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: AppError, NotFoundError, UnauthorizedError, ForbiddenError.
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class AppError(Exception):
    """
    Base application error. Use message_key for localization in Flutter.
    """
    message_key: str
    details: Optional[Any] = None
    status_code: int = 400


@dataclass
class NotFoundError(AppError):
    status_code: int = 404


@dataclass
class UnauthorizedError(AppError):
    status_code: int = 401


@dataclass
class ForbiddenError(AppError):
    status_code: int = 403
