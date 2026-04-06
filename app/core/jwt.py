# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: app.core.config.
# Key symbols/vars: ALGORITHM, create_access_token, decode_token.
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import jwt
from app.core.config import settings

ALGORITHM = "HS256"

def create_access_token(subject: str, expires_minutes: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> str:
    now = datetime.now(timezone.utc)
    exp_minutes = expires_minutes if expires_minutes is not None else settings.access_token_expire_minutes
    expire = now + timedelta(minutes=exp_minutes)

    payload: Dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp())
    }
    if extra:
        payload.update(extra)

    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])