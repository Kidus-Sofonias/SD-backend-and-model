# File role: Shared core utilities for configuration, security, JWT handling, logging, and typed application errors.
# Connects to: app.core.errors.
# Key symbols/vars: _pwd_context, BCRYPT_MAX_BYTES, _bytes_len, hash_password.
from passlib.context import CryptContext
from app.core.errors import AppError

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

BCRYPT_MAX_BYTES = 72


def _bytes_len(s: str) -> int:
    return len(s.encode("utf-8"))


def hash_password(password: str) -> str:
    # bcrypt limitation: 72 bytes max
    if _bytes_len(password) > BCRYPT_MAX_BYTES:
        raise AppError(
            message_key="auth.password_too_long",
            status_code=422,
            details={"max_bytes": BCRYPT_MAX_BYTES},
        )
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    # If user enters too-long password, treat as invalid safely
    if _bytes_len(plain_password) > BCRYPT_MAX_BYTES:
        return False
    return _pwd_context.verify(plain_password, hashed_password)
