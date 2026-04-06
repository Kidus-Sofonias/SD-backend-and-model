# File role: Business-logic service that coordinates repositories/schemas and enforces use-case rules.
# Connects to: app.core.errors, app.core.security, app.core.jwt.
# Key symbols/vars: AuthService.
from app.core.errors import AppError, UnauthorizedError
from app.core.security import hash_password, verify_password
from app.core.jwt import create_access_token
from app.core.config import settings
from app.repositories.user_repository import SqlUserRepository, UserRecord


class AuthService:
    def __init__(self, users: SqlUserRepository) -> None:
        self.users = users

    def register(self, email: str, password: str) -> UserRecord:
        existing = self.users.get_by_email(email)
        if existing:
            raise AppError(message_key="auth.email_already_exists", status_code=409)

        pwd_hash = hash_password(password)
        return self.users.create(email=email, password_hash=pwd_hash, role="driver")

    def login(self, email: str, password: str) -> tuple[str, int, UserRecord]:
        user = self.users.get_by_email(email)
        if not user or not verify_password(password, user.password_hash):
            raise UnauthorizedError(message_key="auth.invalid_credentials")

        token = create_access_token(subject=user.id)
        expires_seconds = settings.access_token_expire_minutes * 60
        return token, expires_seconds, user
