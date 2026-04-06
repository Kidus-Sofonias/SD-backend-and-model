# File role: Dependency providers used by route handlers (auth user, database session, shared request context).
# Connects to: fastapi, app.core.errors, app.core.jwt.
# Key symbols/vars: bearer_scheme, get_users_repo, get_current_user.
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from sqlalchemy.orm import Session

from app.core.errors import UnauthorizedError
from app.core.jwt import decode_token

from app.db.session import get_db
from app.repositories.user_repository import SqlUserRepository, UserRecord

bearer_scheme = HTTPBearer(auto_error=False)

# # Single in-memory repo instance for the app process (Step 1 only)
# _users_repo = InMemoryUserRepository()


def get_users_repo(db:Session = Depends(get_db)) -> SqlUserRepository:
    return SqlUserRepository(db)


def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    users: SqlUserRepository = Depends(get_users_repo),
) -> UserRecord:
    if creds is None or not creds.credentials:
        raise UnauthorizedError(message_key="auth.missing_token")

    token = creds.credentials
    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise UnauthorizedError(message_key="auth.invalid_token")
    except JWTError:
        raise UnauthorizedError(message_key="auth.invalid_token")

    user = users.get_by_id(user_id)
    if not user:
        raise UnauthorizedError(message_key="auth.user_not_found")

    return user
