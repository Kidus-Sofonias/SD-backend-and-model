# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.api.deps, app.repositories.user_repository.
# Key symbols/vars: router, register, login, me.
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user, get_users_repo
from app.repositories.user_repository import SqlUserRepository, UserRecord
from app.schemas.auth import RegisterRequest, LoginRequest
from app.schemas.common import APIResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _user_payload(user: UserRecord) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "is_admin": user.is_admin,
    }


@router.post("/register", response_model=APIResponse)
def register(
    payload: RegisterRequest,
    users: SqlUserRepository = Depends(get_users_repo),
) -> APIResponse:
    service = AuthService(users)
    user = service.register(email=payload.email, password=payload.password)

    return APIResponse.ok(
        "auth.registered",
        data={"user": _user_payload(user)},
    )


@router.post("/login", response_model=APIResponse)
def login(
    payload: LoginRequest,
    users: SqlUserRepository = Depends(get_users_repo),
) -> APIResponse:
    service = AuthService(users)
    token, expires_seconds, user = service.login(email=payload.email, password=payload.password)

    return APIResponse.ok(
        "auth.logged_in",
        data={
            "token": {
                "access_token": token,
                "token_type": "bearer",
                "expires_in_seconds": expires_seconds,
            },
            "user": _user_payload(user),
        },
    )


@router.get("/me", response_model=APIResponse)
def me(current_user: UserRecord = Depends(get_current_user)) -> APIResponse:
    return APIResponse.ok(
        "auth.me",
        data={"user": _user_payload(current_user)},
    )
