# File role: Pydantic schema contract for request validation and response serialization across API layers.
# Connects to: nearby package modules via local imports.
# Key symbols/vars: RegisterRequest, LoginRequest, TokenData, UserPublic.
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    # bcrypt limit (72 bytes). Keep it 72 to avoid truncation/security issues.
    password: str = Field(min_length=8, max_length=72)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=72)


class TokenData(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class UserPublic(BaseModel):
    id: str
    email: EmailStr
    role: str = "driver"
    is_admin: bool = False


class RegisterResponse(BaseModel):
    user: UserPublic


class LoginResponse(BaseModel):
    token: TokenData
    user: UserPublic
