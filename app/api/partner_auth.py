from __future__ import annotations

from dataclasses import dataclass
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_api_key
from app.core.rate_limit import PARTNER_RATE_LIMITER
from app.db.models.organization import Organization, PartnerApiKey
from app.db.session import get_db


@dataclass(frozen=True)
class PartnerContext:
    organization_id: str
    api_key_id: str


def get_partner_context(
    request: Request,
    x_api_key: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> PartnerContext:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key is required")
    key = db.execute(
        select(PartnerApiKey).where(
            PartnerApiKey.key_hash == hash_api_key(x_api_key),
            PartnerApiKey.active.is_(True),
        )
    ).scalar_one_or_none()
    if key is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    organization = db.get(Organization, key.organization_id)
    if organization is None or not organization.active:
        raise HTTPException(status_code=403, detail="Organization is inactive")
    if not PARTNER_RATE_LIMITER.allow(f"api-key:{key.id}"):
        raise HTTPException(status_code=429, detail="Partner API rate limit exceeded")
    return PartnerContext(organization_id=organization.id, api_key_id=key.id)