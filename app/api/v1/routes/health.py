# File role: HTTP route layer that maps requests to services/repositories and returns schema-shaped responses.
# Connects to: fastapi, app.core.config, app.schemas.common.
# Key symbols/vars: logger, router, health_check.
import logging
from fastapi import APIRouter, Request
from app.core.config import settings
from app.schemas.common import APIResponse
from app.schemas.health import HealthData

logger = logging.getLogger("app.health")

router = APIRouter()


@router.get("/health", response_model=APIResponse)
def health_check(request: Request) -> APIResponse:
    request_id = getattr(request.state, "request_id", "-")
    logger.info("Health check", extra={"request_id": request_id})

    data = HealthData(service=settings.app_name, env=settings.app_env, version=settings.app_version)
    return APIResponse.ok("health.ok", data=data.model_dump())
