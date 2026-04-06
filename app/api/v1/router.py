# File role: API router aggregator that mounts route modules under the v1 prefix.
# Connects to: fastapi, app.api.v1.routes.health, app.api.v1.routes.auth.
# Key symbols/vars: api_v1_router.
from fastapi import APIRouter
from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.trips import router as trips_router
from app.api.v1.routes.events import router as events_router
from app.api.v1.routes.sensor_samples import router as sensor_samples_router


api_v1_router = APIRouter()
api_v1_router.include_router(health_router, tags=["health"])
api_v1_router.include_router(auth_router, tags=["auth"])
api_v1_router.include_router(admin_router)
api_v1_router.include_router(trips_router)
api_v1_router.include_router(events_router, tags=["events"])
api_v1_router.include_router(sensor_samples_router, prefix="/trips", tags=["sensor_samples"])
