# File role: Application entrypoint that wires FastAPI app, middleware, and versioned routers.
# Connects to: fastapi, app.core.config, app.core.logging.
# Key symbols/vars: create_app, app.
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.router import api_v1_router
from app.api.middleware.request_id import RequestIDMiddleware
from app.api.exceptions.handlers import (
    app_error_handler,
    http_exception_handler,
    validation_exception_handler,
    unhandled_exception_handler,
    )

from app.core.errors import AppError

from app.db.init_db import init_db

logger = logging.getLogger("app.startup")

@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except Exception:
        logger.exception("Application startup failed during database initialization.")
        raise
    yield

def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    # Router
    app.include_router(api_v1_router, prefix="/api/v1")

    # Exception handlers
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    return app


app = create_app()
