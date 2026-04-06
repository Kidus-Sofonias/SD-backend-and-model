# File role: API exception translation layer that converts domain/runtime errors into consistent HTTP responses.
# Connects to: fastapi, app.core.errors, app.schemas.error.
# Key symbols/vars: logger, _get_request_id, app_error_handler, http_exception_handler.
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import AppError
from app.schemas.error import ErrorResponse, ErrorDetail

logger = logging.getLogger("app.exceptions")


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def app_error_handler(request: Request, exc: AppError):
    request_id = _get_request_id(request)
    payload = ErrorResponse(
        error=ErrorDetail(message_key=exc.message_key, details=exc.details),
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id = _get_request_id(request)

    # Map common HTTP errors to message keys (Flutter localizes)
    status_map = {
        401: "error.unauthorized",
        403: "error.forbidden",
        404: "error.not_found",
        405: "error.method_not_allowed",
        429: "error.rate_limited",
    }
    message_key = status_map.get(exc.status_code, "error.http")

    payload = ErrorResponse(
        error=ErrorDetail(message_key=message_key, details={"status_code": exc.status_code}),
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = _get_request_id(request)

    # Pydantic/FastAPI validation errors (bad request body/query params)
    payload = ErrorResponse(
        error=ErrorDetail(message_key="error.validation", details=exc.errors()),
        request_id=request_id,
    )
    return JSONResponse(status_code=422, content=payload.model_dump())


async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = _get_request_id(request)

    # Log full exception server-side; return safe error to client
    logger.exception("Unhandled exception", extra={"request_id": request_id})

    payload = ErrorResponse(
        error=ErrorDetail(message_key="error.internal_server", details=None),
        request_id=request_id,
    )
    return JSONResponse(status_code=500, content=payload.model_dump())
