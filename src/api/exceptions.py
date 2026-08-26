"""
API exception handlers.
"""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse


logger = logging.getLogger(__name__)


async def inference_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle inference errors without exposing internals."""

    logger.exception(
        "Inference request failed: %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": str(exc),
        },
    )


async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected server errors."""

    logger.exception(
        "Unexpected API error: %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
        },
    )