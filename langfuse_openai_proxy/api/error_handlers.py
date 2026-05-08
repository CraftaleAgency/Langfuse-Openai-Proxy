"""Domain error to HTTP response mapping.

FastAPI exception handlers that convert domain errors to appropriate
HTTP status codes and JSON responses.
"""

from fastapi import Request
from fastapi.responses import JSONResponse

from ..domain.errors import MissingCredentialsError, ProxyError, UpstreamError


def register_error_handlers(app) -> None:
    """Register domain error handlers with the FastAPI app."""

    @app.exception_handler(MissingCredentialsError)
    async def missing_credentials_handler(
        request: Request, exc: MissingCredentialsError
    ):
        return JSONResponse(
            status_code=401,
            content={
                "error": {
                    "message": "Missing Langfuse credentials. Provide Authorization header (secret key) and X-Langfuse-Public-Key header."
                }
            },
        )

    @app.exception_handler(UpstreamError)
    async def upstream_error_handler(request: Request, exc: UpstreamError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": exc.message}},
        )

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError):
        return JSONResponse(
            status_code=400, content={"error": {"message": exc.message}}
        )
