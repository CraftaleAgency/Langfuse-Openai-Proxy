"""Domain error to HTTP response mapping.

FastAPI exception handlers that convert domain errors to appropriate
HTTP status codes and JSON responses.

All error responses use the Anthropic error envelope
``{"type": "error", "error": {"type": <t>, "message": <m>}}``. This shape is
what the Anthropic SDK (Claude Code) keys on — the top-level ``type`` and the
nested ``error.type`` drive its error class selection, while a bare
``{"error": {"message": ...}}`` makes Claude Code treat failures as an
unparseable "proxy intercepting" crash. The OpenAI SDK is also satisfied by
this envelope: it reads ``error.message``, which we preserve.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from ..domain.errors import MissingCredentialsError, ProxyError, UpstreamError


def _error_type_for_status(status_code: int) -> str:
    """Map an HTTP status code to an Anthropic error type string."""
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if status_code == 404:
        return "not_found_error"
    if status_code == 413:
        return "request_too_large"
    if status_code == 429:
        return "rate_limit_error"
    if status_code >= 500:
        return "overloaded_error"
    return "invalid_request_error"


def anthropic_error_response(status_code: int, message: str) -> JSONResponse:
    """Build an Anthropic-shaped JSONResponse for the given status + message.

    Public so route handlers that need to short-circuit (e.g. the streaming
    path, which can't raise after SSE headers are committed) can reuse it.
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": _error_type_for_status(status_code),
                "message": message,
            },
        },
    )


def register_error_handlers(app) -> None:
    """Register domain error handlers with the FastAPI app."""

    @app.exception_handler(MissingCredentialsError)
    async def missing_credentials_handler(request: Request, exc: MissingCredentialsError):
        return anthropic_error_response(
            401,
            (
                "Missing Langfuse credentials."
                " Provide Authorization header (secret key)"
                " and X-Langfuse-Public-Key header."
            ),
        )

    @app.exception_handler(UpstreamError)
    async def upstream_error_handler(request: Request, exc: UpstreamError):
        # Preserve the upstream status (4xx/5xx) rather than flattening to 502,
        # so a 429 from Ollama surfaces as rate_limit_error etc.
        return anthropic_error_response(exc.status_code, exc.message)

    @app.exception_handler(ProxyError)
    async def proxy_error_handler(request: Request, exc: ProxyError):
        return anthropic_error_response(400, exc.message)

    @app.exception_handler(ValueError)
    async def validation_error_handler(request: Request, exc: ValueError):
        return anthropic_error_response(400, str(exc))

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # Only the Anthropic shim raises HTTPException (auth 401, shim-gate
        # 404, misconfig 503). The OpenAI routes raise domain errors, so this
        # handler is effectively shim-scoped while still covering any 404/405
        # Starlette itself raises. The envelope satisfies both SDKs.
        status = exc.status_code
        detail = exc.detail if isinstance(exc.detail, str) else "request failed"
        return anthropic_error_response(status, detail)
