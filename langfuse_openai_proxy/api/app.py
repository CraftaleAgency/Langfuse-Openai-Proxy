"""FastAPI app factory.

Creates and configures the FastAPI application with all routes and error handlers.
"""

from fastapi import FastAPI

from .anthropic_routes import router as anthropic_router
from .error_handlers import register_error_handlers
from .routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Langfuse OpenAI Proxy",
        description=(
            "OpenAI-compatible proxy with Langfuse tracing."
            " Supports per-request project tracing via API keys."
        ),
    )
    # Anthropic router MUST come before the main router — routes.py defines a
    # catch-all /v1/{path:path} that would otherwise shadow /v1/messages.
    app.include_router(anthropic_router)
    app.include_router(router)
    register_error_handlers(app)
    return app
