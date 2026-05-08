"""FastAPI app factory.

Creates and configures the FastAPI application with all routes and error handlers.
"""

from fastapi import FastAPI

from .error_handlers import register_error_handlers
from .routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Langfuse OpenAI Proxy",
        description="OpenAI-compatible proxy with Langfuse tracing. Supports per-request project tracing via API keys.",
    )
    app.include_router(router)
    register_error_handlers(app)
    return app
