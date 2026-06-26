"""Dependency injection wiring.

Constructor injection via FastAPI Depends(). No DI container needed.
"""

from functools import lru_cache

from fastapi import Depends

from ..domain.services import TracingService
from ..infrastructure.config import Settings
from ..infrastructure.langfuse_client import create_langfuse_client
from ..infrastructure.openai_client import create_openai_client


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def get_tracing_service(settings: Settings = Depends(get_settings)) -> TracingService:
    """Get TracingService with wired dependencies."""
    openai = create_openai_client(settings.upstream_base_url, settings.upstream_api_key)
    return TracingService(
        langfuse_client_factory=create_langfuse_client,
        openai_client=openai,
        upstream_base_url=settings.upstream_base_url,
        upstream_api_key=settings.upstream_api_key,
        reasoning_as_content=settings.reasoning_as_content,
        max_tokens_floor=settings.max_tokens_floor,
        chat_think_off=settings.chat_think_off,
    )
