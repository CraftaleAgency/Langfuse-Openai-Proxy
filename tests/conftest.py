"""Shared test fixtures."""

import pytest
from httpx import ASGITransport, AsyncClient

from langfuse_openai_proxy.api.app import create_app


@pytest.fixture
def app():
    """Create a fresh FastAPI app for testing."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
