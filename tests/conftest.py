"""Shared test fixtures.

Unit tests use the in-process ASGI client (no real backends).
Integration tests use PROXY_URL from env to hit a real deployment.
"""

import os

import pytest
import httpx

from langfuse_openai_proxy.api.app import create_app


# --- Unit test fixtures (in-process, no network) ---


@pytest.fixture
def app():
    """Create a fresh FastAPI app for unit testing."""
    return create_app()


@pytest.fixture
async def client(app):
    """Async test client hitting the app in-process (no network)."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def auth_headers():
    """Valid combined-format auth headers for unit testing."""
    return {
        "Authorization": "Bearer pk-lf-test-public|sk-lf-test-secret",
        "Content-Type": "application/json",
    }


# --- Integration test config (from env vars, skipped if missing) ---

PROXY_URL = os.environ.get("PROXY_URL")  # e.g. https://openai.pezserv.org
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: requires PROXY_URL env var")


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if PROXY_URL is not set."""
    if not PROXY_URL:
        skip = pytest.mark.skip(reason="PROXY_URL not set — skipping integration tests")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def proxy_url():
    """Base URL of the deployed proxy (from PROXY_URL env var)."""
    return PROXY_URL


@pytest.fixture
def real_auth():
    """Combined auth header value using real Langfuse credentials."""
    return f"Bearer {LANGFUSE_PUBLIC_KEY}|{LANGFUSE_SECRET_KEY}"


@pytest.fixture
def integration_headers(real_auth):
    """Auth headers for integration tests."""
    return {
        "Authorization": real_auth,
        "Content-Type": "application/json",
    }


@pytest.fixture
async def proxy(proxy_url):
    """Async HTTP client pointing at the real deployed proxy."""
    async with httpx.AsyncClient(base_url=proxy_url, timeout=60) as ac:
        yield ac
