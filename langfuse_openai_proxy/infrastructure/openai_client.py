"""Upstream OpenAI client factory.

Uses the standard openai.AsyncOpenAI (NOT langfuse.openai.AsyncOpenAI
which is the monkey-patched version that causes the bug).
"""

import httpx
from openai import AsyncOpenAI

# Reusable httpx client for non-LLM passthrough endpoints
_http_client = httpx.AsyncClient(timeout=120)


def create_openai_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Create an OpenAI client pointing to an upstream LLM backend.

    Args:
        base_url: Upstream base URL (e.g., http://localhost:4000/v1)
        api_key: API key for upstream (may be empty)

    Returns:
        AsyncOpenAI client instance
    """
    return AsyncOpenAI(
        api_key=api_key or "none",
        base_url=base_url,
    )


def get_http_client() -> httpx.AsyncClient:
    """Get the reusable httpx client for non-LLM passthrough."""
    return _http_client
