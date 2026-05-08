"""Tests for configuration."""

import os

import pytest

from langfuse_openai_proxy.infrastructure.config import Settings


def test_defaults():
    """Settings should use env vars, falling back to defaults."""
    for key in ("UPSTREAM_BASE_URL", "UPSTREAM_API_KEY", "LANGFUSE_DEFAULT_HOST"):
        os.environ.pop(key, None)

    settings = Settings(
        upstream_base_url=os.environ.get("UPSTREAM_BASE_URL", "http://localhost:4000/v1"),
        upstream_api_key=os.environ.get("UPSTREAM_API_KEY", ""),
        langfuse_default_host=os.environ.get("LANGFUSE_DEFAULT_HOST", "https://cloud.langfuse.com"),
    )
    assert settings.upstream_base_url == "http://localhost:4000/v1"
    assert settings.upstream_api_key == ""
    assert settings.langfuse_default_host == "https://cloud.langfuse.com"


def test_settings_is_frozen():
    settings = Settings()
    with pytest.raises(AttributeError):
        settings.upstream_base_url = "changed"
