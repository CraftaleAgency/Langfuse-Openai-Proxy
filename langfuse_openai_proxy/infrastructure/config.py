"""Configuration from environment variables.

No framework imports — pure dataclass reading env vars.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings from environment variables."""

    upstream_base_url: str = os.environ.get(
        "UPSTREAM_BASE_URL", "http://localhost:4000/v1"
    )
    upstream_api_key: str = os.environ.get("UPSTREAM_API_KEY", "")
    langfuse_default_host: str = os.environ.get(
        "LANGFUSE_DEFAULT_HOST", "https://cloud.langfuse.com"
    )
