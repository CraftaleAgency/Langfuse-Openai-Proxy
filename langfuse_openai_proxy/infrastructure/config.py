"""Configuration from environment variables.

No framework imports — pure dataclass reading env vars.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Application settings from environment variables."""

    upstream_base_url: str = os.environ.get("UPSTREAM_BASE_URL", "http://localhost:4000/v1")
    upstream_api_key: str = os.environ.get("UPSTREAM_API_KEY", "")
    langfuse_default_host: str = os.environ.get(
        "LANGFUSE_DEFAULT_HOST", "https://cloud.langfuse.com"
    )
    # Some upstreams (notably Ollama's /v1 endpoint serving reasoning models like
    # gemma4/qwen3) stream the model's output in the non-standard `delta.reasoning`
    # field, leaving `delta.content` empty. OpenAI-compatible clients that only read
    # `content` (e.g. OpenClaw's openai-completions adapter) then see an empty
    # response and abort with stop_reason=length. When enabled, the proxy copies
    # any `reasoning` text into `content` (keeping `reasoning` too) so every client
    # sees a normal content stream. Default off to preserve tracing fidelity for
    # clients that DO distinguish reasoning from content.
    reasoning_as_content: bool = os.environ.get("REASONING_AS_CONTENT", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
