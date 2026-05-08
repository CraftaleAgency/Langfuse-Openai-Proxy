"""Domain models (value objects) for the langfuse-proxy.

Pure dataclasses with no framework dependencies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Credentials:
    """Langfuse credentials extracted from request headers."""

    public_key: str
    secret_key: str


@dataclass
class ChatRequest:
    """Chat completion request parameters."""

    model: str
    messages: list[dict]
    stream: bool = False
    extra_params: dict | None = None


@dataclass
class EmbeddingRequest:
    """Embedding request parameters."""

    model: str
    input: list | str
    extra_params: dict | None = None
