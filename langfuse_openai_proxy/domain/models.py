"""Domain models (value objects) for the langfuse-proxy.

Pure dataclasses with no framework dependencies.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Credentials:
    """Langfuse credentials extracted from request headers."""

    public_key: str
    secret_key: str


def parse_combined_credentials(raw: str) -> Credentials | None:
    """Parse a combined pk/sk credential string into Credentials.

    Order- and separator-insensitive: finds the `pk-lf-…` and `sk-lf-…` pieces
    anywhere in the string (handles `pk|sk`, `sk,pk`, `pk:sk`, concatenated, or
    any other arrangement). Returns None unless BOTH keys are present.

    Shared by the OpenAI chat path and the Anthropic Messages shim so they
    accept the same single-credential formats.
    """
    pk_start = raw.find("pk-lf-")
    sk_start = raw.find("sk-lf-")
    if pk_start == -1 or sk_start == -1:
        return None
    if pk_start < sk_start:
        public_key = raw[pk_start:sk_start].strip(" |,:")
        secret_key = raw[sk_start:].strip(" |,:")
    else:
        secret_key = raw[sk_start:pk_start].strip(" |,:")
        public_key = raw[pk_start:].strip(" |,:")
    return Credentials(public_key=public_key, secret_key=secret_key)


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


@dataclass
class ResponsesRequest:
    """Responses API request parameters."""

    model: str
    input: str | list[dict]
    stream: bool = False
    extra_params: dict | None = None
