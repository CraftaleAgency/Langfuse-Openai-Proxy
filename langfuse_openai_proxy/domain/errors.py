"""Domain errors for the langfuse-proxy.

Framework-agnostic error hierarchy. These are mapped to HTTP responses
in the API layer.
"""


class ProxyError(Exception):
    """Base class for all proxy domain errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class MissingCredentialsError(ProxyError):
    """Raised when Langfuse credentials are missing or incomplete."""


class UpstreamError(ProxyError):
    """Raised when the upstream LLM proxy returns an error."""

    def __init__(self, message: str, status_code: int = 502):
        self.status_code = status_code
        super().__init__(message)
