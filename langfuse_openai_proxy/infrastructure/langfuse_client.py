"""Langfuse client factory.

Creates per-request Langfuse v4 clients for multi-project tracing.
Uses base_url (v4) instead of deprecated host parameter.
"""

from langfuse import Langfuse


def create_langfuse_client(public_key: str, secret_key: str, host: str) -> Langfuse:
    """Create a Langfuse client with the given credentials.

    Args:
        public_key: Langfuse public key (pk-lf-*)
        secret_key: Langfuse secret key (sk-lf-*)
        host: Langfuse host URL

    Returns:
        Langfuse client instance
    """
    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=host,  # v4 uses base_url; 'host' is deprecated
    )
