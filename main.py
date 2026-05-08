"""Langfuse OpenAI Proxy - Entry point.

Drop-in OpenAI-compatible endpoint with Langfuse tracing.
Traces every call to Langfuse before proxying to an upstream LLM backend.

Supports per-request project tracing via API key pairs:
  - Combined: Authorization: Bearer <public_key>|<secret_key>
  - Separate: Authorization: Bearer <secret_key> + X-Langfuse-Public-Key header
"""

from langfuse_openai_proxy.api.app import create_app

app = create_app()
