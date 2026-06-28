"""FastAPI route handlers.

All endpoint definitions, credential parsing, and request/response handling.
Delegates business logic to TracingService.
"""

import json

import httpx
from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response, StreamingResponse

from ..domain.errors import MissingCredentialsError, UpstreamError
from ..domain.models import (
    ChatRequest,
    Credentials,
    EmbeddingRequest,
    ResponsesRequest,
    parse_combined_credentials,
)
from ..domain.services import TracingService
from ..infrastructure.config import Settings
from ..infrastructure.host_validation import validate_langfuse_host
from ..infrastructure.openai_client import create_openai_client, get_http_client
from .dependencies import get_settings, get_tracing_service

router = APIRouter()


def _extract_by_prefix(raw: str) -> Credentials | None:
    """Extract keys by finding pk-lf- and sk-lf- prefixes in a concatenated string.

    Thin wrapper over the shared domain helper so the OpenAI chat path and the
    Anthropic shim parse combined credentials identically.
    """
    return parse_combined_credentials(raw)


def parse_credentials(
    authorization: str | None,
    x_langfuse_public_key: str | None,
    query_public_key: str | None = None,
) -> Credentials:
    """Extract Langfuse public_key and secret_key from request headers.

    Supports multiple formats:
      - Separated: Authorization: Bearer <public_key>|<secret_key> (also , or :)
      - Concatenated: Authorization: Bearer pk-lf-...sk-lf-... (prefix detection)
      - Separate: Authorization: Bearer <secret_key> + X-Langfuse-Public-Key header
      - Query param: Authorization: Bearer <secret_key> + ?langfuse_pk=<public_key>

    Raises:
        MissingCredentialsError: If credentials are missing or incomplete
    """
    raw = (authorization or "").removeprefix("Bearer ").strip()
    if "|" in raw:
        public_key, secret_key = raw.split("|", 1)
        return Credentials(public_key=public_key.strip(), secret_key=secret_key.strip())

    if "," in raw:
        public_key, secret_key = raw.split(",", 1)
        return Credentials(public_key=public_key.strip(), secret_key=secret_key.strip())

    if ":" in raw:
        public_key, secret_key = raw.split(":", 1)
        return Credentials(public_key=public_key.strip(), secret_key=secret_key.strip())

    # Try prefix-based extraction (e.g., pk-lf-abc...sk-lf-xyz)
    creds = _extract_by_prefix(raw)
    if creds:
        return creds

    secret_key = raw
    public_key = (x_langfuse_public_key or query_public_key or "").strip()

    if not secret_key or not public_key:
        raise MissingCredentialsError(
            "Missing Langfuse credentials."
            " Provide Authorization header (secret key)"
            " and X-Langfuse-Public-Key header."
        )

    return Credentials(public_key=public_key, secret_key=secret_key)


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/v1/models")
async def list_models(
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    settings: Settings = Depends(get_settings),
):
    """Proxy model list from upstream. Requires authentication."""
    parse_credentials(authorization, x_langfuse_public_key, request.query_params.get("langfuse_pk"))
    openai = create_openai_client(settings.upstream_base_url, settings.upstream_api_key)
    models = await openai.models.list()
    return models.model_dump()


@router.get("/v1/models/{model}")
async def get_model(
    model: str,
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    settings: Settings = Depends(get_settings),
):
    """Proxy single model info from upstream, with list fallback. Requires authentication."""
    parse_credentials(authorization, x_langfuse_public_key, request.query_params.get("langfuse_pk"))
    openai = create_openai_client(settings.upstream_base_url, settings.upstream_api_key)

    # Try direct lookup first
    try:
        result = await openai.models.retrieve(model)
        return result.model_dump()
    except Exception:
        pass

    # Fallback: filter from list (some backends don't support single lookup)
    models = await openai.models.list()
    for m in models.data:
        if m.id == model:
            return m.model_dump()

    return {"error": {"message": f"Model {model} not found"}}


def _parse_reasoning_as_content(value: str | None) -> bool | None:
    """Parse the X-Reasoning-As-Content header into a tri-state bool.

    True/False forces fold-on/fold-off for this one request; None (header
    absent or unrecognized) falls back to the service default
    (Settings.reasoning_as_content). A reasoning-aware client sends ``false``
    to receive reasoning in its own SSE field instead of folded into content,
    so its thinking panel is populated and the answer stream stays clean —
    without affecting content-only clients that rely on the global fold.
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    x_langfuse_host: str | None = Header(None, alias="X-Langfuse-Host"),
    x_reasoning_as_content: str | None = Header(None, alias="X-Reasoning-As-Content"),
    tracing_service: TracingService = Depends(get_tracing_service),
    settings: Settings = Depends(get_settings),
):
    """Proxy chat completions with Langfuse tracing."""
    langfuse_pk = request.query_params.get("langfuse_pk")
    credentials = parse_credentials(authorization, x_langfuse_public_key, langfuse_pk)
    raw_host = x_langfuse_host or settings.langfuse_default_host
    # Security: validate user-supplied host to prevent SSRF / credential exfiltration
    host = validate_langfuse_host(raw_host) if x_langfuse_host else raw_host
    # Per-request reasoning-fold override (see _parse_reasoning_as_content).
    reasoning_override = _parse_reasoning_as_content(x_reasoning_as_content)

    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    # Extract extra params (everything except model, messages, stream)
    extra_params = {k: v for k, v in body.items() if k not in ("model", "messages", "stream")}

    chat_request = ChatRequest(
        model=model, messages=messages, stream=stream, extra_params=extra_params
    )

    if stream:
        return StreamingResponse(
            tracing_service.stream_chat_completion(
                credentials, chat_request, host, reasoning_as_content=reasoning_override
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return await tracing_service.chat_completion(
        credentials, chat_request, host, reasoning_as_content=reasoning_override
    )


@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    x_langfuse_host: str | None = Header(None, alias="X-Langfuse-Host"),
    tracing_service: TracingService = Depends(get_tracing_service),
    settings: Settings = Depends(get_settings),
):
    """Proxy embeddings with Langfuse tracing."""
    langfuse_pk = request.query_params.get("langfuse_pk")
    credentials = parse_credentials(authorization, x_langfuse_public_key, langfuse_pk)
    raw_host = x_langfuse_host or settings.langfuse_default_host
    # Security: validate user-supplied host to prevent SSRF / credential exfiltration
    host = validate_langfuse_host(raw_host) if x_langfuse_host else raw_host

    body = await request.json()

    # Extract extra params (everything except model, input)
    extra_params = {k: v for k, v in body.items() if k not in ("model", "input")}

    embedding_request = EmbeddingRequest(
        model=body.get("model", ""), input=body.get("input", []), extra_params=extra_params
    )

    return await tracing_service.embedding(credentials, embedding_request, host)


@router.post("/v1/responses")
async def responses(
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    x_langfuse_host: str | None = Header(None, alias="X-Langfuse-Host"),
    tracing_service: TracingService = Depends(get_tracing_service),
    settings: Settings = Depends(get_settings),
):
    """Proxy Responses API with Langfuse tracing."""
    langfuse_pk = request.query_params.get("langfuse_pk")
    credentials = parse_credentials(authorization, x_langfuse_public_key, langfuse_pk)
    raw_host = x_langfuse_host or settings.langfuse_default_host
    host = validate_langfuse_host(raw_host) if x_langfuse_host else raw_host

    body = await request.json()
    model = body.get("model", "")
    input_data = body.get("input", "")
    stream = body.get("stream", False)
    extra_params = {k: v for k, v in body.items() if k not in ("model", "input", "stream")}

    responses_request = ResponsesRequest(
        model=model, input=input_data, stream=stream, extra_params=extra_params
    )

    if stream:
        return StreamingResponse(
            tracing_service.stream_response(credentials, responses_request, host),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result, status_code = await tracing_service.response(credentials, responses_request, host)
    return Response(
        content=json.dumps(result),
        status_code=status_code,
        media_type="application/json",
    )


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy_passthrough(
    path: str,
    request: Request,
    authorization: str | None = Header(None),
    x_langfuse_public_key: str | None = Header(None, alias="X-Langfuse-Public-Key"),
    settings: Settings = Depends(get_settings),
):
    """Generic passthrough for any /v1/* endpoint not explicitly handled.

    Requires authentication to prevent unauthenticated SSRF.
    Rejects path traversal sequences for defense in depth.
    """
    # Security: require authentication on all passthrough requests
    parse_credentials(authorization, x_langfuse_public_key, request.query_params.get("langfuse_pk"))

    # Security: reject path traversal sequences
    if ".." in path or "//" in path:
        return Response(
            content='{"error":{"message":"Invalid path"}}',
            status_code=400,
            media_type="application/json",
        )

    http = get_http_client()
    upstream_url = f"{settings.upstream_base_url}/{path}"
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    headers = {}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"

    body = await request.body()
    try:
        resp = await http.request(
            method=request.method,
            url=upstream_url,
            headers=headers,
            content=body or None,
            timeout=120,
        )
    except httpx.TransportError as e:
        raise UpstreamError("Upstream unreachable") from e
    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
