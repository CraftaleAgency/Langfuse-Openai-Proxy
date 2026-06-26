"""Anthropic Messages API shim routes.

Mounts /v1/messages, /v1/messages/count_tokens, and /v1/messages/{id} that
speak the Anthropic wire format Claude Code expects, translating to/from
the OpenAI Chat Completions format our TracingService already serves.

Auth model: Claude Code sends a single token (x-api-key or Bearer). We
validate it against LANGFUSE_SECRET_KEY from env, then trace using the
env's LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY. This is a trusted
single-client route — not multi-tenant like /v1/chat/completions.
"""

import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..domain.anthropic_translator import (
    anthropic_to_openai,
    estimate_tokens_anthropic,
    openai_to_anthropic_response,
    openai_to_anthropic_stream,
)
from ..domain.models import ChatRequest, Credentials
from ..domain.services import TracingService
from ..infrastructure.config import Settings
from ..infrastructure.langfuse_client import create_langfuse_client
from ..infrastructure.openai_client import create_openai_client
from .dependencies import get_settings

logger = logging.getLogger("uvicorn.error")
_SHIM_DEBUG = os.environ.get("ANTHROPIC_SHIM_DEBUG", "").lower() in ("1", "true", "yes")

router = APIRouter(prefix="/v1", tags=["anthropic"])


def _extract_anthropic_token(authorization: str | None, x_api_key: str | None) -> str | None:
    """Pull the single shared token from either accepted header."""
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        stripped = authorization.removeprefix("Bearer ").strip()
        return stripped or None
    return None


def _resolve_anthropic_credentials(token: str | None, settings: Settings) -> Credentials:
    """Validate the single token against env LANGFUSE_SECRET_KEY.

    Returns Credentials built from the env's Langfuse keys — the Anthropic
    path traces to the proxy's own Langfuse project, not a per-request one.
    """
    if not settings.langfuse_secret_key:
        # Misconfigured deployment — fail closed rather than allowing unauthenticated access.
        raise HTTPException(503, "Anthropic shim not configured: LANGFUSE_SECRET_KEY missing")
    if not token or token != settings.langfuse_secret_key:
        raise HTTPException(401, "invalid api key")
    return Credentials(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
    )


def _build_tracing_service(settings: Settings) -> TracingService:
    """Per-request TracingService with max_tokens_floor DISABLED.

    Anthropic clients always send explicit max_tokens per spec, so the floor
    that protects OpenAI clients from default-50 budgets must not apply here.

    reasoning_as_content is FORCED False here even if the global setting is on.
    That remap copies delta.reasoning into delta.content for OpenAI-only clients
    that don't read the reasoning field — but our anthropic_translator consumes
    delta.reasoning directly to emit Anthropic thinking blocks. With the remap
    on, every reasoning chunk carries BOTH fields and the translator emits
    interleaved thinking+text blocks (thinking→text→thinking), violating the
    Anthropic invariant that all thinking blocks must precede text. The
    translator owns reasoning handling on this path.
    """
    return TracingService(
        langfuse_client_factory=create_langfuse_client,
        openai_client=create_openai_client(settings.upstream_base_url, settings.upstream_api_key),
        upstream_base_url=settings.upstream_base_url,
        upstream_api_key=settings.upstream_api_key,
        reasoning_as_content=False,
        max_tokens_floor=None,
    )


def _resolve_physical_model(anthropic_model: str, settings: Settings) -> str:
    """Map an incoming Anthropic model name to its physical upstream model.

    Tries exact match first, then glob (`claude-sonnet-*`), then falls back
    to ANTHROPIC_DEFAULT_MODEL.
    """
    model_map = settings.anthropic_model_map
    if anthropic_model in model_map:
        return model_map[anthropic_model]

    # Glob match: pattern uses trailing `*`.
    for pattern, physical in model_map.items():
        if pattern.endswith("*") and anthropic_model.startswith(pattern[:-1]):
            return physical

    return settings.anthropic_default_model


async def _gen_openai_chunks(
    sse_iter: AsyncIterator[str],
) -> AsyncIterator[dict[str, Any]]:
    """Adapt TracingService's OpenAI SSE string stream into parsed chunk dicts.

    TracingService.stream_chat_completion yields lines like `data: {...}\\n\\n`
    (plus a terminal `data: [DONE]\\n\\n`). The anthropic_translator consumes
    parsed dicts, so this helper parses each `data:` payload back to JSON and
    skips the `[DONE]` sentinel and any non-data keepalive lines.
    """
    async for raw in sse_iter:
        # Each yielded item may contain one or more `data:` lines separated by blank lines.
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                yield parsed


def _gate(settings: Settings) -> None:
    """404 if the shim is disabled or paused. Called at the top of every route."""
    if not settings.anthropic_shim_enabled:
        raise HTTPException(404, "Anthropic shim not enabled")
    if settings.anthropic_paused:
        raise HTTPException(404, "Anthropic shim paused")


@router.post("/messages")
async def create_message(
    request: Request,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="x-api-key"),
    settings: Settings = Depends(get_settings),
):
    """Translate an Anthropic /v1/messages call to OpenAI and back."""
    _gate(settings)
    token = _extract_anthropic_token(authorization, x_api_key)
    credentials = _resolve_anthropic_credentials(token, settings)

    body = await request.json()
    requested_model = body.get("model", "")
    physical_model = _resolve_physical_model(requested_model, settings)

    openai_req = anthropic_to_openai(body)
    # Re-pin model to the physical upstream name (anthropic_to_openai forwards
    # whatever the client sent, which is the Anthropic alias, not the Ollama name).
    openai_req["model"] = physical_model

    chat_request = ChatRequest(
        model=physical_model,
        messages=openai_req["messages"],
        stream=bool(openai_req.get("stream", False)),
        extra_params={
            k: v for k, v in openai_req.items() if k not in ("model", "messages", "stream")
        },
    )

    service = _build_tracing_service(settings)
    host = settings.langfuse_default_host

    # Echo back the alias the client asked for, not the physical model name —
    # Claude Code's UI and cache key on the model field it sent.
    original_model = requested_model or physical_model

    if chat_request.stream:
        message_id = f"msg_{int(time.time() * 1000)}"

        if _SHIM_DEBUG:
            logger.info(
                "[shim] REQ model=%s phys=%s stream=true thinking=%s tools=%d "
                "msgs=%d max_tokens=%s",
                requested_model,
                physical_model,
                "thinking" in body,
                len(body.get("tools") or []),
                len(body.get("messages") or []),
                body.get("max_tokens"),
            )

        async def anthropic_event_stream() -> AsyncIterator[str]:
            openai_sse = service.stream_chat_completion(
                credentials, chat_request, host, apply_max_tokens_floor=False
            )
            chunk_iter = _gen_openai_chunks(openai_sse)
            blocks: list[str] = []
            stop_reason = None
            usage_seen = None
            async for evt in openai_to_anthropic_stream(
                chunk_iter,
                original_model=original_model,
                message_id=message_id,
                input_tokens=estimate_tokens_anthropic(body),
            ):
                if _SHIM_DEBUG:
                    try:
                        _, data_part = evt.split("data: ", 1)
                        parsed = json.loads(data_part.strip())
                    except (ValueError, json.JSONDecodeError):
                        parsed = None
                    if isinstance(parsed, dict):
                        if parsed.get("type") == "content_block_start":
                            blocks.append((parsed.get("content_block") or {}).get("type"))
                        elif parsed.get("type") == "message_delta":
                            stop_reason = (parsed.get("delta") or {}).get("stop_reason")
                            usage_seen = parsed.get("usage")
                yield evt
            if _SHIM_DEBUG:
                logger.info(
                    "[shim] RESP %s blocks=%s stop=%s usage=%s",
                    message_id,
                    blocks,
                    stop_reason,
                    usage_seen,
                )

        return StreamingResponse(
            anthropic_event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    openai_resp = await service.chat_completion(
        credentials, chat_request, host, apply_max_tokens_floor=False
    )
    anthro_resp = openai_to_anthropic_response(openai_resp, original_model=original_model)
    return JSONResponse(content=anthro_resp)


@router.post("/messages/count_tokens")
async def count_tokens(
    request: Request,
    authorization: str | None = Header(None),
    x_api_key: str | None = Header(None, alias="x-api-key"),
    settings: Settings = Depends(get_settings),
):
    """Estimate input tokens for an Anthropic request body.

    Claude Code uses this for context-window budgeting. We don't need tiktoken
    precision — a chars/4 heuristic is within ±15%, plenty for budgeting.
    """
    _gate(settings)
    token = _extract_anthropic_token(authorization, x_api_key)
    _resolve_anthropic_credentials(token, settings)

    body = await request.json()
    return JSONResponse(content={"input_tokens": estimate_tokens_anthropic(body)})


@router.get("/messages/{message_id}")
async def get_message(
    message_id: str,
    settings: Settings = Depends(get_settings),
):
    """Stub — Claude Code doesn't strictly need message retrieval.

    Returns 404; clients should consult the Langfuse UI for trace history.
    """
    _gate(settings)
    raise HTTPException(404, "message lookup not supported; check Langfuse UI")
