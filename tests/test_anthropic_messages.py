"""Tests for the Anthropic /v1/messages shim.

Covers:
  - Non-streaming text + tool_use round-trip
  - Streaming event sequence (message_start → content_block_* → message_delta → message_stop)
  - Streaming tool_use with input_json_delta
  - /v1/messages/count_tokens returns int
  - Model name mapping (alias → physical) and default fallback
  - System-as-list flattening
  - tool_result block → role:tool message
  - max_tokens NOT floored on this path (Anthropic clients send explicit max_tokens)
  - Auth: rejects missing token, accepts Bearer + x-api-key
  - Shim gating: disabled → 404, paused → 404
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from langfuse_openai_proxy.api import anthropic_routes
from langfuse_openai_proxy.api.app import create_app
from langfuse_openai_proxy.api.dependencies import get_settings
from langfuse_openai_proxy.domain.models import ChatRequest, Credentials
from langfuse_openai_proxy.infrastructure.config import Settings

# --- Fakes ---------------------------------------------------------------

SECRET = "sk-lf-test-secret"
PUBLIC = "pk-lf-test-public"


class FakeTracingService:
    """Captures calls and returns canned OpenAI-shape responses.

    `behavior` selects between fixtures: "text", "tool_use", "stream_text",
    "stream_tool_use". Assertions read off the captured ChatRequest.
    """

    def __init__(self, behavior: str = "text") -> None:
        self.behavior = behavior
        self.captured_requests: list[ChatRequest] = []
        self.captured_apply_floor: list[bool] = []

    async def chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
        apply_max_tokens_floor: bool = True,
    ) -> dict[str, Any]:
        self.captured_requests.append(request)
        self.captured_apply_floor.append(apply_max_tokens_floor)
        if self.behavior == "tool_use":
            return {
                "id": "chatcmpl-test",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps({"city": "Boston"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
            }
        # default: text
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }

    async def stream_chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
        apply_max_tokens_floor: bool = True,
    ) -> AsyncIterator[str]:
        self.captured_requests.append(request)
        self.captured_apply_floor.append(apply_max_tokens_floor)
        if self.behavior == "stream_tool_use":
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_abc",
                                        "function": {"name": "get_weather", "arguments": ""},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"city":"Boston"}'},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            ]
        else:
            # default stream_text
            chunks = [
                {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                },
            ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"


def _make_settings(
    *,
    shim_enabled: bool = True,
    paused: bool = False,
    model_map_raw: str = "",
    default_model: str = "coder14b:latest",
) -> Settings:
    """Construct Settings with the given shim config and env creds."""
    return Settings(
        upstream_base_url="http://upstream/v1",
        upstream_api_key="",
        langfuse_default_host="https://langfuse.example",
        langfuse_public_key=PUBLIC,
        langfuse_secret_key=SECRET,
        anthropic_shim_enabled=shim_enabled,
        anthropic_paused=paused,
        anthropic_model_map_raw=model_map_raw,
        anthropic_default_model=default_model,
    )


@pytest.fixture
def client_factory(monkeypatch):
    """Factory: client_factory(settings, behavior) → (client, captured_services).

    Monkeypatches _build_tracing_service on the anthropic_routes module so the
    route handler uses FakeTracingService instead of a real one, and overrides
    get_settings via FastAPI DI so the route sees our test Settings.
    """
    created_app = None

    def build(
        settings: Settings,
        behavior: str = "text",
    ):
        nonlocal created_app

        captured: list[FakeTracingService] = []

        def fake_build_tracing_service(s: Settings) -> FakeTracingService:
            svc = FakeTracingService(behavior=behavior)
            captured.append(svc)
            return svc

        monkeypatch.setattr(
            anthropic_routes,
            "_build_tracing_service",
            fake_build_tracing_service,
        )

        created_app = create_app()
        created_app.dependency_overrides[get_settings] = lambda: settings

        import httpx

        transport = httpx.ASGITransport(app=created_app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        return client, captured

    return build


# --- Auth tests ----------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_rejects_missing_token(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_auth_rejects_wrong_token(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": "wrong", "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 401
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_auth_accepts_authorization_bearer(client_factory):
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"Authorization": f"Bearer {SECRET}", "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured, "TracingService should have been constructed"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_auth_accepts_x_api_key_header(client_factory):
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured
    finally:
        await client.aclose()


# --- Shim gating ---------------------------------------------------------


@pytest.mark.asyncio
async def test_shim_disabled_returns_404(client_factory):
    settings = _make_settings(shim_enabled=False)
    client, _ = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_shim_paused_returns_404(client_factory):
    settings = _make_settings(paused=True)
    client, _ = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 404
    finally:
        await client.aclose()


# --- Non-streaming translation ------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_text_round_trip(client_factory):
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"] == "claude-sonnet-4-5"
        assert body["stop_reason"] == "end_turn"
        # Single text block with our canned content.
        assert body["content"] == [{"type": "text", "text": "Hi there"}]
        assert body["usage"]["input_tokens"] == 5
        assert body["usage"]["output_tokens"] == 3
        assert body["usage"]["cache_creation_input_tokens"] == 0
        assert body["usage"]["cache_read_input_tokens"] == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_non_streaming_with_tool_use(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="tool_use")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["stop_reason"] == "tool_use"
        tool_blocks = [b for b in body["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "Boston"}
        assert tool_blocks[0]["id"].startswith("call_") or tool_blocks[0]["id"] == "call_abc"
    finally:
        await client.aclose()


# --- Streaming translation -----------------------------------------------


async def _read_sse_stream(resp) -> list[tuple[str, dict]]:
    """Parse an SSE response body into [(event_type, data_dict), ...]."""
    events: list[tuple[str, dict]] = []
    body = resp.content.decode() if hasattr(resp, "content") else await resp.aread()
    text = body if isinstance(body, str) else body.decode()
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event_type = ""
        data_str = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:") :].strip()
        if event_type and data_str:
            events.append((event_type, json.loads(data_str)))
    return events


@pytest.mark.asyncio
async def test_streaming_text_event_sequence(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="stream_text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        events = await _read_sse_stream(resp)
        event_types = [e[0] for e in events]
        # Must start with message_start and end with message_stop.
        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"
        # Required sequence in between.
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        # message_delta must come after content_block_stop.
        cbs_idx = event_types.index("content_block_stop")
        md_idx = event_types.index("message_delta")
        assert md_idx > cbs_idx
        # Verify the text deltas actually arrived.
        text_deltas = [
            e["delta"]["text"]
            for et, e in events
            if et == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
        ]
        assert "".join(text_deltas) == "Hello world"
        # message_start carries the Anthropic model alias, not the physical one.
        start = events[0][1]
        assert start["message"]["model"] == "claude-sonnet-4-5"
        # message_delta carries stop_reason derived from finish_reason=stop.
        md = [e for et, e in events if et == "message_delta"][0]
        assert md["delta"]["stop_reason"] == "end_turn"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_streaming_tool_use_input_json_delta(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="stream_tool_use")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "weather?"}],
            },
        )
        assert resp.status_code == 200
        events = await _read_sse_stream(resp)
        # Must have a content_block_start with type=tool_use.
        tool_starts = [
            e
            for et, e in events
            if et == "content_block_start" and e["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["name"] == "get_weather"
        # Must have input_json_delta events carrying the partial JSON.
        json_deltas = [
            e["delta"]["partial_json"]
            for et, e in events
            if et == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        assert "".join(json_deltas) == '{"city":"Boston"}'
        # Stop reason must be tool_use.
        md = [e for et, e in events if et == "message_delta"][0]
        assert md["delta"]["stop_reason"] == "tool_use"
    finally:
        await client.aclose()


# --- count_tokens --------------------------------------------------------


@pytest.mark.asyncio
async def test_count_tokens_returns_int(client_factory):
    settings = _make_settings()
    client, _ = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages/count_tokens",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello, world!"}],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["input_tokens"], int)
        assert body["input_tokens"] > 0
    finally:
        await client.aclose()


# --- Model mapping -------------------------------------------------------


@pytest.mark.asyncio
async def test_model_name_mapping_sonnet_to_coder14b(client_factory):
    settings = _make_settings(
        model_map_raw="claude-sonnet-*:coder14b:latest,claude-haiku-*:gemma4:12b",
    )
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured[0].captured_requests[0].model == "coder14b:latest"
        # Response echoes the alias, not the physical model.
        assert resp.json()["model"] == "claude-sonnet-4-5"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_model_name_mapping_haiku_glob(client_factory):
    settings = _make_settings(
        model_map_raw="claude-sonnet-*:coder14b:latest,claude-haiku-*:gemma4:12b",
    )
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-haiku-4-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured[0].captured_requests[0].model == "gemma4:12b"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_unknown_model_falls_back_to_default(client_factory):
    settings = _make_settings(
        model_map_raw="claude-sonnet-*:coder14b:latest",
        default_model="fallback-model:latest",
    )
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-opus-9-99",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured[0].captured_requests[0].model == "fallback-model:latest"
    finally:
        await client.aclose()


# --- Request translation edge cases -------------------------------------


@pytest.mark.asyncio
async def test_system_as_list_of_blocks_flattens(client_factory):
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "system": [
                    {"type": "text", "text": "You are a pirate."},
                    {"type": "text", "text": "Speak only in pirate-speak."},
                ],
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        msgs = captured[0].captured_requests[0].messages
        # First message must be the flattened system prompt.
        assert msgs[0] == {
            "role": "system",
            "content": "You are a pirate.\n\nSpeak only in pirate-speak.",
        }
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_tool_result_block_becomes_role_tool_message(client_factory):
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "Boston"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "Sunny, 72F",
                            }
                        ],
                    },
                ],
            },
        )
        assert resp.status_code == 200
        msgs = captured[0].captured_requests[0].messages
        # Find the {role: tool} message that the tool_result block should have become.
        tool_msgs = [m for m in msgs if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "toolu_1"
        assert tool_msgs[0]["content"] == "Sunny, 72F"
        # And the assistant message should carry the tool_calls.
        asst = [m for m in msgs if m.get("role") == "assistant"][0]
        assert asst["tool_calls"][0]["id"] == "toolu_1"
        assert asst["tool_calls"][0]["function"]["name"] == "get_weather"
    finally:
        await client.aclose()


# --- max_tokens floor bypass --------------------------------------------


@pytest.mark.asyncio
async def test_max_tokens_forwarded_not_floored(client_factory):
    """Anthropic path passes apply_max_tokens_floor=False to both methods."""
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured[0].captured_apply_floor == [False]
        # And the max_tokens=50 is forwarded as-is (not raised to any floor).
        assert captured[0].captured_requests[0].extra_params.get("max_tokens") == 50
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_streaming_max_tokens_not_floored(client_factory):
    """Streaming variant also bypasses the floor."""
    settings = _make_settings()
    client, captured = client_factory(settings, behavior="stream_text")
    try:
        resp = await client.post(
            "/v1/messages",
            headers={"x-api-key": SECRET, "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 50,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert resp.status_code == 200
        assert captured[0].captured_apply_floor == [False]
        assert captured[0].captured_requests[0].extra_params.get("max_tokens") == 50
    finally:
        await client.aclose()
