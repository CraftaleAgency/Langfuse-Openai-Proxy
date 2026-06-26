"""End-to-end integration tests for the Anthropic /v1/messages shim.

These are the Anthropic-shim counterpart to tests/test_responses.py (the OpenAI
e2e suite). They hit a REAL deployed proxy (PROXY_URL) backed by a real LLM
(Ollama via the shim's alias→physical model map) and assert the full wire
contract that Claude Code depends on.

Unit coverage (FakeTracingService, no backend) lives in test_anthropic_messages.py.
This file covers the integration path only — the things that can only be verified
against a real round-trip:

  - Auth: the shared token gate (401 without / wrong), and that the real
    claude-local credential form (x-api-key: sk,pk) is accepted.
  - Non-streaming: valid Anthropic message shape + integer usage.
  - Streaming: text/event-stream content-type, well-formed event sequence
    (message_start → content_block_* → message_delta → message_stop), and that
    the text deltas reconstruct the model's answer.
  - Regressions for the three bugs that blocked claude-local:
      * non-zero message_delta usage        (else Claude Code discards streams)
      * output_tokens_details present        (thinking-token-count beta field)
      * stop_reason end_turn, not length      (think=false stops the rambling)
  - count_tokens endpoint returns a positive int.
  - Model alias is echoed back (not the physical Ollama name).

Skipped automatically when PROXY_URL or the LANGFUSE_* creds are unset
(see conftest.py and the module-level pytestmark below).
"""

import json
import os

import pytest

# Real Langfuse project creds from env (same source conftest's real_auth uses).
# The shim validates the secret half against LANGFUSE_SECRET_KEY configured on
# the deployed proxy, so these must match that deployment.
_SECRET = os.environ.get("LANGFUSE_SECRET_KEY", "")
_PUBLIC = os.environ.get("LANGFUSE_PUBLIC_KEY", "")

# A model alias the deployed ANTHROPIC_MODEL_MAP resolves (claude-sonnet-4-5 →
# coder14b:latest in docker-compose.yml). Falls back to ANTHROPIC_DEFAULT_MODEL
# if unmapped, so any deployment answers.
_ALIAS = "claude-sonnet-4-5"

# Mimic the headers Claude Code actually sends. The shim only reads x-api-key /
# Authorization, but including anthropic-version keeps the test faithful.
_ANTHROPIC_VERSION = {"anthropic-version": "2023-06-01"}

# conftest skips this whole suite when PROXY_URL is unset. Guard the other half
# too: if PROXY_URL is set but the LANGFUSE_* creds are not, the fixtures below
# would otherwise send malformed tokens (`Bearer |`, `x-api-key: ,`) and every
# happy-path test would fail with a misleading 401. Skip loudly instead.
pytestmark = pytest.mark.skipif(
    not _SECRET or not _PUBLIC,
    reason="LANGFUSE_SECRET_KEY/LANGFUSE_PUBLIC_KEY not set — skipping Anthropic e2e",
)


# --- fixtures ------------------------------------------------------------


@pytest.fixture
def bearer_headers():
    """Authorization: Bearer pk|sk — the OpenAI-style combined token.

    The shim's shared parser accepts this form too (order/separator insensitive),
    so the same header the OpenAI e2e tests use must work here.
    """
    return {
        "Authorization": f"Bearer {_PUBLIC}|{_SECRET}",
        "content-type": "application/json",
        **_ANTHROPIC_VERSION,
    }


@pytest.fixture
def xapikey_headers():
    """x-api-key: sk,pk — the exact form the claude-local alias sends.

    This is the real Claude Code auth path: a single comma-separated sk,pk pair
    in the x-api-key header. Verifying it end-to-end guards the order-insensitive
    parser + sk validation that gives per-client Langfuse traceability.
    """
    return {
        "x-api-key": f"{_SECRET},{_PUBLIC}",
        "content-type": "application/json",
        **_ANTHROPIC_VERSION,
    }


# --- helpers -------------------------------------------------------------


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into [(event_type, data_dict), ...].

    The shim emits classic `event: <t>\\ndata: <json>\\n\\n` blocks. Non-data
    lines (keepalives) are skipped. A `data:` line that fails to parse as JSON
    is a real regression (the translator only ever emits well-formed JSON), so
    we fail loud with the offending payload rather than silently dropping it —
    dropping would leave an empty/partial event list and surface later as a
    confusing IndexError instead of the actual cause.
    """
    events: list[tuple[str, dict]] = []
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
        if not (event_type and data_str):
            continue
        if data_str == "[DONE]":  # OpenAI sentinel; the Anthropic path never sends it.
            continue
        try:
            events.append((event_type, json.loads(data_str)))
        except json.JSONDecodeError:
            raise AssertionError(
                f"malformed SSE `data:` for event '{event_type}': {data_str!r}"
            ) from None
    return events


def _events_of(events: list[tuple[str, dict]], event_type: str) -> list[dict]:
    """All events of one type, asserting the type is present.

    Guards the common `[e for et, e in events if et == X][0]` pattern: on a
    truncated/corrupt stream the filtered list is empty, and indexing [0] would
    raise a misleading IndexError. This fails with the event types actually seen.
    """
    matched = [e for et, e in events if et == event_type]
    assert matched, f"no '{event_type}' event in stream; saw: {[et for et, _ in events]}"
    return matched


def _join_text(events: list[tuple[str, dict]]) -> str:
    """Reconcatenate text_delta payloads in arrival order."""
    return "".join(
        e["delta"]["text"]
        for et, e in events
        if et == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
    )


# === INTEGRATION TESTS (needs real proxy + LLM backend) ==================


@pytest.mark.integration
class TestAnthropicMessagesIntegration:
    """Verify /v1/messages against the real deployed proxy + LLM backend."""

    # --- auth gate -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_messages_without_auth_returns_401(self, proxy):
        r = await proxy.post(
            "/v1/messages",
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_messages_wrong_token_returns_401(self, proxy):
        r = await proxy.post(
            "/v1/messages",
            headers={"x-api-key": "sk-lf-not-a-real-key", "content-type": "application/json"},
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_x_api_key_combined_sk_comma_pk_accepted(self, proxy, xapikey_headers):
        """The real claude-local credential form (sk,pk via x-api-key) authenticates."""
        r = await proxy.post(
            "/v1/messages",
            headers=xapikey_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "say ok"}],
            },
        )
        assert r.status_code == 200

    # --- non-streaming ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_valid_non_streaming_message(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Reply with the single word: hello"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert isinstance(body["content"], list) and body["content"]
        # First block is text carrying a non-empty answer.
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"].strip()
        # Model echoed back is the alias the client asked for, not the Ollama name.
        assert body["model"] == _ALIAS

    @pytest.mark.asyncio
    async def test_non_streaming_usage_is_integer(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )
        assert r.status_code == 200
        usage = r.json()["usage"]
        assert isinstance(usage["input_tokens"], int)
        assert isinstance(usage["output_tokens"], int)
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_model_alias_echoed_not_physical(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "ok"}],
            },
        )
        assert r.status_code == 200
        assert r.json()["model"] == _ALIAS

    @pytest.mark.asyncio
    async def test_system_prompt_accepted(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 32,
                "system": "You are a helpful assistant. Always reply with: pong",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )
        assert r.status_code == 200
        assert r.json()["content"][0]["text"].strip()

    # --- streaming -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_streaming_content_type(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "say hi"}],
            },
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_streaming_event_sequence_well_formed(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Count from 1 to 3"}],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        assert events, "empty SSE stream"
        types = [et for et, _ in events]

        # Required envelope: starts with message_start, ends with message_stop.
        assert types[0] == "message_start"
        assert types[-1] == "message_stop"
        # Required inner sequence for a text answer.
        assert "content_block_start" in types
        assert "content_block_delta" in types
        assert "content_block_stop" in types
        assert "message_delta" in types
        # message_delta must follow the (last) content_block_stop.
        last_cbs = max(i for i, t in enumerate(types) if t == "content_block_stop")
        assert types.index("message_delta") > last_cbs
        # message_start carries the alias, not the physical model name.
        start = events[0][1]
        assert start["message"]["model"] == _ALIAS
        # Text deltas reconstruct to a non-empty answer.
        assert _join_text(events).strip()

    @pytest.mark.asyncio
    async def test_streaming_message_delta_usage_nonzero(self, proxy, bearer_headers):
        """Regression: message_delta.usage.output_tokens must be > 0.

        Claude Code discards streams whose final usage reports zero output
        tokens. With think=false the upstream reports real completion_tokens;
        if that wiring broke, this catches it.
        """
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "say hello world"}],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        md = _events_of(events, "message_delta")[0]
        assert md["usage"]["output_tokens"] > 0

    @pytest.mark.asyncio
    async def test_streaming_has_output_tokens_details(self, proxy, bearer_headers):
        """Regression: output_tokens_details.thinking_tokens must be present.

        Claude Code sends the thinking-token-count beta header and expects
        output_tokens_details in both message_start and message_delta usage —
        even when think=false (field present, value 0). Absent ⇒ response
        silently discarded as malformed.
        """
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "say ok"}],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)

        ms = _events_of(events, "message_start")[0]
        details_start = ms["message"]["usage"].get("output_tokens_details")
        assert details_start is not None
        assert "thinking_tokens" in details_start

        md = _events_of(events, "message_delta")[0]
        details_delta = md["usage"].get("output_tokens_details")
        assert details_delta is not None
        assert "thinking_tokens" in details_delta

    @pytest.mark.asyncio
    async def test_streaming_think_false_stops_with_end_turn(self, proxy, bearer_headers):
        """Regression: with think=false (default), the model ends cleanly — no rambling.

        A concise prompt under a generous budget must finish with stop_reason
        end_turn, NOT max_tokens (length). A length cutoff here means the
        thinking model is rambling unboundedly again — the 858cacb regression.
        """
        r = await proxy.post(
            "/v1/messages",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "max_tokens": 4096,
                "stream": True,
                "messages": [
                    {"role": "user", "content": "What is 2+2? Answer with just the number."}
                ],
            },
        )
        assert r.status_code == 200
        events = _parse_sse(r.text)
        md = _events_of(events, "message_delta")[0]
        # The regression lock: stop_reason is end_turn, NOT max_tokens. If the
        # thinking model rambled to the 4096 budget, finish_reason would be
        # "length" → stop_reason "max_tokens" and this fails. (We deliberately
        # do NOT assert on output length — a legit verbose answer is still fine
        # as long as the model stopped on its own.)
        assert md["delta"]["stop_reason"] == "end_turn"

    # --- count_tokens ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_count_tokens_returns_positive_int(self, proxy, bearer_headers):
        r = await proxy.post(
            "/v1/messages/count_tokens",
            headers=bearer_headers,
            json={
                "model": _ALIAS,
                "messages": [{"role": "user", "content": "Hello, world! How are you?"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["input_tokens"], int)
        assert body["input_tokens"] > 0
