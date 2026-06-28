"""Tests for the reasoning → content remap (REASONING_AS_CONTENT).

Reasoning models served via Ollama's /v1 endpoint stream the model's output in
`delta.reasoning` with `delta.content` empty. Clients that only read `content`
(OpenClaw's openai-completions adapter) then see an empty stream. When
`reasoning_as_content` is enabled, the proxy copies `reasoning` into `content`.

These tests drive TracingService.stream_chat_completion directly with a fake
OpenAI client so no network/backend is required.
"""

import json

import pytest

from langfuse_openai_proxy.domain.services import TracingService


def _make_chunk(content: str | None, reasoning: str | None, finish_reason: str | None = None):
    """Build an object whose model_dump_json() mimics an OpenAI SDK stream chunk,
    including the non-standard `reasoning` field that Ollama emits."""
    delta = {
        "content": content,
        "reasoning": reasoning,
        "role": "assistant",
        "function_call": None,
        "refusal": None,
        "tool_calls": None,
    }

    # NB: assign in __init__, not the class body. Class bodies don't see the
    # enclosing function's locals, so `content = content` at class scope raises
    # NameError. __init__ is a real closure, so the param is visible. The
    # .content attribute is read by TracingService (chunk.choices[0].delta.content)
    # to collect output for tracing.
    class _Delta:
        def __init__(self):
            self.content = content

    class _Choice:
        delta = _Delta()

    class _Chunk:
        choices = [_Choice()]

        def model_dump_json(self) -> str:
            return json.dumps(
                {
                    "id": "chatcmpl-test",
                    "choices": [
                        {
                            "delta": delta,
                            "finish_reason": finish_reason,
                            "index": 0,
                            "logprobs": None,
                        }
                    ],
                }
            )

    return _Chunk()


class _FakeStream:
    """Async iterator over in-memory chunks."""

    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _FakeOpenAI:
    """Minimal AsyncOpenAI stand-in for chat.completions.create(stream=True)."""

    def __init__(self, chunks):
        self._chunks = chunks

        class _Completions:
            async def create(inner_self, **kwargs):
                assert kwargs.get("stream") is True
                return _FakeStream(list(self._chunks))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _make_service(reasoning_as_content: bool, chunks):
    """Build a TracingService with a no-op Langfuse factory and fake OpenAI client."""

    class _NoopGen:
        def update(self, **kwargs):
            pass

        def end(self):
            pass

    def langfuse_factory(public_key, secret_key, host):
        class _Lf:
            def start_observation(self, **kwargs):
                return _NoopGen()

            def flush(self):
                pass

        return _Lf()

    return TracingService(
        langfuse_client_factory=langfuse_factory,
        openai_client=_FakeOpenAI(chunks),
        upstream_base_url="http://upstream/v1",
        upstream_api_key="",
        reasoning_as_content=reasoning_as_content,
    )


_CREDENTIALS = type("Creds", (), {"public_key": "pk", "secret_key": "sk"})()
_REQUEST = type(
    "Req",
    (),
    {
        "model": "gemma4:12b",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_params": {},
    },
)()


def _parse_sse_events(generator):
    """Collect `data: {...}` payloads from the SSE generator (skips [DONE])."""
    payloads = []
    for line in generator:
        if line.startswith("data: ") and "[DONE]" not in line:
            payloads.append(json.loads(line[len("data: ") :]))
    return payloads


@pytest.mark.asyncio
async def test_remap_disabled_preserves_reasoning_only_stream():
    """With remap OFF, reasoning-only chunks pass through with empty content."""
    chunks = [
        _make_chunk(content="", reasoning="thinking"),
        _make_chunk(content="answer", reasoning=None, finish_reason="stop"),
    ]
    service = _make_service(reasoning_as_content=False, chunks=chunks)

    events = _parse_sse_events(
        [chunk async for chunk in service.stream_chat_completion(_CREDENTIALS, _REQUEST, "host")]
    )

    assert events[0]["choices"][0]["delta"]["content"] == ""  # untouched
    assert events[0]["choices"][0]["delta"]["reasoning"] == "thinking"
    assert events[1]["choices"][0]["delta"]["content"] == "answer"


@pytest.mark.asyncio
async def test_remap_enabled_copies_reasoning_into_content():
    """With remap ON, reasoning text appears in content for content-only clients."""
    chunks = [
        _make_chunk(content="", reasoning="thinking"),
        _make_chunk(content="answer", reasoning=None, finish_reason="stop"),
    ]
    service = _make_service(reasoning_as_content=True, chunks=chunks)

    events = _parse_sse_events(
        [chunk async for chunk in service.stream_chat_completion(_CREDENTIALS, _REQUEST, "host")]
    )

    # reasoning copied into content AND cleared, so reasoning-aware clients
    # don't receive the same token in both fields (which double-renders it).
    assert events[0]["choices"][0]["delta"]["content"] == "thinking"
    assert events[0]["choices"][0]["delta"]["reasoning"] == ""
    # native content chunk is NOT overwritten when content is already present
    assert events[1]["choices"][0]["delta"]["content"] == "answer"


@pytest.mark.asyncio
async def test_remap_does_not_overwrite_existing_content():
    """If a chunk already has content, reasoning must not clobber it."""
    chunks = [
        _make_chunk(content="real", reasoning="should-not-replace", finish_reason="stop"),
    ]
    service = _make_service(reasoning_as_content=True, chunks=chunks)

    events = _parse_sse_events(
        [chunk async for chunk in service.stream_chat_completion(_CREDENTIALS, _REQUEST, "host")]
    )

    assert events[0]["choices"][0]["delta"]["content"] == "real"
