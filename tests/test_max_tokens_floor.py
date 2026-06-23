"""Tests for the `max_tokens` floor (MAX_TOKENS_FLOOR env var).

Many OpenAI clients default to a small max_tokens (50 is common). Reasoning
models served via Ollama (qwen3, gemma4, thinker14b) burn ~100+ tokens on
`<think>...</think>` before any visible output emerges, so a 50-token budget
truncates thinking mid-stream and the client sees an empty response.

When `MAX_TOKENS_FLOOR` is set to a positive int, the proxy:
  - injects `max_tokens=floor` when the client sends none
  - raises `max_tokens` to the floor when the client sends less
  - leaves the request alone when the client already sends >= floor
  - leaves the request alone when the floor is unset or non-positive
"""

import json

import pytest

from langfuse_openai_proxy.domain.services import TracingService, _apply_max_tokens_floor


def test_floor_none_no_op():
    """No floor set → extra_params pass through unchanged (missing max_tokens stays missing)."""
    out = _apply_max_tokens_floor({"temperature": 0.5}, None)
    assert out == {"temperature": 0.5}
    assert "max_tokens" not in out


def test_floor_zero_no_op():
    """Floor of 0 → disabled (treated like unset)."""
    out = _apply_max_tokens_floor({"max_tokens": 50}, 0)
    assert out == {"max_tokens": 50}


def test_floor_negative_no_op():
    """Negative floor → disabled."""
    out = _apply_max_tokens_floor({"max_tokens": 50}, -1)
    assert out == {"max_tokens": 50}


def test_floor_injects_when_missing():
    """No max_tokens on request → floor is injected."""
    out = _apply_max_tokens_floor({"temperature": 0.5}, 2000)
    assert out == {"temperature": 0.5, "max_tokens": 2000}


def test_floor_injects_when_no_extra_params():
    """No extra_params at all → floor still injected."""
    out = _apply_max_tokens_floor(None, 2000)
    assert out == {"max_tokens": 2000}


def test_floor_raises_small_max_tokens():
    """Client sends max_tokens=50, floor=2000 → raised to 2000."""
    out = _apply_max_tokens_floor({"max_tokens": 50, "temperature": 0.5}, 2000)
    assert out == {"max_tokens": 2000, "temperature": 0.5}


def test_floor_preserves_meets_floor():
    """Client sends exactly the floor → untouched."""
    out = _apply_max_tokens_floor({"max_tokens": 2000}, 2000)
    assert out == {"max_tokens": 2000}


def test_floor_preserves_above_floor():
    """Client sends more than the floor → untouched."""
    out = _apply_max_tokens_floor({"max_tokens": 8000}, 2000)
    assert out == {"max_tokens": 8000}


def test_floor_does_not_mutate_input():
    """Helper returns a new dict; caller's extra_params stays clean."""
    src = {"temperature": 0.5}
    out = _apply_max_tokens_floor(src, 2000)
    assert src == {"temperature": 0.5}
    assert out == {"temperature": 0.5, "max_tokens": 2000}


# --- End-to-end via TracingService ---
# Drives stream_chat_completion with a fake OpenAI client and verifies that
# max_tokens forwarded to upstream reflects the floor.


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class _CapturingOpenAI:
    """Captures the kwargs passed to chat.completions.create for assertion."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.captured_kwargs = None

        class _Completions:
            async def create(inner_self, **kwargs):
                self.captured_kwargs = kwargs
                assert kwargs.get("stream") is True
                return _FakeStream(list(self._chunks))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _make_service(max_tokens_floor, chunks):
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
        openai_client=_CapturingOpenAI(chunks),
        upstream_base_url="http://upstream/v1",
        upstream_api_key="",
        max_tokens_floor=max_tokens_floor,
    )


_CREDENTIALS = type("Creds", (), {"public_key": "pk", "secret_key": "sk"})()


def _request(extra_params):
    return type(
        "Req",
        (),
        {
            "model": "qwen3:14b",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_params": extra_params,
        },
    )()


def _make_chunk(content, finish_reason=None):
    payload = {
        "id": "chatcmpl-test",
        "choices": [
            {
                "delta": {
                    "content": content,
                    "reasoning": None,
                    "role": "assistant",
                    "function_call": None,
                    "refusal": None,
                    "tool_calls": None,
                },
                "finish_reason": finish_reason,
                "index": 0,
                "logprobs": None,
            }
        ],
    }

    class _Choice:
        pass

    class _Chunk:
        choices = [_Choice()]

        def model_dump_json(self):
            return json.dumps(payload)

    # Attach `.delta.content` so the production code path that collects native
    # content for tracing (`chunk.choices[0].delta.content`) doesn't blow up.
    _Choice.delta = type("D", (), {"content": payload["choices"][0]["delta"]["content"]})()

    return _Chunk()


@pytest.mark.asyncio
async def test_service_injects_floor_into_upstream_call():
    """End-to-end: client sends no max_tokens → upstream sees the floor."""
    chunks = [_make_chunk("answer", finish_reason="stop")]
    service = _make_service(max_tokens_floor=2000, chunks=chunks)
    fake_openai: _CapturingOpenAI = service._openai

    async for _ in service.stream_chat_completion(_CREDENTIALS, _request({}), "host"):
        pass

    assert fake_openai.captured_kwargs is not None
    # extra_body carries everything that wasn't a named arg in the SDK call
    assert fake_openai.captured_kwargs["extra_body"].get("max_tokens") == 2000


@pytest.mark.asyncio
async def test_service_raises_small_max_tokens_to_floor():
    """End-to-end: client sends max_tokens=50 → upstream sees 2000."""
    chunks = [_make_chunk("answer", finish_reason="stop")]
    service = _make_service(max_tokens_floor=2000, chunks=chunks)
    fake_openai: _CapturingOpenAI = service._openai

    async for _ in service.stream_chat_completion(
        _CREDENTIALS, _request({"max_tokens": 50}), "host"
    ):
        pass

    assert fake_openai.captured_kwargs["extra_body"].get("max_tokens") == 2000


@pytest.mark.asyncio
async def test_service_preserves_large_max_tokens():
    """End-to-end: client sends max_tokens=8000 → upstream sees 8000."""
    chunks = [_make_chunk("answer", finish_reason="stop")]
    service = _make_service(max_tokens_floor=2000, chunks=chunks)
    fake_openai: _CapturingOpenAI = service._openai

    async for _ in service.stream_chat_completion(
        _CREDENTIALS, _request({"max_tokens": 8000}), "host"
    ):
        pass

    assert fake_openai.captured_kwargs["extra_body"].get("max_tokens") == 8000


@pytest.mark.asyncio
async def test_service_floor_unset_preserves_client_budget():
    """End-to-end: floor unset → client's max_tokens=50 passes through."""
    chunks = [_make_chunk("answer", finish_reason="stop")]
    service = _make_service(max_tokens_floor=None, chunks=chunks)
    fake_openai: _CapturingOpenAI = service._openai

    async for _ in service.stream_chat_completion(
        _CREDENTIALS, _request({"max_tokens": 50}), "host"
    ):
        pass

    assert fake_openai.captured_kwargs["extra_body"].get("max_tokens") == 50
