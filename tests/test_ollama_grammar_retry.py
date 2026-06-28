"""Tests for the Ollama transient-grammar-400 retry.

Ollama (llama.cpp) intermittently 400s a tool-bearing /api/chat with a
malformed-grammar message even though the schema is valid (a grammar-cache
bug). The same payload succeeds ~100% on retry, so the proxy transparently
retries only this specific error. These tests lock in the detector and the
non-streaming retry loop.
"""

from __future__ import annotations

import httpx
import pytest

from langfuse_openai_proxy.domain import services
from langfuse_openai_proxy.domain.errors import UpstreamError
from langfuse_openai_proxy.domain.models import ChatRequest

BRACE = '{"error":"Value looks like object, but can\'t find closing \'}\' symbol"}'


# --- detector ---------------------------------------------------------------


def test_detector_matches_brace_error_variants():
    assert services._is_transient_grammar_400(400, BRACE)
    assert services._is_transient_grammar_400(400, "can't find closing '}' symbol")
    assert services._is_transient_grammar_400(400, "NOT A VALID GRAMMAR")


def test_detector_rejects_non_grammar_errors():
    # Not a 400 at all.
    assert not services._is_transient_grammar_400(500, BRACE)
    assert not services._is_transient_grammar_400(429, "slow down")
    # A 400 that is a genuine schema/model error — must NOT be retried.
    assert not services._is_transient_grammar_400(400, "model 'foo' not found")
    assert not services._is_transient_grammar_400(400, "")


# --- non-streaming retry loop ----------------------------------------------


class _FakeResp:
    def __init__(self, status: int, text: str = "", payload: dict | None = None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "err",
                request=httpx.Request("POST", "http://ollama/api/chat"),
                response=httpx.Response(self.status_code, text=self.text),
            )

    def json(self):
        return self._payload


class _FakeHttp:
    """Returns a scripted sequence of responses from .post()."""

    def __init__(self, responses: list[_FakeResp]):
        self._responses = list(responses)
        self.post_calls = 0

    async def post(self, *args, **kwargs):
        r = self._responses[self.post_calls]
        self.post_calls += 1
        return r


def _service() -> services.TracingService:
    return services.TracingService(
        langfuse_client_factory=lambda *a, **k: None,
        openai_client=None,
        upstream_base_url="http://ollama:11434/v1",
        upstream_api_key="",
    )


def _request() -> ChatRequest:
    return ChatRequest(
        model="gemma:latest",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_params={"think": False},
    )


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_400s(monkeypatch):
    """Two transient grammar 400s, then a 200 — should return the 200 payload."""
    ok_payload = {
        "message": {"role": "assistant", "content": "hi"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 2,
    }
    fake = _FakeHttp(
        [_FakeResp(400, BRACE), _FakeResp(400, BRACE), _FakeResp(200, payload=ok_payload)]
    )
    monkeypatch.setattr(services, "get_http_client", lambda: fake)

    data = await _service()._ollama_native_chat(_request())

    assert fake.post_calls == 3  # retried twice, succeeded on the third
    assert data["choices"][0]["message"]["content"] == "hi"


@pytest.mark.asyncio
async def test_non_transient_400_not_retried(monkeypatch):
    """A genuine (non-grammar) 400 surfaces immediately, no retry."""
    fake = _FakeHttp([_FakeResp(400, "model 'nope' not found")])
    monkeypatch.setattr(services, "get_http_client", lambda: fake)

    with pytest.raises(UpstreamError):
        await _service()._ollama_native_chat(_request())

    assert fake.post_calls == 1  # never retried


@pytest.mark.asyncio
async def test_transient_400_exhausts_retries_then_surfaces(monkeypatch):
    """If every attempt is the transient error, it surfaces after the max retries."""
    fake = _FakeHttp([_FakeResp(400, BRACE)] * services._OLLAMA_GRAMMAR_RETRIES)
    monkeypatch.setattr(services, "get_http_client", lambda: fake)

    with pytest.raises(UpstreamError):
        await _service()._ollama_native_chat(_request())

    assert fake.post_calls == services._OLLAMA_GRAMMAR_RETRIES
