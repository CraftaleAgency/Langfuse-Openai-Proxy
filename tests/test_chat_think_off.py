"""Tests for the CHAT_THINK_OFF setting.

When enabled, generic /v1/chat/completions requests that don't specify `think`
are routed through Ollama's native /api/chat with think=false — the only endpoint
that honors it (the OpenAI-compat /v1 silently ignores `think`, letting reasoning
models ramble). Keeps non-shim clients concise without touching the Anthropic
shim, which already routes native by default.
"""

import asyncio

import pytest

from langfuse_openai_proxy.domain.models import ChatRequest, Credentials
from langfuse_openai_proxy.domain.services import TracingService

_CREDENTIALS = Credentials(public_key="pk", secret_key="sk")


class _NoopGen:
    def update(self, **kwargs):
        pass

    def end(self):
        pass


def _noop_langfuse_factory(*args, **kwargs):
    class _Lf:
        def start_observation(self, **kw):
            return _NoopGen()

        def flush(self):
            pass

    return _Lf()


class _GuardOpenAI:
    """If its create() is reached, the /v1 path was wrongly used — fail."""

    def __init__(self):
        class _Completions:
            async def create(inner, **kwargs):
                raise AssertionError(
                    "OpenAI /v1 path used; chat_think_off should route via /api/chat"
                )

        self.chat = type("_Chat", (), {"completions": _Completions()})


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeHttpClient:
    """Records native /api/chat POSTs instead of hitting the network."""

    def __init__(self):
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _FakeResponse(
            {"message": {"content": "ok"}, "done": True, "eval_count": 1, "prompt_eval_count": 1}
        )


def _service(*, chat_think_off):
    return TracingService(
        langfuse_client_factory=_noop_langfuse_factory,
        openai_client=_GuardOpenAI(),
        upstream_base_url="http://ollama:11434/v1",
        upstream_api_key="",
        chat_think_off=chat_think_off,
    )


@pytest.mark.asyncio
async def test_chat_think_off_routes_native_with_think_false(monkeypatch):
    """No-think request + chat_think_off → native /api/chat with think=false."""
    fake_http = _FakeHttpClient()
    monkeypatch.setattr("langfuse_openai_proxy.domain.services.get_http_client", lambda: fake_http)
    svc = _service(chat_think_off=True)

    request = ChatRequest(
        model="coder14b:latest",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_params={},
    )

    data = await svc.chat_completion(_CREDENTIALS, request, host="https://lf")
    # Let the background Langfuse flush (now fire-and-forget) settle.
    await asyncio.sleep(0)

    # Routed native — the /v1 guard was not triggered and /api/chat was hit once.
    assert len(fake_http.calls) == 1
    url, kwargs = fake_http.calls[0]
    assert url.endswith("/api/chat")
    # think=false was injected and forwarded in the native body.
    assert kwargs["json"]["think"] is False
    # Translated OpenAI-shape response came back.
    assert data["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_chat_think_off_respects_explicit_think(monkeypatch):
    """An explicit think=true is not clobbered to false by chat_think_off."""
    fake_http = _FakeHttpClient()
    monkeypatch.setattr("langfuse_openai_proxy.domain.services.get_http_client", lambda: fake_http)
    svc = _service(chat_think_off=True)

    request = ChatRequest(
        model="coder14b:latest",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        extra_params={"think": True},
    )

    await svc.chat_completion(_CREDENTIALS, request, host="https://lf")
    await asyncio.sleep(0)

    assert len(fake_http.calls) == 1
    # Caller's explicit think=True preserved (not forced to false).
    assert fake_http.calls[0][1]["json"]["think"] is True
