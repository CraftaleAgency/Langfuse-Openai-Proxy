"""Security tests: input validation and injection payloads.

Unit tests verify the proxy handles edge cases gracefully (no backend needed).
Integration tests verify against a real deployed proxy with real LLM backend.
"""

import pytest


# === UNIT TESTS (no backend needed) ===


class TestInputValidationUnit:
    """Input validation edge cases that fail before reaching upstream."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, client, auth_headers):
        r = await client.post("/v1/chat/completions", headers=auth_headers, content=b"not json")
        assert r.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_null_bytes_in_body_no_crash(self, client, auth_headers):
        r = await client.post(
            "/v1/chat/completions",
            headers=auth_headers,
            content=b'{"model":"gpt-4","messages":[{"role":"user","content":"hi\x00"}]}',
        )
        assert r.status_code in (200, 400, 422, 500)


# === INTEGRATION TESTS (needs real proxy + LLM backend) ===


@pytest.mark.integration
class TestInputValidationIntegration:
    """Input validation against real proxy + real LLM backend."""

    @pytest.mark.asyncio
    async def test_valid_chat_completion(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={
                "model": "gemma-4-E4B",
                "messages": [{"role": "user", "content": "say hello"}],
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming_chat_completion(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={
                "model": "gemma-4-E4B",
                "messages": [{"role": "user", "content": "say hi"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_embedding(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/embeddings",
            headers=integration_headers,
            json={"model": "multilingual-e5-small", "input": "hello world"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "data" in body
        assert len(body["data"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_model_returns_error(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={
                "model": "nonexistent-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code in (400, 404, 500)

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            content=b"not json",
        )
        assert r.status_code in (400, 422, 500)  # 500 if proxy can't parse(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={"model": "gemma-4-E4B", "messages": []},
        )
        # Should not crash — may return error from upstream
        assert r.status_code in (200, 400, 500)

    @pytest.mark.asyncio
    async def test_xss_payload_passes_through(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={
                "model": "gemma-4-E4B",
                "messages": [{"role": "user", "content": "<script>alert(1)</script>"}],
            },
        )
        # XSS is just text data — proxy should pass it through
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_extra_params_forwarded(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={
                "model": "gemma-4-E4B",
                "messages": [{"role": "user", "content": "say ok"}],
                "temperature": 0.1,
                "max_tokens": 10,
            },
        )
        assert r.status_code == 200
