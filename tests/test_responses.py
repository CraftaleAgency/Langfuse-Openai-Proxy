"""Tests for the /v1/responses endpoint.

Unit tests verify auth enforcement and error handling (no backend needed).
Integration tests verify against a real deployed proxy with real LLM backend.
"""

import pytest

# === UNIT TESTS (no backend needed) ===


class TestResponsesAuthUnit:
    """Verify /v1/responses requires valid credentials (in-process)."""

    @pytest.mark.asyncio
    async def test_responses_requires_auth(self, client):
        r = await client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hello"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_responses_empty_bearer_rejected(self, client):
        r = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer "},
            json={"model": "gpt-4o", "input": "hello"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_responses_error_is_json(self, client):
        r = await client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hello"},
        )
        assert r.status_code == 401
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]

    @pytest.mark.asyncio
    async def test_responses_no_stack_trace_in_error(self, client):
        r = await client.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hello"},
        )
        assert "Traceback" not in r.text
        assert ".py" not in r.text


class TestResponsesRoutingUnit:
    """Verify /v1/responses route takes precedence over passthrough."""

    @pytest.mark.asyncio
    async def test_responses_not_caught_by_passthrough(self, client, auth_headers):
        """With auth, the response route should process (not passthrough)."""
        r = await client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"model": "test", "input": "hi"},
        )
        # Will fail connecting to upstream, but should NOT be a 401
        # (proving the specific route matched, not passthrough)
        assert r.status_code != 401


class TestResponsesInputValidationUnit:
    """Input validation edge cases for /v1/responses."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, client, auth_headers):
        r = await client.post("/v1/responses", headers=auth_headers, content=b"not json")
        assert r.status_code in (400, 422)

    @pytest.mark.asyncio
    async def test_null_bytes_in_body_no_crash(self, client, auth_headers):
        r = await client.post(
            "/v1/responses",
            headers=auth_headers,
            content=b'{"model":"gpt-4o","input":"hi\x00"}',
        )
        assert r.status_code in (200, 400, 422, 500)

    @pytest.mark.asyncio
    async def test_input_as_list(self, client, auth_headers):
        """Responses API accepts input as a list of message objects."""
        r = await client.post(
            "/v1/responses",
            headers=auth_headers,
            json={
                "model": "test",
                "input": [{"role": "user", "content": "hello"}],
            },
        )
        # Will fail connecting to upstream, but shouldn't crash on input parsing
        assert r.status_code != 401

    @pytest.mark.asyncio
    async def test_stream_flag_accepted(self, client, auth_headers):
        """The stream parameter should be accepted without crashing."""
        r = await client.post(
            "/v1/responses",
            headers=auth_headers,
            json={"model": "test", "input": "hi", "stream": True},
        )
        assert r.status_code != 401


# === INTEGRATION TESTS (needs real proxy + LLM backend) ===


@pytest.mark.integration
class TestResponsesIntegration:
    """Verify /v1/responses against real deployed proxy."""

    @pytest.mark.asyncio
    async def test_responses_without_auth_returns_401(self, proxy):
        r = await proxy.post(
            "/v1/responses",
            json={"model": "gpt-4o", "input": "hello"},
        )
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_response(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/responses",
            headers=integration_headers,
            json={"model": "gemma-4-E4B", "input": "say hello"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "output" in body

    @pytest.mark.asyncio
    async def test_streaming_response(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/responses",
            headers=integration_headers,
            json={"model": "gemma-4-E4B", "input": "say hi", "stream": True},
        )
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_response_with_extra_params(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/responses",
            headers=integration_headers,
            json={
                "model": "gemma-4-E4B",
                "input": "say ok",
                "temperature": 0.1,
                "max_output_tokens": 10,
            },
        )
        assert r.status_code == 200
