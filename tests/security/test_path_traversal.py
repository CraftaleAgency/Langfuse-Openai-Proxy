"""Security tests: path traversal and SSRF via X-Langfuse-Host.

Unit tests verify rejection logic in-process (no network needed).
Integration tests verify against a real deployed proxy.

Note: Path traversal unit tests can't use httpx because it normalizes URLs
before sending (e.g., /v1/../../etc becomes /etc). Path traversal is
tested in integration tests where raw HTTP is used.
"""

import pytest


# === UNIT TESTS ===


class TestHostValidationUnit:
    """X-Langfuse-Host SSRF prevention (in-process, no network)."""

    @pytest.mark.asyncio
    async def test_evil_host_rejected(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://evil.com"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400
        assert "not in the allowed list" in r.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_private_ip_rejected(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://192.168.1.1"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400
        assert "IP" in r.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_http_scheme_rejected(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "http://cloud.langfuse.com"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400
        assert "HTTPS" in r.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_aws_metadata_rejected(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://169.254.169.254"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_subdomain_bypass_rejected(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://cloudlangfuse.com.evil.com"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_embedding_host_also_validated(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://evil.com"}
        r = await client.post(
            "/v1/embeddings",
            headers=headers,
            json={"model": "text-embedding-3-small", "input": "hello"},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_host_error_no_stack_trace(self, client, auth_headers):
        headers = {**auth_headers, "X-Langfuse-Host": "https://evil.com"}
        r = await client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert "Traceback" not in r.text
        assert "/app/" not in r.text
        assert "/home/" not in r.text


# === INTEGRATION TESTS ===


@pytest.mark.integration
class TestPathTraversalIntegration:
    """Path traversal tests against real proxy (raw HTTP, no normalization)."""

    @pytest.mark.asyncio
    async def test_double_dot_rejected(self, proxy, integration_headers):
        r = await proxy.request("GET", "/v1/../../etc/passwd", headers=integration_headers)
        # Real HTTP server should either reject or return 400
        assert r.status_code in (400, 404)

    @pytest.mark.asyncio
    async def test_double_slash_rejected(self, proxy, integration_headers):
        r = await proxy.request("GET", "/v1//admin", headers=integration_headers)
        assert r.status_code in (400, 404)


@pytest.mark.integration
class TestSSRFIntegration:
    """SSRF protections against real proxy."""

    @pytest.mark.asyncio
    async def test_evil_host_rejected(self, proxy, integration_headers):
        headers = {**integration_headers, "X-Langfuse-Host": "https://evil.com"}
        r = await proxy.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gemma-4-E4B", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400
        assert "not in the allowed list" in r.json()["error"]["message"]

    @pytest.mark.asyncio
    async def test_private_ip_rejected(self, proxy, integration_headers):
        headers = {**integration_headers, "X-Langfuse-Host": "https://192.168.1.1"}
        r = await proxy.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gemma-4-E4B", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_aws_metadata_rejected(self, proxy, integration_headers):
        headers = {**integration_headers, "X-Langfuse-Host": "https://169.254.169.254"}
        r = await proxy.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gemma-4-E4B", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_host_works(self, proxy, integration_headers):
        headers = {**integration_headers, "X-Langfuse-Host": "https://cloud.langfuse.com"}
        r = await proxy.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "gemma-4-E4B", "messages": [{"role": "user", "content": "say ok"}]},
        )
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_default_host_works(self, proxy, integration_headers):
        r = await proxy.post(
            "/v1/chat/completions",
            headers=integration_headers,
            json={"model": "gemma-4-E4B", "messages": [{"role": "user", "content": "say ok"}]},
        )
        assert r.status_code == 200
