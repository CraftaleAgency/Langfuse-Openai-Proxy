# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run the proxy locally
python -m langfuse_openai_proxy
# or: uvicorn main:app --reload

# Run all tests
pytest -v

# Run a single test file
pytest tests/test_credentials.py -v

# Run only unit tests (skip integration)
pytest -v -m "not integration"

# Integration tests (require PROXY_URL env var, auto-skipped otherwise)
pytest -v -m integration

# Lint
ruff check .

# Format
ruff format .

# Check formatting without changes
ruff format --check .
```

## Architecture

Three-layer clean architecture in `langfuse_openai_proxy/`:

**API layer** (`api/`) — FastAPI routes, dependency injection, error-to-HTTP mapping. `app.py` is the factory, `routes.py` has all endpoints, `dependencies.py` wires DI, `error_handlers.py` maps domain errors to HTTP responses.

**Domain layer** (`domain/`) — Pure business logic with no framework imports. `services.py` contains `TracingService` (core logic). `models.py` has dataclasses for `Credentials`, `ChatRequest`, `EmbeddingRequest`. `errors.py` defines a framework-agnostic error hierarchy.

**Infrastructure layer** (`infrastructure/`) — External integrations. `config.py` reads env vars into a Settings dataclass. `langfuse_client.py` and `openai_client.py` are client factories. `host_validation.py` handles SSRF prevention via allowlisting.

### Request Flow

```
Request → Parse Credentials → Validate Host → Create Langfuse Client
        → Start Observation → Proxy to Upstream → Record Result
        → Flush Langfuse → Return Response
```

Credentials are passed per-request (not global config) via two formats:
- Combined: `Authorization: Bearer pk-lf-...|sk-lf-...`
- Separate: `Authorization: Bearer sk-lf-...` + `X-Langfuse-Public-Key: pk-lf-...`

## Key Configuration

- `UPSTREAM_BASE_URL` — OpenAI-compatible backend (default: `http://localhost:4000/v1`)
- `UPSTREAM_API_KEY` — API key for upstream (optional)
- `LANGFUSE_DEFAULT_HOST` — Default Langfuse host (default: `https://cloud.langfuse.com`)

## Code Style

- Ruff with line length 100, rules: E, W, F, I, UP, B, SIM
- Python 3.11+ (uses `X | Y` union syntax)
- FastAPI `Depends()` for constructor injection (B008 is intentionally ignored)

## Testing

Unit tests use an in-process ASGI client (no network). Fixtures are in `conftest.py` — `app` (fresh FastAPI instance), `client` (async test client), `auth_headers` (valid credentials). Security tests live in `tests/security/` covering auth enforcement, input validation, and path traversal prevention. Integration tests require `PROXY_URL` and are marked `@pytest.mark.integration`.
