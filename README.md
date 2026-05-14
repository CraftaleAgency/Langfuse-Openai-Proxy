# Langfuse OpenAI Proxy

[![CI](https://github.com/CraftaleAgency/Langfuse-Openai-Proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/CraftaleAgency/Langfuse-Openai-Proxy/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)

An OpenAI-compatible proxy that adds **per-request Langfuse tracing** to any LLM API call. Drop it between your application and any OpenAI-compatible backend (LiteLLM, vLLM, Ollama, etc.) and get full observability with zero code changes.

## Features

- **Per-request project tracing** -- Each request carries its own Langfuse API key pair, routing traces to the correct project automatically
- **Two credential formats** -- Combined (`pk-lf-...|sk-lf-...` or `pk-lf-...,sk-lf-...`) or separate headers (`Authorization` + `X-Langfuse-Public-Key`)
- **Full OpenAI API compatibility** -- Chat completions, embeddings, model listing, and generic `/v1/*` passthrough
- **SSE streaming support** -- Streaming responses are traced with content collection
- **Clean layered architecture** -- API, Domain, and Infrastructure layers with clear separation
- **Docker-ready** -- Single-container deployment with built-in healthcheck
- **Zero config for basic usage** -- Sensible defaults, configure via environment variables
- **Langfuse v4 SDK** -- Uses `start_observation` API with proper `update()`/`end()` lifecycle

## Quick Start

### Docker (recommended)

```bash
docker run -p 8000:8000 \
  -e UPSTREAM_BASE_URL=http://your-llm-backend:4000/v1 \
  craftaleagency/langfuse-openai-proxy
```

### Docker Compose

```yaml
services:
  langfuse-proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - UPSTREAM_BASE_URL=http://litellm-proxy:4000/v1
      - LANGFUSE_DEFAULT_HOST=https://cloud.langfuse.com
```

### pip

```bash
pip install langfuse-openai-proxy
langfuse-openai-proxy
# or: python -m langfuse_openai_proxy
```

## Usage

Point any OpenAI SDK at the proxy and add Langfuse credentials:

### Combined format (single header)

Use `|` (pipe) or `,` (comma) to separate the keys:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer pk-lf-...|sk-lf-..." \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

> **Cloudflare / WAF note:** If your proxy sits behind Cloudflare, the `|` character may be blocked by WAF rules. Use `,` instead: `pk-lf-...,sk-lf-...`

### Query parameter (for tools with limited config)

If your tool only has API Base URL and API Key fields (e.g., Onyx), pass the public key as a query parameter:

- **API Base URL:** `http://localhost:8000/v1?langfuse_pk=pk-lf-...`
- **API Key:** `sk-lf-...`

### Separate headers

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-lf-..." \
  -H "X-Langfuse-Public-Key: pk-lf-..." \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="pk-lf-...|sk-lf-...",  # Combined format (use ',' behind Cloudflare)
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### Custom Langfuse host per request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer pk-lf-...|sk-lf-..." \
  -H "X-Langfuse-Host: https://your-langfuse-instance.com" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}'
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `UPSTREAM_BASE_URL` | `http://localhost:4000/v1` | OpenAI-compatible backend to proxy to |
| `UPSTREAM_API_KEY` | (empty) | API key for the upstream backend |
| `LANGFUSE_DEFAULT_HOST` | `https://cloud.langfuse.com` | Default Langfuse host (overridable per-request via `X-Langfuse-Host`) |

## Architecture

```
Request → [Credential Parsing] → [Create per-request Langfuse client]
        → [Start Langfuse observation]
        → [Proxy to upstream]
        → [Record result in Langfuse]
        → [Return response]
```

Three layers:
- **API** (`api/`) -- FastAPI routes, error handlers, dependency injection
- **Domain** (`domain/`) -- Models, errors, TracingService (business logic)
- **Infrastructure** (`infrastructure/`) -- Config, Langfuse client factory, OpenAI client factory

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List available models |
| GET | `/v1/models/{model}` | Get model info |
| POST | `/v1/chat/completions` | Chat completion (with tracing) |
| POST | `/v1/embeddings` | Embeddings (with tracing) |
| * | `/v1/{path}` | Generic passthrough |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check .
ruff format --check .

# Run locally
uvicorn main:app --reload
```

## License

Apache License 2.0 -- see [LICENSE](LICENSE).
