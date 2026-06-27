# Claude Code Wire Contract — Anthropic Shim Reference

This is the durable reference for the wire contract Claude Code (the Anthropic CLI
client) expects from this proxy's Anthropic shim, what the shim currently emits,
and where the two diverge. The request path is `POST /v1/messages`. The motivating
client is **claude-local** — a Claude Code instance pointed at this proxy — so
every behavior below is specified against what that client actually sends and
validates. Audience: future maintainers of the proxy. Everything here is concrete
(field names, event names, file:line anchors); nothing is aspirational.

---

## 1. Request contract (Claude Code → proxy)

### Transport

- `POST /v1/messages` — the only chat path Claude Code uses.
- **Always streaming.** Claude Code sets `stream: true` on every turn. The
  non-streaming branch exists for completeness and for `count_tokens`-style
  probes, not for real traffic.
- `maxRetries: 0` — the SDK does **not** retry. A single bad frame aborts the
  turn; surface errors cleanly (see section 4).
- SDK timeout **~300000 ms (~300 s)** per request. The proxy's own upstream
  timeout is **600 s** (`services.py:520`, `services.py:566`), so on a slow turn
  **Claude Code preempts first** — the proxy never sees the cancellation as an
  error, just a dropped SSE connection.

### Headers

| Header | Notes |
|---|---|
| `x-api-key` **OR** `Authorization: Bearer …` | Never both. `_extract_anthropic_token` (`anthropic_routes.py:42`) accepts either; `x-api-key` wins if both are present. |
| `anthropic-version: 2023-06-01` | Sent but not validated by the shim. |
| `anthropic-beta: …` | A long comma-separated beta list. The shim does not parse it; relevant betas are mirrored in the request body as `betas: [...]`. |
| `user-agent` | e.g. `claude-cli/x.y.z`. Logged when `ANTHROPIC_SHIM_DEBUG=1`. |

### Body fields

| Field | Type | Notes |
|---|---|---|
| `model` | string | Anthropic alias (e.g. `claude-sonnet-4-5-20250929`). Mapped to a physical model via `ANTHROPIC_MODEL_MAP` (`anthropic_routes.py:108`). |
| `max_tokens` | int | Required by Anthropic spec. Range observed: **32k–128k** depending on model. The shim forwards as-is; no floor is applied (`anthropic_routes.py:299`, `apply_max_tokens_floor=False`). |
| `system` | **array** of `{type:"text", text:"…", cache_control:{type:"ephemeral", ttl:"5m"}}` blocks | **NOT a string.** `_flatten_system` (`anthropic_translator.py:52`) joins `text` fields into one OpenAI system message; `cache_control` is dropped (no real prompt cache upstream). |
| `messages` | array | Standard `{role, content}` where `content` is a string or a list of `{type, …}` blocks (`text`, `tool_use`, `tool_result`). |
| `tools` | array of `{name, description, input_schema, type:"function"}` | **200–500+ entries** in real Claude Code sessions. Each is translated to an OpenAI function tool (`anthropic_translator.py:201`) and then sanitized for Ollama's grammar (`services.py:93`). |
| `tool_choice` | `{type:"auto"|"any"|"tool", name?}` | Mapped at `anthropic_translator.py:216`. `auto`→`"auto"`, `any`→`"required"`, `tool`→`{type:"function",function:{name}}`. |
| `temperature` | number | Present sometimes; forwarded verbatim. |
| `thinking` | `{type:"enabled"|"disabled"|"adaptive", budget_tokens}` | See gap matrix. The shim **forces thinking off** by default (`ANTHROPIC_SHIM_THINK=false`). |
| `betas` | array of strings | Forwarded unmodified into `extra_body`. |
| `metadata` | `{user_id: "…"}` | Currently ignored. |

### Continuity rules the client enforces

- The **last user message** must contain **only** `tool_result` blocks (no text)
  immediately after an assistant `tool_use` turn. The translator fans these out
  into separate `{role:"tool", tool_call_id, content}` messages
  (`anthropic_translator.py:101`).
- Each `tool_result.tool_use_id` must be a non-empty string matching the prior
  assistant `tool_use.id`. Empty/missing ids produce a client-side validation
  error before the request reaches the model.

---

## 2. Response / streaming contract (proxy → Claude Code)

### SSE event sequence

```
message_start
└─ content_block_start      (per block: text / tool_use / thinking)
│   └─ content_block_delta  (one or more)
│   └─ content_block_stop
… (repeat per block)
message_delta
message_stop
```

With periodic `ping` frames during upstream silence (every 15 s by default,
`anthropic_translator.py:358`).

### Per-event validation rules (Claude Code enforces these)

| Event | Required fields |
|---|---|
| `message_start` | `message.id` (string) **and** `message.usage` (object with `input_tokens`). Emitted lazily on first content chunk (`anthropic_translator.py:378`). |
| `content_block_start` (text) | `index`, `content_block.type:"text"`, `content_block.text:""`. |
| `content_block_start` (tool_use) | `index`, `content_block.type:"tool_use"`, `content_block.id` (non-empty), `content_block.name`. |
| `content_block_start` (thinking) | `index`, `content_block.type:"thinking"`, `content_block.thinking:""`. **No `signature` field on start** — see note below. |
| `content_block_delta` | One of: `text_delta` (`text`), `input_json_delta` (`partial_json` string), `thinking_delta` (`thinking`), `signature_delta` (`signature`). |
| `content_block_stop` | `index`. |
| `message_delta` | `delta.stop_reason`, `usage.output_tokens`, `usage.output_tokens_details.thinking_tokens`. |
| `message_stop` | (none beyond `type`). |
| `ping` | `type:"ping"`. |
| `error` | `type:"error"`, `error.type`, `error.message`. |

Cache token accounting: the client reads `cache_creation_input_tokens` and
`cache_read_input_tokens` from **both** `message_start` and `message_delta`. The
shim always emits both as `0` (no real prompt cache upstream).

**Thinking-block ordering invariant:** all `thinking` blocks must precede all
`text`/`tool_use` blocks. A thinking block is closed with a `signature_delta`
followed by `content_block_stop` before any subsequent block opens
(`anthropic_translator.py:405`, `anthropic_translator.py:666`). Including a
`signature` field on `content_block_start` makes the strict TS SDK reject the
block — the shim deliberately omits it (`anthropic_translator.py:470`).

### Two client validators

1. **Non-streaming `_5i` validator** — requires `content` (non-empty array) +
   `model` (string) + `usage` (object). Failure throws
   *"API returned an empty or malformed response — check for a proxy
   intercepting"*. The shim's empty-content fallback (`"(no output)"` text block)
   exists to defeat this (`anthropic_translator.py:293`).
2. **Streaming per-event validation** — the rules in the table above, applied to
   each SSE frame as it arrives.

---

## 3. Error contract

### Envelope shape (non-streaming)

```json
{"type":"error","error":{"type":"<t>","message":"<m>"}}
```

Both the top-level `type` and the nested `error.type` drive the Anthropic SDK's
error-class selection. A bare `{"error":{"message":…}}` makes Claude Code treat
the failure as an unparseable "proxy intercepting" crash.

### Status → type mapping

| status | type |
|---|---|
| 401 | `authentication_error` |
| 403 | `permission_error` |
| 404 | `not_found_error` |
| 413 | `request_too_large` |
| 429 | `rate_limit_error` |
| 5xx | `overloaded_error` |
| other 4xx | `invalid_request_error` |

Implemented in `error_handlers.py:21` (`_error_type_for_status`).

### Streaming errors

A streaming error is an **`event: error` SSE frame followed by connection
close** — **not** a bare `data: {"error":…}` plus `[DONE]`. The bare form makes
the client loop on retries. The shim emits the proper frame in three places:
the translator's upstream-error branch (`anthropic_translator.py:543`), the
stream wrapper's exception handler (`anthropic_routes.py:253`), and the native
stream path's 4xx/5xx/transport branches (`services.py:593`, `services.py:706`).

---

## 4. What this proxy emits today

### `langfuse_openai_proxy/api/anthropic_routes.py`

Mounts `/v1/messages`, `/v1/messages/count_tokens`, `/v1/messages/{id}`.
`_resolve_anthropic_credentials` (`:52`) validates the caller's single token
against `LANGFUSE_SECRET_KEY` and resolves Langfuse pk/sk for tracing — fail
closed on missing config (`:68`) or bad token (`:70`, `:77`). `_resolve_physical_model`
(`:108`) maps the Anthropic alias via `ANTHROPIC_MODEL_MAP` (exact match → glob
→ `ANTHROPIC_DEFAULT_MODEL`). The streaming generator `anthropic_event_stream`
(`:224`) wires `TracingService.stream_chat_completion` through the translator
and wraps the whole thing in a `try/except` that emits a clean `event: error`
frame on any post-header exception (`:253`). Thinking control: when
`ANTHROPIC_SHIM_THINK` is false (default), the request is pinned `think=false`
(`:199`) so `TracingService` routes through native `/api/chat`. The
`/v1/messages/{id}` route is a stub returning 404 (`:325`).

### `langfuse_openai_proxy/domain/anthropic_translator.py`

`anthropic_to_openai` (`:135`) flattens the Anthropic system array, fans
`tool_result` blocks into `{role:"tool"}` messages, and rewrites tools/tool_choice
into OpenAI shape. `openai_to_anthropic_response` (`:245`) rebuilds the
non-streaming Anthropic envelope; it emits a `thinking` block (with synthetic
signature, `:263`) when upstream produced `reasoning`, surfaces `tool_calls`
(`:275`), and falls back to a `"(no output)"` text block when content is empty
(`:293`). `openai_to_anthropic_stream` (`:354`) emits the full SSE sequence with
lazy `message_start`, per-block open/close bookkeeping (`_StreamState` `:322`),
heartbeat `ping` events, `output_tokens_details.thinking_tokens` in
`message_delta` (`:703`), and a `"(no output)"` safeguard for empty streams
(`:651`). Upstream error chunks are converted to a proper `event: error` frame
(`:543`). The upstream read uses `asyncio.wait` (not `wait_for`) so ping
heartbeats during Ollama's long prompt-eval silence don't cancel the in-flight
httpx read (`:524`). Cache token fields (`cache_creation_input_tokens`,
`cache_read_input_tokens`) are always `0`. Non-streaming responses carry an
OpenAI-shaped `id` (`chatcmpl-…`) — harmless, since Claude Code only checks
`model` + `content` + `usage`.

### `langfuse_openai_proxy/domain/services.py`

`_ollama_native_stream` (`:548`) is the native `/api/chat` streaming path used
when `think` is set. On upstream `>= 400` it reads the error body **before** the
stream context closes, logs it, dumps the rejected tools payload to
`/tmp/ollama_bad_tools_*.json` for offline debugging (`:588`), and emits a
data-frame error with `_anthropic_err_type` (`:593`); `TransportError` produces
an `overloaded_error` frame (`:706`). `_ollama_native_to_openai` (`:224`)
surfaces `message.tool_calls` in OpenAI shape so the translator can emit
`tool_use` blocks. `_sanitize_json_schema` (`:93`) and
`_sanitize_tool_for_ollama` (`:151`) coerce each tool's parameter schema into
the subset `llama.cpp`'s `json_schema_to_grammar` accepts — dropping `$ref`,
`$defs`, object-valued `additionalProperties`, unions, etc. The upstream timeout
is **600 s** (`:520`, `:566`).

### `langfuse_openai_proxy/api/error_handlers.py`

`anthropic_error_response` (`:38`) builds the Anthropic-shaped `JSONResponse`.
`_error_type_for_status` (`:21`) implements the status→type table. Handlers are
registered for `MissingCredentialsError` (401), `UpstreamError` (preserves the
upstream status), `ProxyError` (400), `ValueError` (400), and `HTTPException`
(scopes to the shim's auth/gate/misconfig raises plus any Starlette 404/405).

### Known-current behaviors (also in gap matrix)

- **Thinking is forced off** via `ANTHROPIC_SHIM_THINK=false` (default). Native
  `/api/chat` with `think=false` runs; gemma/qwen-class models still think by
  default regardless of the flag.
- **Cache token fields are always 0** — no real prompt cache upstream; cosmetic
  only.
- **Non-streaming `id` is OpenAI-shaped** — `chatcmpl-ollama-…`. Claude Code
  does not validate this field.

---

## 5. Gap matrix

| Contract requirement | Proxy behavior | Severity | Status |
|---|---|---|---|
| Anthropic-shaped error envelope (non-stream) | `error_handlers.py` emits `{"type":"error","error":{type,message}}` for all paths | High | **Fixed** |
| Streaming `event: error` frame on upstream failure | Translator (`anthropic_translator.py:543`) + stream wrapper (`anthropic_routes.py:253`) + native path (`services.py:593`,`:706`) | High | **Fixed** |
| Empty-output safeguard (no silent blanks) | Translator emits `"(no output)"` text block in both stream (`:651`) and non-stream (`:293`) paths | Med | **Fixed** |
| Tool-schema sanitizer for Ollama grammar | `_sanitize_json_schema` (`services.py:93`) drops unsupported JSON-Schema keywords per tool | High | **Fixed** |
| Tool-call surfacing on native `/api/chat` | `_ollama_native_to_openai` (`:224`) and the native streamer (`:638`) emit OpenAI `tool_calls` | High | **Fixed** |
| `thinking` field passthrough | Forced off via `ANTHROPIC_SHIM_THINK=false` default; gemma thinks by default regardless. Revisit if Claude Code sends `thinking:enabled` and quality suffers | Med | **Documented** |
| Prompt-cache token accounting | `cache_creation_input_tokens` / `cache_read_input_tokens` always `0`; no real cache upstream | Low | **Out-of-scope** (cosmetic) |
| Conversation-quality ceiling | Bound by 4–12B open models; see `MODELS.md` tiering | Med | **Capability-bound** (not a proxy bug) |
| Cold-load latency | Controlled by `OLLAMA_KEEP_ALIVE` + `num_ctx`; deployment concern, not proxy code | Med | **Infra** |

---

## 6. Deployment / config notes

Environment variables that control the shim (read in
`infrastructure/config.py`):

| Var | Purpose |
|---|---|
| `ANTHROPIC_SHIM_ENABLED` | Mount the `/v1/messages*` routes at all. Default off. |
| `ANTHROPIC_PAUSED` | Hard killswitch — routes 404 regardless of `ENABLED`. For emergencies without redeploy. |
| `ANTHROPIC_MODEL_MAP` | Comma-separated `pattern:physical` pairs, e.g. `claude-opus-*:hf.co/.../gemma-4-12B-agentic-…-tau2-GGUF:Q4_K_M`. Glob via trailing `*`; exact match wins. |
| `ANTHROPIC_DEFAULT_MODEL` | Fallback when no map entry matches. Default `coder14b:latest`. |
| `ANTHROPIC_SHIM_THINK` | Emit thinking blocks (route via `/v1`)? Default **false** → native `/api/chat` with `think=false`. Set true to restore thinking blocks (rambling/timeout risk). |
| `ANTHROPIC_SHIM_DEBUG` | Log per-request model/tool/usage summaries and per-response block/stop/usage. |
| `LANGFUSE_SECRET_KEY` | The shared token Claude Code presents; shim fails closed if unset. |
| `LANGFUSE_PUBLIC_KEY` | Public key used for tracing when the token doesn't carry one. |

Two operational gotchas:

- **`OLLAMA_NUM_CTX` env is ignored.** Ollama reads `num_ctx` from the
  Modelfile, not the environment. Tier models need `num_ctx 32768` (or higher)
  in the Modelfile or claude-local 400s on large tool-heavy prompts.
- **Timeout asymmetry.** Proxy upstream timeout is **600 s**; the Claude Code
  SDK timeout is **~300 s**. On a slow turn the client preempts first — the
  proxy sees a dropped SSE connection, not a translator error. Lengthen
  `OLLAMA_KEEP_ALIVE` (avoid cold load) and keep `num_ctx` sized for the
  prompt, not the SDK timeout.
