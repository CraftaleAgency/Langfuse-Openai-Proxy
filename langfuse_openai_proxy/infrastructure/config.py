"""Configuration from environment variables.

No framework imports — pure dataclass reading env vars.
"""

import os
from dataclasses import dataclass, field


def _parse_anthropic_model_map(raw: str) -> dict[str, str]:
    """Parse a `ANTHROPIC_MODEL_MAP` env value into a {pattern: physical_model} dict.

    Format: comma-separated `pattern:physical` pairs. Whitespace trimmed.
    Empty segments are skipped. Example:
        "claude-sonnet-*:coder14b:latest, claude-haiku-*:gemma4:12b"
    """
    out: dict[str, str] = {}
    if not raw:
        return out
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        pattern, physical = pair.split(":", 1)
        pattern = pattern.strip()
        physical = physical.strip()
        if pattern and physical:
            out[pattern] = physical
    return out


@dataclass(frozen=True)
class Settings:
    """Application settings from environment variables."""

    upstream_base_url: str = os.environ.get("UPSTREAM_BASE_URL", "http://localhost:4000/v1")
    upstream_api_key: str = os.environ.get("UPSTREAM_API_KEY", "")
    langfuse_default_host: str = os.environ.get(
        "LANGFUSE_DEFAULT_HOST", "https://cloud.langfuse.com"
    )
    # Default Langfuse project credentials. Existing /v1/chat/completions traffic
    # still takes pk/sk per-request from headers. The Anthropic shim is a trusted
    # single-client (Claude Code), so it traces using these env credentials and
    # only validates a single shared token for auth.
    langfuse_public_key: str = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.environ.get("LANGFUSE_SECRET_KEY", "")
    # Some upstreams (notably Ollama's /v1 endpoint serving reasoning models like
    # gemma4/qwen3) stream the model's output in the non-standard `delta.reasoning`
    # field, leaving `delta.content` empty. OpenAI-compatible clients that only read
    # `content` (e.g. OpenClaw's openai-completions adapter) then see an empty
    # response and abort with stop_reason=length. When enabled, the proxy copies
    # any `reasoning` text into `content` (keeping `reasoning` too) so every client
    # sees a normal content stream. Default off to preserve tracing fidelity for
    # clients that DO distinguish reasoning from content.
    reasoning_as_content: bool = os.environ.get("REASONING_AS_CONTENT", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Many OpenAI clients default to a small `max_tokens` (50 is common). Reasoning
    # models served via Ollama (qwen3, gemma4, thinker14b) burn ~100+ tokens on
    # `<think>...</think>` before any visible output emerges, so a 50-token budget
    # truncates thinking mid-stream and the client sees an empty response. When
    # set to a positive int, the proxy injects `max_tokens=floor` when the client
    # sends none, and raises `max_tokens` to the floor when the client sends less.
    # Leave unset (or 0) to disable — clients' own budgets are then honored.
    max_tokens_floor: int | None = (
        int(v) if (v := os.environ.get("MAX_TOKENS_FLOOR", "").strip()).isdigit() else None
    )
    # Anthropic /v1/messages shim. When True, the proxy mounts the
    # /v1/messages, /v1/messages/count_tokens, /v1/messages/{id}, and
    # /v1/models endpoints that emit Anthropic-shape responses. This is
    # what lets Claude Code (which only speaks the Anthropic API) talk to
    # OpenAI-compatible upstreams via this proxy. Default off — the
    # existing /v1/chat/completions route is unaffected either way.
    anthropic_shim_enabled: bool = os.environ.get("ANTHROPIC_SHIM_ENABLED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    # Comma-separated `pattern:physical` pairs mapping incoming Anthropic
    # model names (which may be globs using `*`) to upstream OpenAI model
    # names. Example: "claude-sonnet-*:coder14b:latest, claude-haiku-*:gemma4:12b".
    # Parsed into anthropic_model_map below.
    anthropic_model_map_raw: str = os.environ.get("ANTHROPIC_MODEL_MAP", "")
    # Parsed view of anthropic_model_map_raw. Frozen dataclass forces us to
    # use field(default_factory=...) with a separate parse call.
    anthropic_model_map: dict[str, str] = field(default_factory=dict)
    # Fallback physical model when an incoming Anthropic model name doesn't
    # match any pattern in ANTHROPIC_MODEL_MAP.
    anthropic_default_model: str = os.environ.get("ANTHROPIC_DEFAULT_MODEL", "coder14b:latest")
    # Hard killswitch. When True, the Anthropic routes always 404 regardless
    # of anthropic_shim_enabled. Useful for emergencies without redeploy.
    anthropic_paused: bool = os.environ.get("ANTHROPIC_PAUSED", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    def __post_init__(self) -> None:
        # frozen=True blocks normal assignment, so use object.__setattr__.
        if not self.anthropic_model_map:
            object.__setattr__(
                self, "anthropic_model_map", _parse_anthropic_model_map(self.anthropic_model_map_raw)
            )
