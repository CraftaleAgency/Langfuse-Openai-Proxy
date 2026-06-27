"""Anthropic Messages API ↔ OpenAI Chat Completions API translator.

Pure functions — no I/O, no framework imports. The API layer calls these
to convert between the Anthropic wire format (what Claude Code speaks) and
the OpenAI wire format (what this proxy's TracingService already serves).

See `/tmp/claude-shim-plan.md` for the full translation matrix and SSE
event-sequence rationale.
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------


def _sse(event_type: str, data: dict[str, Any]) -> str:
    """Format a single Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _toolu_id(index: int) -> str:
    return f"toolu_{int(time.time() * 1000)}_{index}"


def _msg_id() -> str:
    return f"msg_{int(time.time() * 1000)}"


def _thinking_signature(message_id: str, block_index: int) -> str:
    """Synthesize an opaque signature for a thinking block.

    Anthropic's real API signs thinking blocks so clients can verify provenance
    when replaying them in later turns. We're translating from Ollama which
    doesn't sign anything, so we emit a deterministic synthetic signature.
    Claude Code requires the field to be present but does not validate it
    against Anthropic-issued keys for third-party endpoints.
    """
    return f"{message_id}.tik.{block_index}"


# ---------------------------------------------------------------------------
# Request translation: Anthropic → OpenAI
# ---------------------------------------------------------------------------


def _flatten_system(system: Any) -> str:
    """Anthropic `system` field (str or list of text blocks) → plain string."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n\n".join(parts)
    return ""


def _content_blocks_to_openai(
    blocks: list[dict],
    role: str,
) -> tuple[Any, list[dict], list[dict]]:
    """Translate Anthropic content blocks for one message.

    Returns (content, tool_calls, tool_messages):
      - content: OpenAI message content (str, None, or multipart list)
      - tool_calls: OpenAI tool_calls list (for assistant messages with tool_use)
      - tool_messages: separate {role: tool} messages (for user messages with tool_result)
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_messages: list[dict] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif btype == "tool_use" and role == "assistant":
            tool_calls.append(
                {
                    "id": block.get("id", _toolu_id(len(tool_calls))),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif btype == "tool_result":
            # tool_result blocks appear in user messages; OpenAI models expect
            # each result as a separate {role: tool} message keyed by tool_call_id.
            content = block.get("content", "")
            if isinstance(content, list):
                # Multipart tool result — flatten text parts.
                pieces = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                content = "\n".join(pieces)
            elif not isinstance(content, str):
                content = json.dumps(content)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": content,
                }
            )

    if role == "assistant":
        content = "\n".join(text_parts) if text_parts else ""
        # OpenAI wants content: null (not "") when only tool_calls are present.
        if not content and tool_calls:
            content = None
        return content, tool_calls, tool_messages

    # User message: text becomes content, tool_results become separate messages.
    content = "\n".join(text_parts) if text_parts else ""
    return content, tool_calls, tool_messages


def anthropic_to_openai(req: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic /v1/messages request to an OpenAI chat payload.

    Returns a dict ready to feed TracingService.chat_completion — keys:
    model, messages, stream, plus extra params (max_tokens, temperature,
    tools, tool_choice, stop, etc.) that TracingService forwards as extra_body.
    """
    out_messages: list[dict[str, Any]] = []

    system_text = _flatten_system(req.get("system"))
    if system_text:
        out_messages.append({"role": "system", "content": system_text})

    for msg in req.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            out_messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_content, tool_calls, tool_messages = _content_blocks_to_openai(content, role)
            # tool_result blocks emit as separate {role: tool} messages first.
            out_messages.extend(tool_messages)
            if role == "assistant":
                assistant_msg: dict[str, Any] = {"role": "assistant", "content": text_content}
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                out_messages.append(assistant_msg)
            else:
                # User message: only append if there's text content. If the
                # user message was purely tool_results, the tool messages
                # above already carry the data.
                if text_content:
                    out_messages.append({"role": role, "content": text_content})
            continue

        # Fallback: unknown content shape, pass through best-effort.
        out_messages.append({"role": role, "content": content})

    out: dict[str, Any] = {
        "model": req.get("model", ""),
        "messages": out_messages,
        "stream": bool(req.get("stream", False)),
    }

    # Ask the upstream to emit a terminal usage chunk when streaming. Ollama's
    # OpenAI-compat endpoint honors stream_options.include_usage and returns a
    # final {usage: {...}} chunk, which the streaming translator carries into
    # message_delta. Without it we fall back to a char-based estimate.
    if out["stream"]:
        out["stream_options"] = {"include_usage": True}

    # max_tokens is required by Anthropic spec; forward as-is (do NOT apply
    # MAX_TOKENS_FLOOR on this path).
    if "max_tokens" in req:
        out["max_tokens"] = req["max_tokens"]
    if "temperature" in req:
        out["temperature"] = req["temperature"]
    if "top_p" in req:
        out["top_p"] = req["top_p"]
    if "stop_sequences" in req:
        out["stop"] = req["stop_sequences"]

    # Tools: Anthropic {name, description, input_schema} → OpenAI function tool.
    if req.get("tools"):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object"}),
                },
            }
            for t in req["tools"]
            if isinstance(t, dict)
        ]

    # tool_choice mapping.
    tc = req.get("tool_choice")
    if isinstance(tc, dict):
        tctype = tc.get("type")
        if tctype == "auto":
            out["tool_choice"] = "auto"
        elif tctype == "any":
            out["tool_choice"] = "required"
        elif tctype == "tool":
            out["tool_choice"] = {
                "type": "function",
                "function": {"name": tc.get("name", "")},
            }

    return out


# ---------------------------------------------------------------------------
# Response translation: OpenAI → Anthropic (non-streaming)
# ---------------------------------------------------------------------------


_FINISH_TO_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(
    openai_resp: dict[str, Any], original_model: str
) -> dict[str, Any]:
    """Translate an OpenAI non-streaming chat completion to Anthropic shape."""
    choices = openai_resp.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    finish = choice.get("finish_reason", "stop")

    content_blocks: list[dict[str, Any]] = []

    # Ollama's OpenAI-compat endpoint exposes thinking-model reasoning under
    # `message.reasoning` (separate from `message.content`). Claude Code, when
    # the request enabled thinking, expects an Anthropic thinking block BEFORE
    # any text block. We emit one whenever upstream produced reasoning, signed
    # with a synthetic opaque signature (see _thinking_signature).
    reasoning = message.get("reasoning") or ""
    if reasoning:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": _thinking_signature(openai_resp.get("id") or _msg_id(), 0),
            }
        )

    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            tool_input = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            tool_input = {"_raw": fn.get("arguments", "")}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", _toolu_id(len(content_blocks))),
                "name": fn.get("name", ""),
                "input": tool_input,
            }
        )

    # Anthropic requires content to be a non-empty list. If the model returned
    # no content and no tool calls, emit a single short "(no output)" text block
    # so the user sees a clear signal instead of a silent blank turn.
    if not content_blocks:
        content_blocks.append({"type": "text", "text": "(no output)"})

    usage = openai_resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    return {
        "id": openai_resp.get("id") or _msg_id(),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": _FINISH_TO_STOP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Streaming translation: OpenAI SSE chunks → Anthropic SSE events
# ---------------------------------------------------------------------------


class _StreamState:
    """Per-stream bookkeeping for the OpenAI→Anthropic SSE translator.

    Tracks which content blocks have been opened so we emit content_block_start
    exactly once per block, and content_block_stop in the right order at the end.
    """

    def __init__(self) -> None:
        self.message_started = False
        # Block index → block descriptor {"type": "text"|"tool_use"|"thinking", "closed": bool}
        self.open_blocks: dict[int, dict[str, Any]] = {}
        # Next content block index to assign.
        self.next_index = 0
        # Per-tool-index accumulator for streaming JSON arguments.
        # {tool_delta_index: {"block_index": int, "emitted_len": int, "full_args": str}}
        self.tool_buffers: dict[int, dict[str, Any]] = {}
        # Final usage/stop_reason carried by the upstream's final chunk.
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] = {}
        # Accumulated output character count (thinking + text + tool args).
        # Used to estimate output_tokens when the upstream emits no final
        # usage chunk — Ollama's OpenAI-compat stream often omits usage, and
        # reporting output_tokens=0 makes some clients (Claude Code) treat the
        # response as empty and retry until they give up.
        self.output_chars = 0
        # Thinking-only character count, tracked separately so we can emit
        # output_tokens_details.thinking_tokens in the final message_delta.
        # Claude Code sends the thinking-token-count beta header and expects
        # this breakdown; without it the client silently discards the response.
        self.thinking_chars = 0


async def openai_to_anthropic_stream(
    openai_chunk_iter: AsyncIterator[dict[str, Any]],
    original_model: str,
    message_id: str,
    heartbeat_seconds: float = 15.0,
    input_tokens: int = 0,
) -> AsyncIterator[str]:
    """Translate an OpenAI streaming chat completion into an Anthropic SSE stream.

    Consumes parsed OpenAI chunk dicts (caller parses the `data: {...}` lines).
    Yields Anthropic SSE event strings.

    `input_tokens` is reported in message_start.usage. Ollama's stream doesn't
    carry input usage, so the caller passes a chars-based estimate; without it
    some clients (Claude Code) see input_tokens=0 on a large request and behave
    oddly. Defaults to 0 for backward compatibility.

    Emits a `ping` event every `heartbeat_seconds` of upstream silence so
    Claude Code doesn't time out during long thinking gaps.
    """
    state = _StreamState()

    # message_start is emitted lazily on the first content/finish chunk so we
    # don't emit a header for a stream that errors immediately.
    def _ensure_started() -> str:
        if state.message_started:
            return ""
        state.message_started = True
        return _sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": original_model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 0,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "output_tokens_details": {"thinking_tokens": 0},
                    },
                },
            },
        )

    def _close_open_thinking_block() -> str:
        # Close any thinking block that's still open BEFORE opening the next
        # block. Anthropic requires signature_delta + content_block_stop to
        # fully close a thinking block before subsequent blocks begin. Ollama
        # gives no explicit "reasoning phase ended" signal, so we close lazily
        # on the first non-reasoning delta. No-op if no thinking block is open.
        out: list[str] = []
        for idx, blk in state.open_blocks.items():
            if blk["type"] == "thinking" and not blk["closed"]:
                out.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "signature_delta",
                                "signature": _thinking_signature(message_id, idx),
                            },
                        },
                    )
                )
                out.append(_sse("content_block_stop", {"type": "content_block_stop", "index": idx}))
                blk["closed"] = True
        return "".join(out)

    def _open_text_block() -> tuple[int, str] | None:
        # Reuse an existing open text block, or open a new one.
        for idx, blk in state.open_blocks.items():
            if blk["type"] == "text" and not blk["closed"]:
                return idx, ""
        if state.next_index in state.open_blocks:
            # Can't open text after tools — Anthropic expects contiguous text
            # at the start. Bail and drop the delta.
            return None
        # First close any open thinking block (Anthropic ordering invariant).
        prefix = _close_open_thinking_block()
        idx = state.next_index
        state.next_index += 1
        state.open_blocks[idx] = {"type": "text", "closed": False}
        return idx, prefix + _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            },
        )

    def _open_thinking_block() -> tuple[int, str]:
        # Reuse an existing open thinking block, or open a new one.
        # Anthropic permits multiple separate thinking blocks, but Ollama emits
        # reasoning as one contiguous phase before content — so one block is enough.
        for idx, blk in state.open_blocks.items():
            if blk["type"] == "thinking" and not blk["closed"]:
                return idx, ""
        idx = state.next_index
        state.next_index += 1
        state.open_blocks[idx] = {"type": "thinking", "closed": False}
        return idx, _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                # NOTE: no `signature` here. Real Anthropic omits the field on
                # content_block_start — the signature is delivered only via a
                # later signature_delta (which we emit before content_block_stop).
                # Including `"signature": ""` makes Claude Code's strict TS SDK
                # reject the block, abort the stream, and auto-retry to empty.
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )

    def _open_tool_block(tool_delta_index: int, tc_delta: dict[str, Any]) -> tuple[int, str]:
        buf = state.tool_buffers.get(tool_delta_index)
        if buf is not None:
            return buf["block_index"], ""
        # First close any open thinking block (Anthropic ordering invariant).
        prefix = _close_open_thinking_block()
        idx = state.next_index
        state.next_index += 1
        fn = tc_delta.get("function") or {}
        tool_id = tc_delta.get("id") or _toolu_id(idx)
        tool_name = fn.get("name", "")
        state.open_blocks[idx] = {"type": "tool_use", "closed": False}
        state.tool_buffers[tool_delta_index] = {
            "block_index": idx,
            "emitted_len": 0,
            "full_args": "",
            "id": tool_id,
            "name": tool_name,
        }
        return idx, prefix + _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {},
                },
            },
        )

    iterator = openai_chunk_iter.__aiter__()
    exhausted = False

    while not exhausted:
        # Wait for the next upstream chunk WITHOUT cancelling it on timeout.
        # The previous implementation used asyncio.wait_for, which cancels its
        # inner awaitable when the timeout fires — and cancelling an in-flight
        # httpx stream read severs the upstream connection. During Ollama's long
        # prompt-eval silence (40s+ on large tool-heavy prompts, while
        # heartbeat_seconds is ~15s) that cancellation killed the stream and we
        # fell through to the empty-stream fallback, yielding a blank response.
        # asyncio.wait does NOT cancel pending tasks, so we can emit ping
        # keepalives while the slow upstream chunk stays in flight.
        next_task = asyncio.ensure_future(iterator.__anext__())
        while True:
            done, _pending = await asyncio.wait({next_task}, timeout=heartbeat_seconds)
            if done:
                break
            if state.message_started:
                yield _sse("ping", {"type": "ping"})
        try:
            chunk = next_task.result()
        except StopAsyncIteration:
            exhausted = True
            break

        # Upstream error mid-stream (Ollama 4xx/5xx or connection failure from
        # the native stream path, which emits it as a data frame since SSE
        # headers are already committed). Emit a proper Anthropic `event: error`
        # and terminate — do NOT fall through to the blank empty-stream fallback,
        # which would mask the failure as an empty response and make Claude Code
        # loop. Anthropic's overload stream is just the error event + close.
        err = chunk.get("error")
        if isinstance(err, dict):
            yield _sse(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": err.get("type") or "overloaded_error",
                        "message": err.get("message") or "upstream stream error",
                    },
                },
            )
            return

        # Carry usage/finish forward for the final message_delta.
        if chunk.get("usage"):
            state.usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        finish = choice.get("finish_reason")

        events: list[str] = []

        # OpenAI sometimes sends a final chunk with only usage + finish_reason
        # (stream_options.include_usage). Don't treat that as content.
        text_delta = delta.get("content")
        reasoning_delta = delta.get("reasoning") or ""
        tool_call_deltas = delta.get("tool_calls") or []

        if text_delta or reasoning_delta or tool_call_deltas:
            events.append(_ensure_started())

        # Reasoning phase (thinking-model only): Ollama separates this from
        # content. Emit as Anthropic thinking_delta on a thinking block.
        # Must precede text per Anthropic spec; Ollama emits reasoning first
        # so natural arrival order keeps this invariant.
        if reasoning_delta:
            idx, start_evt = _open_thinking_block()
            events.append(start_evt)
            events.append(
                _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "thinking_delta", "thinking": reasoning_delta},
                    },
                )
            )
            state.output_chars += len(reasoning_delta)
            state.thinking_chars += len(reasoning_delta)

        if text_delta:
            opened = _open_text_block()
            if opened is not None:
                idx, start_evt = opened
                events.append(start_evt)
                events.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "text_delta", "text": text_delta},
                        },
                    )
                )
            state.output_chars += len(text_delta)

        for tc_delta in tool_call_deltas:
            tdi = tc_delta.get("index", 0)
            idx, start_evt = _open_tool_block(tdi, tc_delta)
            events.append(start_evt)
            buf = state.tool_buffers[tdi]
            args_chunk = (tc_delta.get("function") or {}).get("arguments", "") or ""
            buf["full_args"] += args_chunk
            new_slice = args_chunk  # OpenAI sends the delta, not cumulative
            if new_slice:
                events.append(
                    _sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {"type": "input_json_delta", "partial_json": new_slice},
                        },
                    )
                )
                buf["emitted_len"] += len(new_slice)

        for evt in events:
            yield evt

        if finish:
            state.finish_reason = finish
            break

    # Stream ended — close any open blocks and emit the terminal events.
    if not state.message_started:
        # Empty stream: upstream yielded no content and no tool calls (e.g. a
        # small model that stopped immediately, max_tokens=0, or a stop sequence
        # matched at the start). Emit a single short text block with a clear
        # "(no output)" signal so the user sees something instead of a silent
        # blank turn — a blank assistant message makes Claude Code loop
        # fruitlessly retrying for real output.
        yield _ensure_started()
        opened = _open_text_block()
        if opened is not None:
            idx, start_evt = opened
            yield start_evt
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": "(no output)"},
                },
            )
            state.output_chars += len("(no output)")

    # Close all open blocks in index order.
    for idx in sorted(state.open_blocks.keys()):
        blk = state.open_blocks[idx]
        if not blk["closed"]:
            # Anthropic requires thinking blocks to receive a signature_delta
            # before the stop so clients can persist provenance. We synthesize
            # one — see _thinking_signature.
            if blk["type"] == "thinking":
                yield _sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {
                            "type": "signature_delta",
                            "signature": _thinking_signature(message_id, idx),
                        },
                    },
                )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
            blk["closed"] = True

    stop_reason = _FINISH_TO_STOP.get(state.finish_reason or "stop", "end_turn")
    # Prefer real usage from the upstream's final chunk; fall back to a
    # char-based estimate. Ollama's OpenAI-compat stream frequently omits a
    # terminal usage chunk, and reporting output_tokens=0 makes clients like
    # Claude Code discard the response as empty and retry to no avail.
    out_tokens = state.usage.get("completion_tokens") or max(1, state.output_chars // 4)
    thinking_tokens = max(0, state.thinking_chars // 4)

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {
                "output_tokens": out_tokens,
                "output_tokens_details": {"thinking_tokens": thinking_tokens},
            },
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens_anthropic(req: dict[str, Any]) -> int:
    """Rough token count for /v1/messages/count_tokens.

    No tiktoken dep — uses the standard ~4 chars/token English heuristic with
    a small uplift for JSON tool schemas. Off by ±15% on tool-heavy requests,
    which is acceptable for Claude Code's context-window budgeting.
    """
    total_chars = 0

    total_chars += len(_flatten_system(req.get("system")))

    for msg in req.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    total_chars += len(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    # Input JSON is denser — count chars and add 20% overhead.
                    raw = json.dumps(block.get("input", {}))
                    total_chars += int(len(raw) * 1.2)
                elif block.get("type") == "tool_result":
                    rc = block.get("content", "")
                    if isinstance(rc, str):
                        total_chars += len(rc)
                    else:
                        total_chars += len(json.dumps(rc)) + 4

    # Tool definitions also count toward input tokens.
    for tool in req.get("tools") or []:
        if isinstance(tool, dict):
            raw = json.dumps(tool)
            total_chars += int(len(raw) * 1.2)

    # ~4 chars per token, round up, minimum 1.
    return max(1, (total_chars + 3) // 4)
