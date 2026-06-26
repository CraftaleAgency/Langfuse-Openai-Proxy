"""TracingService - core business logic for Langfuse-traced LLM calls.

Uses Langfuse v4 observations via lf.start_observation(as_type='generation').
The returned LangfuseGeneration object uses:
  - .update(output=..., usage_details=..., level=...) to record results
  - .end() to finalize the observation
"""

import asyncio
import json
import time
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI

from ..infrastructure.openai_client import get_http_client
from .models import ChatRequest, Credentials, EmbeddingRequest, ResponsesRequest


def _extract_input_text(input_data: str | list[dict]) -> str:
    """Extract readable text from Responses API input for Langfuse tracing."""
    if isinstance(input_data, str):
        return input_data
    texts = []
    for item in input_data:
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("input_text", "text"):
                        texts.append(part.get("text", ""))
    return " ".join(texts)


def _wants_ollama_native(extra_params: dict | None) -> bool:
    """Detect when a caller explicitly requests Ollama-specific routing.

    Ollama's OpenAI-compat `/v1/chat/completions` endpoint silently ignores
    the `think` parameter — thinking models (gemma4, qwen3) still emit on
    `delta.reasoning` with empty `delta.content`, then abort with
    `finish_reason=length` once max_tokens is exhausted. The only path
    that honors `think: false` is the native `/api/chat` endpoint.

    Trigger native routing whenever a caller passes `think` explicitly
    (True or False) or sets `ollama_native: true`. This lets JSON-mode
    clients like MiroFish suppress reasoning, while chat clients that
    rely on REASONING_AS_CONTENT keep using the OpenAI-compat path.
    """
    if not extra_params:
        return False
    if "think" in extra_params:
        return True
    return bool(extra_params.get("ollama_native"))


def _build_ollama_native_body(model: str, messages: list, extra_params: dict) -> dict:
    """Translate an OpenAI chat request to Ollama's native /api/chat format."""
    body: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    options: dict = {}
    if "max_tokens" in extra_params:
        options["num_predict"] = extra_params["max_tokens"]
    if "temperature" in extra_params:
        options["temperature"] = extra_params["temperature"]
    if "top_p" in extra_params:
        options["top_p"] = extra_params["top_p"]
    if "options" in extra_params and isinstance(extra_params["options"], dict):
        # Caller-provided options win over individual knobs above.
        options.update(extra_params["options"])
    if options:
        body["options"] = options

    if "think" in extra_params:
        body["think"] = bool(extra_params["think"])

    # response_format: {"type": "json_object"} → format: "json"
    rf = extra_params.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_object":
        body["format"] = "json"

    # Pass through any Ollama-native keys the caller knows about.
    for k in ("format", "keep_alive", "seed", "tools", "tool_choice"):
        if k in extra_params:
            body[k] = extra_params[k]

    return body


def _ollama_native_to_openai(model: str, resp: dict) -> dict:
    """Translate an Ollama /api/chat response to OpenAI chat-completion shape."""
    msg = resp.get("message", {}) or {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning", "") or ""
    finish = "stop"
    if resp.get("done_reason") == "length":
        finish = "length"
    elif resp.get("done_reason") == "tool_calls":
        finish = "tool_calls"

    message = {"role": msg.get("role", "assistant"), "content": content}
    if reasoning:
        message["reasoning"] = reasoning

    usage = {
        "prompt_tokens": resp.get("prompt_eval_count", 0) or 0,
        "completion_tokens": resp.get("eval_count", 0) or 0,
        "total_tokens": (resp.get("prompt_eval_count", 0) or 0) + (resp.get("eval_count", 0) or 0),
    }

    return {
        "id": f"chatcmpl-ollama-{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }
        ],
        "usage": usage,
    }


def _apply_max_tokens_floor(extra_params: dict | None, floor: int | None) -> dict:
    """Inject or raise `max_tokens` so reasoning models don't get starved.

    Many OpenAI clients default to a small max_tokens (50 is common). Reasoning
    models served via Ollama (qwen3, gemma4, thinker14b) burn ~100+ tokens on
    `<think>...</think>` before any visible output emerges, so a 50-token budget
    truncates thinking mid-stream and the client sees an empty response. With a
    floor set, requests with no max_tokens get the floor, and requests with a
    max_tokens below the floor are raised to it. Requests already at or above
    the floor pass through untouched.
    """
    out = dict(extra_params) if extra_params else {}
    if not floor or floor <= 0:
        return out
    current = out.get("max_tokens")
    if current is None or (isinstance(current, int) and current < floor):
        out["max_tokens"] = floor
    return out


def _ollama_native_base_url(upstream_base_url: str) -> str:
    """Strip the trailing /v1 from the upstream URL so we can hit /api/chat.

    UPSTREAM_BASE_URL is normally configured as http://ollama:11434/v1.
    The native chat endpoint lives at /api/chat (no /v1 prefix).
    """
    url = upstream_base_url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def _extract_output_text(response_data: dict) -> str:
    """Extract readable text from Responses API output for Langfuse tracing."""
    texts = []
    for item in response_data.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") in ("output_text", "text"):
                    texts.append(content.get("text", ""))
    return " ".join(texts)


class TracingService:
    """Orchestrates LLM calls with Langfuse tracing.

    Uses Langfuse v4 SDK: lf.start_observation(as_type='generation') returns
    a LangfuseGeneration with .update() for data and .end() to finalize.
    """

    def __init__(
        self,
        langfuse_client_factory: type,
        openai_client: AsyncOpenAI,
        upstream_base_url: str,
        upstream_api_key: str,
        reasoning_as_content: bool = False,
        max_tokens_floor: int | None = None,
    ):
        self._create_langfuse = langfuse_client_factory
        self._openai = openai_client
        self._upstream_base_url = upstream_base_url
        self._upstream_api_key = upstream_api_key
        # When True, copy upstream `reasoning` deltas into `content` so clients
        # that only read `content` (OpenClaw's openai-completions adapter) see the
        # model's output instead of an empty stream. See Settings.reasoning_as_content.
        self._reasoning_as_content = reasoning_as_content
        # When set, inject/raise max_tokens on chat requests so reasoning models
        # don't burn their entire (small) budget on hidden thinking. See
        # Settings.max_tokens_floor.
        self._max_tokens_floor = max_tokens_floor

    async def chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
        apply_max_tokens_floor: bool = True,
    ) -> dict:
        """Execute non-streaming chat completion with Langfuse tracing."""
        # Apply max_tokens floor before any routing decision — both the native
        # /api/chat path and the OpenAI-compat /v1 path read max_tokens from
        # extra_params. See _apply_max_tokens_floor(). The Anthropic shim path
        # passes apply_max_tokens_floor=False because Anthropic clients always
        # send an explicit max_tokens per spec.
        if apply_max_tokens_floor:
            request.extra_params = _apply_max_tokens_floor(request.extra_params, self._max_tokens_floor)

        lf = self._create_langfuse(
            credentials.public_key,
            credentials.secret_key,
            host,
        )

        generation = lf.start_observation(
            name="chat-completion",
            as_type="generation",
            model=request.model,
            input=request.messages,
            metadata={"stream": False},
        )

        # Route to Ollama's native /api/chat when the caller explicitly
        # requests thinking control — the OpenAI-compat /v1 endpoint
        # silently ignores `think` and lets reasoning models burn the
        # entire max_tokens budget on hidden reasoning, surfacing as
        # an empty content response. See _wants_ollama_native().
        if _wants_ollama_native(request.extra_params):
            try:
                data = await self._ollama_native_chat(request, non_stream=True)
                generation.update(
                    output=(data.get("choices", [{}])[0].get("message") or {}).get("content", ""),
                    usage_details={
                        "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        "total_tokens": data.get("usage", {}).get("total_tokens", 0),
                    },
                )
                generation.end()
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                generation.end()
                raise
            finally:
                await asyncio.to_thread(lf.flush)
            return data

        try:
            kwargs = request.extra_params or {}
            response = await self._openai.chat.completions.create(
                model=request.model,
                messages=request.messages,
                stream=False,
                extra_body=kwargs,
            )

            generation.update(
                output=response.choices[0].message.content,
                usage_details={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
            )
            generation.end()
        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            raise
        finally:
            await asyncio.to_thread(lf.flush)

        data = response.model_dump()

        # Non-stream remap: same reasoning → content fallback as the streaming
        # path, for clients that only read `message.content`.
        if self._reasoning_as_content:
            for choice in data.get("choices", []):
                message = choice.get("message")
                if isinstance(message, dict) and not message.get("content"):
                    reasoning = message.get("reasoning")
                    if reasoning:
                        message["content"] = reasoning

        return data

    async def _ollama_native_chat(self, request: ChatRequest, non_stream: bool = True) -> dict:
        """Call Ollama's native /api/chat endpoint with full param support.

        Used when the caller sets `think` explicitly — the OpenAI-compat
        layer at /v1/chat/completions silently drops it. Returns the
        response translated to OpenAI chat-completion shape.
        """
        http = get_http_client()
        base = _ollama_native_base_url(self._upstream_base_url)
        url = f"{base}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self._upstream_api_key:
            headers["Authorization"] = f"Bearer {self._upstream_api_key}"

        body = _build_ollama_native_body(
            request.model, request.messages, request.extra_params or {}
        )
        body["stream"] = False

        resp = await http.post(url, headers=headers, json=body, timeout=600)
        resp.raise_for_status()
        payload = resp.json()
        return _ollama_native_to_openai(request.model, payload)

    async def _ollama_native_stream(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Stream from Ollama's native /api/chat in OpenAI SSE shape."""
        http = get_http_client()
        base = _ollama_native_base_url(self._upstream_base_url)
        url = f"{base}/api/chat"
        headers = {"Content-Type": "application/json"}
        if self._upstream_api_key:
            headers["Authorization"] = f"Bearer {self._upstream_api_key}"

        body = _build_ollama_native_body(
            request.model, request.messages, request.extra_params or {}
        )
        body["stream"] = True

        created = int(time.time())
        completion_id = f"chatcmpl-ollama-{int(time.time() * 1000)}"

        async with http.stream("POST", url, headers=headers, json=body, timeout=600) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message", {}) or {}
                content = msg.get("content", "") or ""
                reasoning = msg.get("reasoning", "") or ""
                if not content and reasoning and self._reasoning_as_content:
                    content = reasoning
                if not content and not reasoning:
                    # Keep-alive or mid-stream empty delta; skip unless final.
                    if chunk.get("done"):
                        break
                    continue
                finish = None
                if chunk.get("done"):
                    finish = "length" if chunk.get("done_reason") == "length" else "stop"
                data = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content} if content else {},
                            "finish_reason": finish,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"
                if chunk.get("done"):
                    break

        yield "data: [DONE]\n\n"

    async def stream_chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
        apply_max_tokens_floor: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion with Langfuse tracing.

        Yields SSE-formatted chunks. Collects content for tracing after stream ends.
        """
        # Apply max_tokens floor before any routing decision — see chat_completion().
        # The Anthropic shim path passes apply_max_tokens_floor=False.
        if apply_max_tokens_floor:
            request.extra_params = _apply_max_tokens_floor(request.extra_params, self._max_tokens_floor)

        lf = self._create_langfuse(
            credentials.public_key,
            credentials.secret_key,
            host,
        )

        generation = lf.start_observation(
            name="chat-completion",
            as_type="generation",
            model=request.model,
            input=request.messages,
            metadata={"stream": True},
        )

        # Native routing for the same reason as the non-streaming path —
        # Ollama's /v1 endpoint silently ignores `think` and lets reasoning
        # models emit on `delta.reasoning` with empty `delta.content`. See
        # _wants_ollama_native() and chat_completion() for the full rationale.
        if _wants_ollama_native(request.extra_params):
            collected_content = []
            try:
                async for sse_chunk in self._ollama_native_stream(request):
                    yield sse_chunk
                    # Pull the delta text back out for tracing. Format matches
                    # what _ollama_native_stream emits.
                    if sse_chunk.startswith("data: ") and sse_chunk != "data: [DONE]\n\n":
                        try:
                            payload = json.loads(sse_chunk[6:])
                            choices = payload.get("choices") or []
                            if choices:
                                delta = choices[0].get("delta") or {}
                                if delta.get("content"):
                                    collected_content.append(delta["content"])
                        except json.JSONDecodeError:
                            pass
                generation.update(output="".join(collected_content))
            except Exception as e:
                generation.update(level="ERROR", status_message=str(e))
                raise
            finally:
                # end() must run in finally: when the consumer stops early
                # (e.g. the Anthropic translator breaks on finish_reason and
                # closes this generator via aclose), the try-body's end() is
                # never reached. OTel does not export unended spans, so without
                # this the streaming trace is silently lost.
                generation.end()
                await asyncio.to_thread(lf.flush)
            return

        collected_content = []

        try:
            kwargs = request.extra_params or {}
            stream = await self._openai.chat.completions.create(
                model=request.model,
                messages=request.messages,
                stream=True,
                extra_body=kwargs,
            )

            async for chunk in stream:
                data = json.loads(chunk.model_dump_json())

                # Remap reasoning → content for clients that only read `content`.
                # Ollama's /v1 endpoint streams reasoning-model output in
                # `delta.reasoning` with `delta.content` empty; without this,
                # such clients see an empty stream and abort (stop_reason=length).
                if self._reasoning_as_content and data.get("choices"):
                    for choice in data["choices"]:
                        delta = choice.get("delta")
                        if not isinstance(delta, dict):
                            continue
                        reasoning = delta.get("reasoning")
                        if reasoning and not delta.get("content"):
                            delta["content"] = reasoning
                            # Reflect the remapped text in tracing too.
                            collected_content.append(reasoning)

                yield f"data: {json.dumps(data)}\n\n"

                # Collect content for tracing (native content, not already captured above)
                if chunk.choices and chunk.choices[0].delta.content:
                    collected_content.append(chunk.choices[0].delta.content)

            yield "data: [DONE]\n\n"

            generation.update(output="".join(collected_content))

        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            raise
        finally:
            # See native-stream path: end() in finally so early-closing
            # consumers don't leave the span unended (and thus unexported).
            generation.end()
            await asyncio.to_thread(lf.flush)

    async def embedding(
        self,
        credentials: Credentials,
        request: EmbeddingRequest,
        host: str,
    ) -> dict:
        """Execute embedding with Langfuse tracing."""
        lf = self._create_langfuse(
            credentials.public_key,
            credentials.secret_key,
            host,
        )

        generation = lf.start_observation(
            name="embedding",
            as_type="generation",
            model=request.model,
            input=request.input,
        )

        try:
            kwargs = request.extra_params or {}
            response = await self._openai.embeddings.create(
                model=request.model,
                input=request.input,
                **kwargs,
            )

            generation.update(
                output={"usage": response.usage.model_dump() if response.usage else None},
            )
            generation.end()
        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            raise
        finally:
            await asyncio.to_thread(lf.flush)

        return response.model_dump()

    async def response(
        self,
        credentials: Credentials,
        request: ResponsesRequest,
        host: str,
    ) -> tuple[dict, int]:
        """Execute non-streaming Responses API call with Langfuse tracing."""
        lf = self._create_langfuse(
            credentials.public_key,
            credentials.secret_key,
            host,
        )

        generation = lf.start_observation(
            name="response",
            as_type="generation",
            model=request.model,
            input=_extract_input_text(request.input),
            metadata={"stream": False},
        )

        try:
            http = get_http_client()
            url = f"{self._upstream_base_url}/responses"
            headers = {"Content-Type": "application/json"}
            if self._upstream_api_key:
                headers["Authorization"] = f"Bearer {self._upstream_api_key}"
            body = {"model": request.model, "input": request.input}
            if request.extra_params:
                body.update(request.extra_params)

            resp = await http.post(url, headers=headers, json=body, timeout=120)
            response_data = resp.json()

            usage = response_data.get("usage", {})
            generation.update(
                output=_extract_output_text(response_data),
                usage_details={
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            )
            generation.end()
        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            raise
        finally:
            await asyncio.to_thread(lf.flush)

        return response_data, resp.status_code

    async def stream_response(
        self,
        credentials: Credentials,
        request: ResponsesRequest,
        host: str,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming Responses API call with Langfuse tracing.

        Forwards raw SSE events from upstream. Collects text deltas and usage
        from the response.completed event for Langfuse tracing.
        """
        lf = self._create_langfuse(
            credentials.public_key,
            credentials.secret_key,
            host,
        )

        generation = lf.start_observation(
            name="response",
            as_type="generation",
            model=request.model,
            input=_extract_input_text(request.input),
            metadata={"stream": True},
        )

        collected_deltas = []
        usage_data = {}

        try:
            http = get_http_client()
            url = f"{self._upstream_base_url}/responses"
            headers = {"Content-Type": "application/json"}
            if self._upstream_api_key:
                headers["Authorization"] = f"Bearer {self._upstream_api_key}"
            body = {"model": request.model, "input": request.input, "stream": True}
            if request.extra_params:
                body.update(request.extra_params)

            buffer = ""
            async with http.stream("POST", url, headers=headers, json=body, timeout=120) as resp:
                async for chunk in resp.aiter_text():
                    yield chunk
                    buffer += chunk
                    while "\n\n" in buffer:
                        event_text, buffer = buffer.split("\n\n", 1)
                        for line in event_text.split("\n"):
                            if line.startswith("data: "):
                                try:
                                    data = json.loads(line[6:])
                                    if data.get("type") == "response.output_text.delta":
                                        collected_deltas.append(data.get("delta", ""))
                                    elif data.get("type") == "response.completed":
                                        response_obj = data.get("response", {})
                                        usage_data = response_obj.get("usage", {})
                                except json.JSONDecodeError:
                                    pass

            generation.update(
                output="".join(collected_deltas),
                usage_details={
                    "input_tokens": usage_data.get("input_tokens", 0),
                    "output_tokens": usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            )
            generation.end()
        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            raise
        finally:
            await asyncio.to_thread(lf.flush)
