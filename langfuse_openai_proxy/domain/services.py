"""TracingService - core business logic for Langfuse-traced LLM calls.

Uses Langfuse v4 observations via lf.start_observation(as_type='generation').
The returned LangfuseGeneration object uses:
  - .update(output=..., usage_details=..., level=...) to record results
  - .end() to finalize the observation
"""

import asyncio
import json
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
    ):
        self._create_langfuse = langfuse_client_factory
        self._openai = openai_client
        self._upstream_base_url = upstream_base_url
        self._upstream_api_key = upstream_api_key
        # When True, copy upstream `reasoning` deltas into `content` so clients
        # that only read `content` (OpenClaw's openai-completions adapter) see the
        # model's output instead of an empty stream. See Settings.reasoning_as_content.
        self._reasoning_as_content = reasoning_as_content

    async def chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
    ) -> dict:
        """Execute non-streaming chat completion with Langfuse tracing."""
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

    async def stream_chat_completion(
        self,
        credentials: Credentials,
        request: ChatRequest,
        host: str,
    ) -> AsyncGenerator[str, None]:
        """Execute streaming chat completion with Langfuse tracing.

        Yields SSE-formatted chunks. Collects content for tracing after stream ends.
        """
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
            generation.end()

        except Exception as e:
            generation.update(level="ERROR", status_message=str(e))
            generation.end()
            raise
        finally:
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
