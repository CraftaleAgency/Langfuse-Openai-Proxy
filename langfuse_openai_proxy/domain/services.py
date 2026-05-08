"""TracingService - core business logic for Langfuse-traced LLM calls.

Uses Langfuse v4 observations via lf.start_observation(as_type='generation').
The returned LangfuseGeneration object uses:
  - .update(output=..., usage_details=..., level=...) to record results
  - .end() to finalize the observation
"""

import asyncio
from typing import AsyncGenerator

from langfuse import Langfuse
from openai import AsyncOpenAI

from .models import ChatRequest, Credentials, EmbeddingRequest


class TracingService:
    """Orchestrates LLM calls with Langfuse tracing.

    Uses Langfuse v4 SDK: lf.start_observation(as_type='generation') returns
    a LangfuseGeneration with .update() for data and .end() to finalize.
    """

    def __init__(
        self,
        langfuse_client_factory: type,
        openai_client: AsyncOpenAI,
    ):
        self._create_langfuse = langfuse_client_factory
        self._openai = openai_client

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
                **kwargs,
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

        return response.model_dump()

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
                **kwargs,
            )

            async for chunk in stream:
                data = chunk.model_dump_json()
                yield f"data: {data}\n\n"

                # Collect content for tracing
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
