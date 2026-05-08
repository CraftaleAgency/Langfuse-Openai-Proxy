"""Tests for domain models."""

from langfuse_openai_proxy.domain.models import (
    ChatRequest,
    Credentials,
    EmbeddingRequest,
)


def test_credentials_is_frozen():
    creds = Credentials(public_key="pk", secret_key="sk")
    assert creds.public_key == "pk"
    assert creds.secret_key == "sk"


def test_chat_request_defaults():
    req = ChatRequest(model="gpt-4", messages=[{"role": "user", "content": "hi"}])
    assert req.stream is False
    assert req.extra_params is None


def test_chat_request_with_extras():
    req = ChatRequest(
        model="gpt-4",
        messages=[],
        stream=True,
        extra_params={"temperature": 0.7, "max_tokens": 100},
    )
    assert req.stream is True
    assert req.extra_params["temperature"] == 0.7


def test_embedding_request_with_string_input():
    req = EmbeddingRequest(model="text-embedding-3-small", input="hello")
    assert req.input == "hello"


def test_embedding_request_with_list_input():
    req = EmbeddingRequest(model="text-embedding-3-small", input=["hello", "world"])
    assert len(req.input) == 2
