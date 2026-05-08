"""Tests for credential parsing."""

import pytest

from langfuse_openai_proxy.api.routes import parse_credentials
from langfuse_openai_proxy.domain.errors import MissingCredentialsError


def test_combined_format_splits_correctly():
    creds = parse_credentials("Bearer pk-lf-abc|sk-lf-xyz", None)
    assert creds.public_key == "pk-lf-abc"
    assert creds.secret_key == "sk-lf-xyz"


def test_combined_format_strips_whitespace():
    creds = parse_credentials("Bearer  pk-lf-abc | sk-lf-xyz ", None)
    assert creds.public_key == "pk-lf-abc"
    assert creds.secret_key == "sk-lf-xyz"


def test_separate_headers():
    creds = parse_credentials("Bearer sk-lf-xyz", "pk-lf-abc")
    assert creds.public_key == "pk-lf-abc"
    assert creds.secret_key == "sk-lf-xyz"


def test_missing_both_raises():
    with pytest.raises(MissingCredentialsError):
        parse_credentials(None, None)


def test_missing_public_key_raises():
    with pytest.raises(MissingCredentialsError):
        parse_credentials("Bearer sk-lf-xyz", None)


def test_missing_secret_key_raises():
    with pytest.raises(MissingCredentialsError):
        parse_credentials(None, "pk-lf-abc")


def test_empty_bearer_raises():
    with pytest.raises(MissingCredentialsError):
        parse_credentials("Bearer ", "pk-lf-abc")


def test_pipe_takes_precedence_over_separate_header():
    """Combined format should win even if X-Langfuse-Public-Key is also set."""
    creds = parse_credentials("Bearer pk-combined|sk-combined", "pk-separate")
    assert creds.public_key == "pk-combined"
    assert creds.secret_key == "sk-combined"
