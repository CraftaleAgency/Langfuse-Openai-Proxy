"""Tests for domain errors."""

from langfuse_openai_proxy.domain.errors import (
    MissingCredentialsError,
    ProxyError,
    UpstreamError,
)


def test_proxy_error_hierarchy():
    assert issubclass(MissingCredentialsError, ProxyError)
    assert issubclass(UpstreamError, ProxyError)


def test_upstream_error_default_status():
    err = UpstreamError("bad gateway")
    assert err.status_code == 502
    assert err.message == "bad gateway"


def test_upstream_error_custom_status():
    err = UpstreamError("rate limited", status_code=429)
    assert err.status_code == 429


def test_missing_credentials_is_proxy_error():
    err = MissingCredentialsError("no creds")
    assert isinstance(err, ProxyError)
    assert err.message == "no creds"
