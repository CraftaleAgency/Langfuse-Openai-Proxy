"""Tests for host validation (SSRF prevention)."""

import pytest

from langfuse_openai_proxy.infrastructure.host_validation import validate_langfuse_host


def test_valid_cloud_langfuse():
    assert validate_langfuse_host("https://cloud.langfuse.com") == "https://cloud.langfuse.com"


def test_valid_us_cloud():
    assert validate_langfuse_host("https://us.cloud.langfuse.com") == "https://us.cloud.langfuse.com"


def test_valid_eu_cloud():
    assert validate_langfuse_host("https://eu.cloud.langfuse.com") == "https://eu.cloud.langfuse.com"


def test_valid_subdomain_of_allowed():
    assert validate_langfuse_host("https://selfhosted.cloud.langfuse.com") == "https://selfhosted.cloud.langfuse.com"


def test_rejects_http():
    with pytest.raises(ValueError, match="HTTPS"):
        validate_langfuse_host("http://cloud.langfuse.com")


def test_rejects_private_ip():
    with pytest.raises(ValueError, match="IP"):
        validate_langfuse_host("https://192.168.1.1")


def test_rejects_localhost():
    with pytest.raises(ValueError, match="IP"):
        validate_langfuse_host("https://127.0.0.1")


def test_rejects_aws_metadata():
    with pytest.raises(ValueError, match="IP"):
        validate_langfuse_host("https://169.254.169.254")


def test_rejects_untrusted_domain():
    with pytest.raises(ValueError, match="not in the allowed list"):
        validate_langfuse_host("https://evil.com")


def test_rejects_missing_hostname():
    with pytest.raises(ValueError, match="hostname"):
        validate_langfuse_host("https://")


def test_custom_allowed_hosts():
    assert (
        validate_langfuse_host(
            "https://langfuse.example.com",
            allowed_hosts=["example.com"],
        )
        == "https://langfuse.example.com"
    )
