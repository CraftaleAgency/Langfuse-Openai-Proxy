"""Host validation for Langfuse host URLs.

Prevents SSRF by rejecting private IPs, non-HTTPS schemes,
and untrusted domains in user-supplied X-Langfuse-Host headers.
"""

import ipaddress
from urllib.parse import urlparse


def validate_langfuse_host(host: str, allowed_hosts: list[str] | None = None) -> str:
    """Validate and return a safe Langfuse host URL.

    Args:
        host: The user-supplied host URL from X-Langfuse-Host header.
        allowed_hosts: Optional list of allowed hostnames.
            Defaults to common Langfuse hosts.

    Returns:
        The validated host URL string.

    Raises:
        ValueError: If the host is invalid or not allowed.
    """
    if allowed_hosts is None:
        allowed_hosts = [
            "cloud.langfuse.com",
            "us.cloud.langfuse.com",
            "eu.cloud.langfuse.com",
        ]

    parsed = urlparse(host)

    # Must use HTTPS
    if parsed.scheme != "https":
        raise ValueError(f"Invalid Langfuse host: must use HTTPS, got '{parsed.scheme}'")

    # Must have a hostname
    if not parsed.hostname:
        raise ValueError("Invalid Langfuse host: missing hostname")

    hostname = parsed.hostname.lower()

    # Reject IP addresses (prevents SSRF to internal IPs)
    try:
        ipaddress.ip_address(hostname)
        raise ValueError("Invalid Langfuse host: direct IP addresses are not allowed")
    except ValueError as e:
        # If this is our own "not allowed" error, re-raise it
        if "direct IP addresses" in str(e):
            raise
        # Otherwise it's ip_address() saying "not an IP" — expected, continue

    # Must match allowed hosts (exact match or subdomain)
    if not any(
        hostname == allowed or hostname.endswith("." + allowed) for allowed in allowed_hosts
    ):
        raise ValueError(
            f"Invalid Langfuse host: '{hostname}' is not in the allowed list. "
            f"Allowed: {', '.join(allowed_hosts)}"
        )

    return host
