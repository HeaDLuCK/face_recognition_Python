from urllib.parse import urlsplit, urlunsplit


def redact_url_credentials(value: str) -> str:
    """Mask URL user information without changing host, path, or query details."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "<redacted-url>"
    if "@" not in parsed.netloc:
        return value
    host = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, "***:***@" + host, parsed.path, parsed.query, parsed.fragment))
