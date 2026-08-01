"""Shared outbound HTTP/WS setup.

Python verifies TLS against whatever CA store its OpenSSL build was pointed
at, and that store is not always populated. A Homebrew Python on macOS looks
for `/opt/homebrew/etc/openssl@3/cert.pem`, which does not exist unless the
`ca-certificates` formula happens to be installed — and with an empty store
every chain fails as "self-signed certificate in certificate chain", even a
perfectly ordinary DigiCert one. Meanwhile `curl` works, because the system
curl uses the macOS keychain instead, which makes the failure look like a
network problem when it is a configuration one.

Depending on the host's OpenSSL layout is the actual mistake. `certifi` ships
Mozilla's CA bundle with the package, so verification behaves the same on
every machine. This is what `requests` and `httpx` do by default; aiohttp and
`websockets` do not, so we pass the context in explicitly.

Verification is never disabled here. A certificate error means the identity
of the far end could not be established, and turning the check off does not
fix that — it only stops us hearing about it.
"""

from __future__ import annotations

import ssl

import aiohttp


def ssl_context() -> ssl.SSLContext:
    """A verifying context backed by certifi when it is available."""
    try:
        import certifi
    except ImportError:
        # No bundled roots; fall back to the system store and let it speak
        # for itself. Still verifying, just less portable.
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def make_session(**kwargs) -> aiohttp.ClientSession:
    """An aiohttp session that verifies against the bundled roots.

    `trust_env` stays on so HTTPS_PROXY and friends are still honoured.
    """
    kwargs.setdefault("trust_env", True)
    kwargs.setdefault("connector", aiohttp.TCPConnector(ssl=ssl_context()))
    return aiohttp.ClientSession(**kwargs)


def describe_tls_error(exc: BaseException) -> str | None:
    """Turn a TLS failure into something the user can act on.

    Returns None when the error is not certificate-related, so callers can
    fall through to their normal reporting.
    """
    text = f"{type(exc).__name__}: {exc}"
    if "CERTIFICATE_VERIFY_FAILED" not in text and "SSLCertVerificationError" not in text:
        return None

    return (
        "TLS証明書の検証に失敗しました。\n"
        "  通信が届いていないのではなく、証明書を検証するためのルート証明書が\n"
        "  見つかっていない可能性が高いです。次で確認できます:\n"
        "    .venv/bin/pip install certifi\n"
        "  それでも直らない場合、間に何かが入っていないか確認してください:\n"
        "    openssl s_client -connect api.binance.com:443 </dev/null 2>/dev/null | grep issuer="
    )
