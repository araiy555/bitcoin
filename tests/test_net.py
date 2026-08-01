"""TLS setup: verification stays on, and cert failures explain themselves."""

import ssl

from jsboard.net import describe_tls_error, make_session, ssl_context


class TestSSLContext:
    def test_verification_is_never_disabled(self):
        ctx = ssl_context()

        assert ctx.verify_mode is ssl.CERT_REQUIRED
        assert ctx.check_hostname is True

    def test_roots_are_actually_loaded(self):
        """An empty CA store is the bug this exists to prevent."""
        ctx = ssl_context()

        assert len(ctx.get_ca_certs()) > 0

    def test_it_uses_the_bundled_roots_when_available(self):
        import certifi

        bundled = ssl.create_default_context(cafile=certifi.where())

        assert len(ssl_context().get_ca_certs()) == len(bundled.get_ca_certs())

    def test_it_falls_back_when_certifi_is_missing(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def no_certifi(name, *args, **kwargs):
            if name == "certifi":
                raise ImportError("no certifi")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_certifi)

        ctx = ssl_context()

        # Still a verifying context, just using the system store.
        assert ctx.verify_mode is ssl.CERT_REQUIRED


class TestSession:
    async def test_session_carries_the_verifying_context(self):
        session = make_session()
        try:
            assert session.trust_env is True
            assert session.connector is not None
        finally:
            await session.close()

    async def test_caller_can_override(self):
        session = make_session(trust_env=False)
        try:
            assert session.trust_env is False
        finally:
            await session.close()


class TestErrorHint:
    def test_certificate_failures_are_recognised(self):
        exc = ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "self-signed certificate in certificate chain"
        )

        hint = describe_tls_error(exc)

        assert hint is not None
        assert "certifi" in hint

    def test_unrelated_errors_get_no_hint(self):
        assert describe_tls_error(ConnectionResetError(104, "Connection reset by peer")) is None
        assert describe_tls_error(TimeoutError("too slow")) is None

    def test_it_matches_the_wrapped_aiohttp_form(self):
        """aiohttp reports the cause inside a connector error string."""

        class ClientConnectorCertificateError(Exception):
            pass

        exc = ClientConnectorCertificateError(
            "Cannot connect to host api.binance.com:443 ssl:True "
            "[SSLCertVerificationError: (1, '[SSL: CERTIFICATE_VERIFY_FAILED] ...')]"
        )

        assert describe_tls_error(exc) is not None
