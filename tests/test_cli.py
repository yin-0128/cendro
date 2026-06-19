"""CLI tests — language detection and the sensitive-file guard on `cendro review`."""

from __future__ import annotations

from cendro import cli


def test_guess_language():
    assert cli._guess_language("foo.py") == "python"
    assert cli._guess_language("a/b/foo.tsx") == "typescript"
    assert cli._guess_language("README.unknown") == "text"


def test_sensitive_reason_flags_secrets():
    assert cli._sensitive_reason(".env")
    assert cli._sensitive_reason("config/.env.local")
    assert cli._sensitive_reason("/home/u/.ssh/id_rsa")
    assert cli._sensitive_reason("certs/server.pem")
    assert cli._sensitive_reason("/home/u/.aws/credentials")


def test_sensitive_reason_allows_normal_code():
    assert cli._sensitive_reason("main.py") is None
    assert cli._sensitive_reason("src/app.ts") is None


def test_review_refuses_sensitive_file_without_force(tmp_path, capsys):
    secret = tmp_path / ".env"
    secret.write_text("API_KEY=super-secret")
    rc = cli.main(["review", str(secret)])
    assert rc == 1
    assert "Refusing" in capsys.readouterr().err
