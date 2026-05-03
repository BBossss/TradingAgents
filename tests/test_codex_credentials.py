import base64
import json
from pathlib import Path

import pytest

from tradingagents.llm_clients.codex_credentials import (
    CodexCredentialError,
    load_codex_access_token,
)


def _jwt(payload: dict) -> str:
    def enc(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return f"{enc({'alg': 'none'})}.{enc(payload)}.sig"


def _write_auth(path: Path, *, token: str | None, account_id: str | None, refresh=True):
    tokens = {}
    if token is not None:
        tokens["access_token"] = token
    if account_id is not None:
        tokens["account_id"] = account_id
    if refresh:
        tokens["refresh_token"] = "must-not-be-required"
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": tokens}), encoding="utf-8")


def test_load_codex_access_token_returns_redacted_metadata_without_refresh(tmp_path):
    auth = tmp_path / "auth.json"
    token = _jwt(
        {
            "exp": 2_000,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        }
    )
    _write_auth(auth, token=token, account_id="acct_123", refresh=False)

    before = auth.stat().st_mtime_ns
    result = load_codex_access_token(auth, now_s=1_000)
    after = auth.stat().st_mtime_ns

    assert result.access_token == token
    assert result.account_id == "acct_123"
    assert result.expires_at_ms == 2_000_000
    assert result.redacted == {
        "has_access_token": True,
        "account_id": "acct_123",
        "expires_at_ms": 2_000_000,
    }
    assert after == before


def test_load_codex_access_token_fails_closed_for_missing_file(tmp_path):
    with pytest.raises(CodexCredentialError, match="not found"):
        load_codex_access_token(tmp_path / "missing.json", now_s=1_000)


def test_load_codex_access_token_fails_closed_for_incompatible_file(tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"tokens": {"account_id": "acct_123"}}), encoding="utf-8")

    with pytest.raises(CodexCredentialError, match="no access token"):
        load_codex_access_token(auth, now_s=1_000)


def test_load_codex_access_token_fails_closed_when_expiry_unknown(tmp_path):
    auth = tmp_path / "auth.json"
    token = _jwt({"https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"}})
    _write_auth(auth, token=token, account_id="acct_123")

    with pytest.raises(CodexCredentialError, match="no expiry"):
        load_codex_access_token(auth, now_s=1_000)


def test_load_codex_access_token_fails_closed_without_account_claim(tmp_path):
    auth = tmp_path / "auth.json"
    token = _jwt({"exp": 2_000})
    _write_auth(auth, token=token, account_id="acct_123")

    with pytest.raises(CodexCredentialError, match="no ChatGPT account claim"):
        load_codex_access_token(auth, now_s=1_000)


def test_load_codex_access_token_fails_closed_when_expired(tmp_path):
    auth = tmp_path / "auth.json"
    token = _jwt(
        {
            "exp": 1_010,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        }
    )
    _write_auth(auth, token=token, account_id="acct_123")

    with pytest.raises(CodexCredentialError, match="expired"):
        load_codex_access_token(auth, now_s=1_000, min_ttl_s=60)


def test_load_codex_access_token_rejects_account_mismatch(tmp_path):
    auth = tmp_path / "auth.json"
    token = _jwt(
        {
            "exp": 2_000,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_token"},
        }
    )
    _write_auth(auth, token=token, account_id="acct_file")

    with pytest.raises(CodexCredentialError, match="does not match"):
        load_codex_access_token(auth, now_s=1_000)
