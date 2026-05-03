"""Read-only access to Codex-owned ChatGPT OAuth credentials."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CodexCredentialError(RuntimeError):
    """Raised when Codex credentials cannot be consumed safely."""


@dataclass(frozen=True)
class CodexAccessToken:
    """Access-token-only view of Codex credentials."""

    access_token: str
    account_id: str
    expires_at_ms: int

    @property
    def redacted(self) -> dict[str, object]:
        return {
            "has_access_token": bool(self.access_token),
            "account_id": self.account_id,
            "expires_at_ms": self.expires_at_ms,
        }


def default_codex_auth_path() -> Path:
    """Return the Codex-owned auth file path, optionally overridden for tests."""
    override = os.environ.get("TRADINGAGENTS_CODEX_AUTH_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex" / "auth.json"


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise CodexCredentialError("Codex access token is not a JWT.")

    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except Exception as exc:
        raise CodexCredentialError("Codex access token payload is invalid.") from exc

    if not isinstance(data, dict):
        raise CodexCredentialError("Codex access token payload is not an object.")
    return data


def _claim_account_id(payload: dict[str, Any]) -> str | None:
    claim = payload.get("https://api.openai.com/auth")
    if not isinstance(claim, dict):
        return None
    account_id = claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def load_codex_access_token(
    auth_path: str | Path | None = None,
    *,
    now_s: int | None = None,
    min_ttl_s: int = 60,
) -> CodexAccessToken:
    """Load a usable Codex access token without consuming refresh credentials.

    This function intentionally reads only the access token and account id from
    Codex-owned auth state. It never refreshes, saves, deletes, or copies
    credentials. Expired or expiry-unknown tokens fail closed.
    """
    path = Path(auth_path).expanduser() if auth_path is not None else default_codex_auth_path()
    if not path.exists():
        raise CodexCredentialError(f"Codex auth file not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CodexCredentialError(f"Codex auth file is unreadable: {path}") from exc

    if not isinstance(data, dict):
        raise CodexCredentialError("Codex auth file is not a JSON object.")

    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        raise CodexCredentialError("Codex auth file has no tokens object.")

    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id")
    if not isinstance(access_token, str) or not access_token:
        raise CodexCredentialError("Codex auth file has no access token.")
    if not isinstance(account_id, str) or not account_id:
        raise CodexCredentialError("Codex auth file has no account id.")

    payload = _decode_jwt_payload(access_token)
    exp = payload.get("exp")
    if not isinstance(exp, (int, float)):
        raise CodexCredentialError("Codex access token has no expiry claim.")

    claim_account_id = _claim_account_id(payload)
    if not claim_account_id:
        raise CodexCredentialError("Codex access token has no ChatGPT account claim.")
    if claim_account_id != account_id:
        raise CodexCredentialError("Codex token account id does not match auth file.")

    now = int(time.time()) if now_s is None else now_s
    if int(exp) - now <= min_ttl_s:
        raise CodexCredentialError("Codex access token is expired or too close to expiry.")

    return CodexAccessToken(
        access_token=access_token,
        account_id=account_id,
        expires_at_ms=int(exp) * 1000,
    )
