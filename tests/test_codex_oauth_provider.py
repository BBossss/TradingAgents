import base64
import builtins
import json
import sys
import types
from dataclasses import dataclass

import pytest

from tradingagents.llm_clients.factory import _OPENAI_COMPATIBLE, create_llm_client
from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS, get_known_models
from tradingagents.llm_clients.openai_client import OpenAIClient
from tradingagents.llm_clients.validators import validate_model


def _jwt(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    return f"header.{body}.sig"


def test_codex_oauth_catalog_and_validation_are_catalog_driven():
    assert "codex_oauth" in MODEL_OPTIONS
    assert "codex_oauth" in get_known_models()
    assert validate_model("codex_oauth", "gpt-5.5")


def test_codex_oauth_is_separate_from_openai_compatible_branch():
    assert "codex_oauth" not in _OPENAI_COMPATIBLE
    assert isinstance(create_llm_client("openai", "gpt-5.5"), OpenAIClient)


def test_create_codex_oauth_client_uses_lazy_dependency_and_read_only_store(
    monkeypatch, tmp_path
):
    auth = tmp_path / "auth.json"
    token = _jwt(
        {
            "exp": 2_000_000_000,
            "https://api.openai.com/auth": {"chatgpt_account_id": "acct_123"},
        }
    )
    auth.write_text(
        json.dumps({"tokens": {"access_token": token, "account_id": "acct_123"}}),
        encoding="utf-8",
    )

    class FakeChatCodexOAuth:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _stream(self, *_args, **_kwargs):
            from langchain_core.messages import AIMessageChunk
            from langchain_core.outputs import ChatGenerationChunk

            yield ChatGenerationChunk(message=AIMessageChunk(content="Hello"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="!"))

    @dataclass(frozen=True)
    class FakeOAuthCredentials:
        access: str
        refresh: str
        expires: int
        account_id: str

    class FakeNotAuthenticatedError(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "langchain_codex_oauth",
        types.SimpleNamespace(ChatCodexOAuth=FakeChatCodexOAuth),
    )
    monkeypatch.setitem(
        sys.modules,
        "codex_oauth.store",
        types.SimpleNamespace(OAuthCredentials=FakeOAuthCredentials),
    )
    monkeypatch.setitem(
        sys.modules,
        "codex_oauth.exceptions",
        types.SimpleNamespace(NotAuthenticatedError=FakeNotAuthenticatedError),
    )

    client = create_llm_client(
        "codex_oauth",
        "gpt-5.5",
        codex_auth_path=str(auth),
        reasoning_effort="medium",
    )
    llm = client.get_llm()
    creds = llm.kwargs["auth_store"].load()

    assert isinstance(llm, FakeChatCodexOAuth)
    assert llm.kwargs["reasoning_effort"] == "medium"
    assert creds.access == token
    assert creds.refresh == ""
    assert creds.account_id == "acct_123"
    with pytest.raises(RuntimeError, match="does not write"):
        llm.kwargs["auth_store"].save(creds)
    result = llm._generate([])
    assert result.generations[0].message.content == "Hello!"


def test_missing_codex_oauth_dependency_error_has_no_tradingagents_auth_command(
    monkeypatch,
):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "langchain_codex_oauth":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    client = create_llm_client("codex_oauth", "gpt-5.5")

    with pytest.raises(ModuleNotFoundError) as exc:
        client.get_llm()

    assert "tradingagents auth" not in str(exc.value).lower()
