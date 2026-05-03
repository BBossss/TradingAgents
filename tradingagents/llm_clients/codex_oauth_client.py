from __future__ import annotations

from typing import Any

from .base_client import BaseLLMClient
from .codex_credentials import CodexCredentialError, load_codex_access_token
from .validators import validate_model


class _ReadOnlyCodexAuthStore:
    """Access-token-only auth store for langchain-codex-oauth."""

    def __init__(self, auth_path: str | None = None) -> None:
        self.auth_path = auth_path

    def load(self):
        try:
            from codex_oauth.exceptions import NotAuthenticatedError
            from codex_oauth.store import OAuthCredentials
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langchain-codex-oauth is required for llm_provider='codex_oauth'. "
                "Install the optional codex-oauth extra."
            ) from exc

        try:
            token = load_codex_access_token(self.auth_path)
        except CodexCredentialError as exc:
            raise NotAuthenticatedError(
                f"{exc} Authenticate with Codex/Codex App first, or use the "
                "default OpenAI API-key provider."
            ) from exc

        return OAuthCredentials(
            access=token.access_token,
            refresh="",
            expires=token.expires_at_ms,
            account_id=token.account_id,
        )

    def save(self, _creds) -> None:
        raise RuntimeError("TradingAgents does not write Codex OAuth credentials.")

    def delete(self) -> None:
        raise RuntimeError("TradingAgents does not delete Codex OAuth credentials.")


def _stream_aggregating_chat_model(chat_model_cls):
    """Adapt ChatCodexOAuth invoke() to the backend's streaming-only text shape."""

    class StreamAggregatingChatCodexOAuth(chat_model_cls):
        def _generate(
            self,
            messages,
            stop=None,
            run_manager=None,
            **kwargs,
        ):
            from langchain_core.messages.utils import message_chunk_to_message
            from langchain_core.outputs import ChatGeneration, ChatResult

            generation = None
            for chunk in self._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            ):
                generation = chunk if generation is None else generation + chunk

            if generation is None:
                return super()._generate(
                    messages,
                    stop=stop,
                    run_manager=run_manager,
                    **kwargs,
                )

            message = message_chunk_to_message(generation.message)
            return ChatResult(generations=[ChatGeneration(message=message)])

    return StreamAggregatingChatCodexOAuth


class CodexOAuthClient(BaseLLMClient):
    """Client for Codex/ChatGPT OAuth models using Codex-owned credentials."""

    def get_llm(self) -> Any:
        """Return configured ChatCodexOAuth instance."""
        self.warn_if_unknown_model()
        try:
            from langchain_codex_oauth import ChatCodexOAuth
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "langchain-codex-oauth is required for llm_provider='codex_oauth'. "
                "Install the optional codex-oauth extra. TradingAgents does not "
                "provide its own OAuth login command."
            ) from exc

        llm_kwargs = {
            "model": self.model,
            "auth_store": _ReadOnlyCodexAuthStore(self.kwargs.get("codex_auth_path")),
        }

        if self.base_url:
            llm_kwargs["base_url"] = self.base_url

        for key in (
            "timeout",
            "max_retries",
            "reasoning_effort",
            "max_tokens",
            "temperature",
            "callbacks",
        ):
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        StreamingChatCodexOAuth = _stream_aggregating_chat_model(ChatCodexOAuth)
        return StreamingChatCodexOAuth(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for codex_oauth."""
        return validate_model("codex_oauth", self.model)
