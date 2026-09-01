"""Provider-specific ChatModel construction."""

from __future__ import annotations

import httpx

from gewu_agent_runtime.adapters.llm.anthropic_chat import AnthropicChatModel
from gewu_agent_runtime.adapters.llm.openai_chat import OpenAIChatModel
from gewu_agent_runtime.adapters.llm.unicom_chat import ChinaUnicomOpenServiceChatModel
from gewu_agent_runtime.llm import ChatModel, ModelRuntimeConfig


def build_chat_model(
    config: ModelRuntimeConfig,
    *,
    http_client: httpx.AsyncClient | None = None,
    connect_timeout_seconds: float = 5.0,
    pool_timeout_seconds: float = 5.0,
) -> ChatModel:
    """Build a concrete model from one authorized provider configuration."""

    if config.provider == "openai":
        return OpenAIChatModel.from_runtime_config(
            config,
            http_client=http_client,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
        )
    if config.provider == "anthropic":
        return AnthropicChatModel.from_runtime_config(
            config,
            http_client=http_client,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
        )
    if config.provider == "unicom":
        return ChinaUnicomOpenServiceChatModel.from_runtime_config(
            config,
            http_client=http_client,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
        )
    raise ValueError(f"Unsupported runtime LLM provider: {config.provider}")


class DefaultProviderChatModelFactory:
    """Build turn-scoped adapters over one shared outbound HTTP pool."""

    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 5.0,
    ) -> None:
        self._http_client = http_client or httpx.AsyncClient()
        self._owns_http_client = http_client is None
        self._connect_timeout_seconds = connect_timeout_seconds
        self._pool_timeout_seconds = pool_timeout_seconds
        self._closed = False

    def create(self, config: ModelRuntimeConfig) -> ChatModel:
        """Create one provider adapter, rejecting use after shutdown."""

        if self._closed:
            raise RuntimeError("LLM provider factory is closed.")
        return build_chat_model(
            config,
            http_client=self._http_client,
            connect_timeout_seconds=self._connect_timeout_seconds,
            pool_timeout_seconds=self._pool_timeout_seconds,
        )

    async def aclose(self) -> None:
        """Close an internally owned pool and stop creating adapters."""

        if self._closed:
            return
        self._closed = True
        if self._owns_http_client:
            await self._http_client.aclose()
