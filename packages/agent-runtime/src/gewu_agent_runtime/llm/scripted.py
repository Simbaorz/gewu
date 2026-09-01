"""Deterministic chat model used by tests and local examples."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence

from gewu_agent_runtime.llm.contracts import Message, ModelStreamChunk, ModelTool


class ScriptedChatModel:
    """Return a preconfigured sequence of response streams."""

    model_ref = "scripted:test"
    provider = "scripted"
    model_name = "scripted"
    support_vision = True
    context_window = 1_000_000

    def __init__(self, responses: Iterable[Iterable[ModelStreamChunk]]) -> None:
        """Initialize model responses consumed one request at a time."""

        self._responses = [tuple(response) for response in responses]
        self.requests: list[tuple[tuple[Message, ...], tuple[ModelTool, ...]]] = []

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Yield the next scripted response."""

        self.requests.append((tuple(messages), tuple(tools)))
        if not self._responses:
            raise RuntimeError("Scripted model has no response remaining.")
        for chunk in self._responses.pop(0):
            yield chunk
