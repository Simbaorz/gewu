"""Subscriber capability preparation and Runtime handoff contracts."""

from __future__ import annotations

from typing import Protocol, TypeVar

from gewu_agent_runtime import (
    AgentRuntime,
    PreparedAgentTurn,
    PrincipalRef,
    TurnSession,
)
from gewu_core import ApplicationError, ApplicationErrorKind

CommandT = TypeVar("CommandT", contravariant=True)


class SubscriberRuntimeProvider(Protocol[CommandT]):
    """Prepare one complete, authorized Runtime turn for a subscriber command."""

    async def prepare_turn(
        self,
        principal: PrincipalRef,
        command: CommandT,
    ) -> PreparedAgentTurn:
        """Bind subscriber policy and capability preparation before Runtime handoff."""


async def prepare_and_start_turn[PreparedCommandT](
    runtime: AgentRuntime,
    provider: SubscriberRuntimeProvider[PreparedCommandT],
    *,
    principal: PrincipalRef,
    command: PreparedCommandT,
) -> TurnSession:
    """Start an authorized provider turn while binding it to the authenticated invoker."""

    prepared = await provider.prepare_turn(principal, command)
    if prepared.request.invoker != principal:
        raise ApplicationError(
            ApplicationErrorKind.FORBIDDEN,
            "Prepared turn invoker does not match the authenticated principal.",
        )
    return await runtime.start_prepared_turn(prepared)
