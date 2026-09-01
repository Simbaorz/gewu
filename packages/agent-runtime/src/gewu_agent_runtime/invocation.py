"""Typed, transport-neutral resource invocation selected by a host application."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InvocationTargetKind(StrEnum):
    """Runtime resource kinds that can be selected before a turn starts."""

    SKILL = "skill"
    SCENE = "scene"


class InvocationTarget(BaseModel):
    """Opaque resource selection already normalized by the host application."""

    model_config = ConfigDict(frozen=True)

    kind: InvocationTargetKind
    resource_id: str = Field(default="", max_length=255)
    name: str = Field(default="", max_length=255)
    arguments: str = ""

    @model_validator(mode="after")
    def require_resource_reference(self) -> InvocationTarget:
        """Require an opaque ID or model-visible resource name."""

        if not self.resource_id.strip() and not self.name.strip():
            raise ValueError("Invocation target requires resource_id or name.")
        return self
