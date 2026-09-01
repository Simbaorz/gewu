"""Neutral subscriber and principal references."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PrincipalType(StrEnum):
    """Kinds of authenticated principals accepted by a host application."""

    USER = "user"
    SERVICE = "service"


class PrincipalRef(BaseModel):
    """Stable identity reference without organization or authorization semantics."""

    model_config = ConfigDict(frozen=True)

    subscriber_id: str = Field(min_length=1, max_length=128)
    principal_id: str = Field(min_length=1, max_length=128)
    principal_type: PrincipalType
