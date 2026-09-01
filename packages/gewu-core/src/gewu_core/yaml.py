"""Structured YAML helpers that preserve top-level mapping entry order."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import yaml
from yaml.nodes import MappingNode


class YamlMappingIssue(StrEnum):
    """Machine-readable failures while decoding one YAML mapping."""

    EMPTY = "empty"
    ROOT_NOT_MAPPING = "root_not_mapping"
    NON_STRING_KEY = "non_string_key"
    INVALID = "invalid"


class YamlMappingError(ValueError):
    """YAML mapping failure with a stable category and parser detail."""

    def __init__(self, issue: YamlMappingIssue, detail: str = "") -> None:
        super().__init__(detail or issue.value)
        self.issue = issue
        self.detail = detail


def load_yaml_mapping_entries(content: str) -> list[tuple[str, Any]]:
    """Load top-level entries without collapsing duplicate mapping keys."""

    loader = yaml.SafeLoader(content)
    try:
        node = loader.get_single_node()
        if node is None:
            raise YamlMappingError(YamlMappingIssue.EMPTY)
        if not isinstance(node, MappingNode):
            raise YamlMappingError(YamlMappingIssue.ROOT_NOT_MAPPING)
        entries: list[tuple[str, Any]] = []
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=True)
            if not isinstance(key, str):
                raise YamlMappingError(YamlMappingIssue.NON_STRING_KEY)
            entries.append((key, loader.construct_object(value_node, deep=True)))
        return entries
    except yaml.YAMLError as exc:
        raise YamlMappingError(YamlMappingIssue.INVALID, str(exc)) from exc
    finally:
        loader.dispose()


def dump_yaml_mapping(value: dict[str, Any]) -> str:
    """Serialize a mapping using stable, human-readable source order."""

    return yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()
