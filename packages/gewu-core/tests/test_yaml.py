"""Structured YAML helper behavior."""

from __future__ import annotations

import pytest

from gewu_core.yaml import (
    YamlMappingError,
    YamlMappingIssue,
    dump_yaml_mapping,
    load_yaml_mapping_entries,
)


def test_mapping_loader_preserves_duplicate_entries_in_source_order() -> None:
    assert load_yaml_mapping_entries("name: first\nname: second\n") == [
        ("name", "first"),
        ("name", "second"),
    ]


@pytest.mark.parametrize(
    ("content", "issue"),
    [
        ("", YamlMappingIssue.EMPTY),
        ("- item\n", YamlMappingIssue.ROOT_NOT_MAPPING),
        ("1: value\n", YamlMappingIssue.NON_STRING_KEY),
        ("[invalid", YamlMappingIssue.INVALID),
    ],
)
def test_mapping_loader_classifies_failures(content: str, issue: YamlMappingIssue) -> None:
    with pytest.raises(YamlMappingError) as raised:
        load_yaml_mapping_entries(content)
    assert raised.value.issue is issue


def test_mapping_dump_preserves_insertion_order() -> None:
    assert dump_yaml_mapping({"name": "报告", "enabled": True}) == ("name: 报告\nenabled: true")
