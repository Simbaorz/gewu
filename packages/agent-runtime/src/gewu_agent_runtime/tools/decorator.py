"""Decorator for concise custom tool definitions."""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, Literal, get_args, get_origin, get_type_hints

from gewu_agent_runtime.tools.contracts import (
    PersistencePolicy,
    Tool,
    ToolContext,
    ToolResult,
)


def with_tool_description(value: Tool, description: str) -> Tool:
    """Return a Tool with subscriber-owned model text and matching argument docs."""

    normalized = description.strip()
    argument_descriptions = _extract_argument_descriptions(normalized)
    input_schema = deepcopy(value.input_schema)
    properties = input_schema.get("properties")
    if isinstance(properties, dict):
        for name, schema in properties.items():
            if not isinstance(schema, dict):
                continue
            argument_description = argument_descriptions.get(name)
            if argument_description is None:
                schema.pop("description", None)
            else:
                schema["description"] = argument_description
    return value.model_copy(
        update={
            "description": normalized,
            "input_schema": input_schema,
        }
    )


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    writes_workspace: bool = False,
    allow_parallel: bool = False,
    retry_on_failure: bool = True,
    max_retries: int = 2,
    persistence_policy: PersistencePolicy = PersistencePolicy.FULL,
    trace_result: bool = True,
) -> Callable[[Callable[..., ToolResult | Awaitable[ToolResult]]], Tool]:
    """Build a Tool from an annotated function without global registration."""

    def decorate(function: Callable[..., ToolResult | Awaitable[ToolResult]]) -> Tool:
        signature = inspect.signature(function)
        hints = get_type_hints(function)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in {"self", "runtime"}:
                continue
            properties[parameter_name] = _schema_for(
                hints.get(parameter_name, parameter.annotation)
            )
            argument_descriptions = _extract_argument_descriptions(
                description or function.__doc__ or ""
            )
            if parameter_name in argument_descriptions:
                properties[parameter_name]["description"] = argument_descriptions[parameter_name]
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)

        @functools.wraps(function)
        async def invoke(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
            kwargs = {
                key: _coerce_value(
                    key,
                    value,
                    hints.get(key, signature.parameters[key].annotation),
                )
                for key, value in arguments.items()
                if key in signature.parameters and key not in {"self", "runtime"}
            }
            if "runtime" in signature.parameters:
                kwargs["runtime"] = context
            result = function(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        return Tool(
            name=name or function.__name__,
            description=(description or function.__doc__ or f"Execute {function.__name__}").strip(),
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
            function=invoke,
            category=category,
            writes_workspace=writes_workspace,
            allow_parallel=allow_parallel,
            retry_on_failure=retry_on_failure,
            max_retries=max_retries,
            persistence_policy=persistence_policy,
            trace_result=trace_result,
        )

    return decorate


def _schema_for(annotation: object) -> dict[str, Any]:
    """Return a small JSON Schema fragment for a Python annotation."""

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    args = get_args(annotation)
    if args and type(None) in args:
        non_none = tuple(item for item in args if item is not type(None))
        if len(non_none) == 1:
            return _schema_for(non_none[0])
    if origin is Literal:
        values = list(args)
        value_schema = (
            _schema_for(type(values[0]))
            if values and all(type(value) is type(values[0]) for value in values)
            else {}
        )
        return {**value_schema, "enum": values}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if origin is list:
        return {"type": "array", "items": _schema_for(args[0]) if args else {}}
    if origin is dict:
        value_schema = _schema_for(args[1]) if len(args) == 2 else {}
        return {"type": "object", "additionalProperties": value_schema}
    return {"type": "string"}


def _coerce_value(name: str, value: Any, annotation: object) -> Any:
    """Coerce primitive provider arguments used by reference tools."""

    if value is None:
        return None
    args = get_args(annotation)
    if args and type(None) in args:
        non_none = tuple(item for item in args if item is not type(None))
        if len(non_none) == 1:
            annotation = non_none[0]
    if annotation is int:
        if isinstance(value, bool):
            raise ValueError(f"Tool argument '{name}' must be an integer.")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ValueError(f"Tool argument '{name}' must be an integer.") from exc
    return value


def _extract_argument_descriptions(docstring: str) -> dict[str, str]:
    """Extract Google-style argument descriptions for model-visible schemas."""

    cleaned = inspect.cleandoc(docstring)
    descriptions: dict[str, str] = {}
    current_name = ""
    current_lines: list[str] = []
    in_args = False
    pattern = re.compile(r"^([A-Za-z_]\w*):\s*(.*)$")
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if in_args and stripped in {"Returns:", "Raises:", "Examples:", "Notes:", "Usage:"}:
            break
        if not in_args:
            continue
        match = pattern.match(stripped)
        if match:
            if current_name:
                descriptions[current_name] = " ".join(current_lines).strip()
            current_name = match.group(1)
            current_lines = [match.group(2).strip()]
        elif current_name and stripped:
            current_lines.append(stripped)
    if current_name:
        descriptions[current_name] = " ".join(current_lines).strip()
    return descriptions
