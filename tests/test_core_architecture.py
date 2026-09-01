"""Executable dependency rules for the shared core package."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "packages" / "gewu-core" / "src" / "gewu_core"
PROJECT = PACKAGE.parents[1] / "pyproject.toml"


def test_core_does_not_depend_on_runtime_or_applications() -> None:
    banned_roots = {
        "gewu_agent_runtime",
        "online_consultation",
        "online_consultation_admin_api",
        "online_consultation_api",
        "online_consultation_platform",
    }
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                if module.split(".", 1)[0] in banned_roots:
                    violations.append(f"{path.relative_to(PACKAGE)}: {module}")
    assert not violations, "Core imports an application package:\n" + "\n".join(violations)


def test_core_keeps_infrastructure_stacks_in_optional_extras() -> None:
    project = tomllib.loads(PROJECT.read_text(encoding="utf-8"))["project"]
    base = {_dependency_name(value) for value in project["dependencies"]}
    extras = {
        name: {_dependency_name(value) for value in values}
        for name, values in project["optional-dependencies"].items()
    }

    assert project["name"] == "gewu-core"
    assert base == {"cryptography", "filelock", "pydantic", "python-dotenv", "pyyaml"}
    assert set(extras) == {"apollo", "database", "redis", "http", "all"}
    assert extras["apollo"] == {"httpx"}
    assert extras["database"] == {"aiomysql", "aiosqlite", "greenlet", "sqlalchemy"}
    assert extras["redis"] == {"redis"}
    assert extras["http"] == {"fastapi", "starlette", "uvicorn"}
    assert extras["all"] == (
        extras["apollo"] | extras["database"] | extras["redis"] | extras["http"]
    )
    assert not base & extras["all"]


def test_core_http_does_not_encode_subscriber_names_or_route_policy() -> None:
    http_root = PACKAGE / "http"
    violations: list[str] = []
    for path in http_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        if "subscriber" in source.lower() or "/api/" in source or "bootstrap.mode" in source:
            violations.append(path.relative_to(PACKAGE).as_posix())

    assert not violations, "Core HTTP contains subscriber policy: " + ", ".join(violations)


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
