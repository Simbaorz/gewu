"""Executable dependency and business-neutrality rules for agent-runtime."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "packages" / "agent-runtime" / "src" / "gewu_agent_runtime"
PACKAGE_PROJECT = PACKAGE.parents[1] / "pyproject.toml"
BUSINESS_NEUTRAL_PACKAGES = (PACKAGE,)
BANNED_BUSINESS_TERMS = {
    "accessscope",
    "tenant",
    "province",
    "city",
    "team",
    "touchpoint",
}
BANNED_SUBSCRIBER_CONVENTIONS = {
    "/workspace/private",
    "/workspace/shared",
    "private workspace",
    "shared workspace",
    "bot.md",
    "user.md",
    "botinfo",
    "userinfo",
}
EXTERNAL_ADAPTER_MODULES = {"redis", "sqlalchemy"}
MODEL_PROVIDER_MODULES = {
    "anthropic",
    "boto3",
    "google.generativeai",
    "litellm",
    "ollama",
    "openai",
}


def test_runtime_and_generic_host_have_no_organization_business_vocabulary() -> None:
    violations: list[str] = []
    for source_root in BUSINESS_NEUTRAL_PACKAGES:
        for path in source_root.rglob("*.py"):
            content = path.read_text(encoding="utf-8").lower()
            for term in BANNED_BUSINESS_TERMS:
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                    content,
                ):
                    relative = path.relative_to(source_root.parent)
                    violations.append(f"{relative}: {term}")
    assert not violations, "Generic Runtime packages contain organization terms:\n" + "\n".join(
        violations
    )


def test_runtime_source_has_no_subscriber_workspace_or_profile_conventions() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        content = path.read_text(encoding="utf-8").lower()
        for convention in BANNED_SUBSCRIBER_CONVENTIONS:
            if convention in content:
                violations.append(f"{path.relative_to(PACKAGE)}: {convention}")
    assert not violations, "Runtime contains Subscriber conventions:\n" + "\n".join(violations)


def test_runtime_workspace_contract_exposes_only_logical_paths() -> None:
    contract = PACKAGE / "workspace" / "contracts.py"
    tree = ast.parse(contract.read_text(encoding="utf-8"), filename=str(contract))
    path_annotations = [
        node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == "Path"
    ]

    assert not path_annotations
    assert "pathlib" not in contract.read_text(encoding="utf-8")


def test_runtime_does_not_register_subscriber_business_data_tool() -> None:
    allowed = {
        Path("builtins/business_data.py"),
        Path("builtins/__init__.py"),
    }
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "business_data_tool"
            ):
                violations.append(f"{relative}:{node.lineno}")

    assert not violations, "Runtime registers a subscriber tool:\n" + "\n".join(violations)


def test_core_does_not_import_external_adapters() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] == "adapters":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for module in names:
                root = module.split(".", 1)[0]
                if root in EXTERNAL_ADAPTER_MODULES or module.startswith(
                    "gewu_agent_runtime.adapters"
                ):
                    violations.append(f"{relative}: {module}")
    assert not violations, "Core imports adapter technology:\n" + "\n".join(violations)


def test_runtime_has_no_concrete_model_provider_imports() -> None:
    violations: list[str] = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[:2] == ("adapters", "llm"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for module in names:
                if any(
                    module == provider or module.startswith(f"{provider}.")
                    for provider in MODEL_PROVIDER_MODULES
                ):
                    violations.append(f"{relative}: {module}")
    assert not violations, "Runtime imports a concrete model provider:\n" + "\n".join(violations)


def test_runtime_is_one_core_package_with_mysql_and_redis_extras() -> None:
    project = tomllib.loads(PACKAGE_PROJECT.read_text(encoding="utf-8"))["project"]
    core = {_dependency_name(value) for value in project["dependencies"]}
    extras = {
        name: {_dependency_name(value) for value in values}
        for name, values in project["optional-dependencies"].items()
    }

    assert project["name"] == "gewu-agent-runtime"
    assert core == {
        "gewu-core",
        "google-re2",
        "pydantic",
        "tiktoken",
    }
    assert set(extras) == {"mysql", "redis", "host", "providers", "all"}
    assert extras["mysql"] == {
        "aiomysql",
        "greenlet",
        "sqlalchemy",
    }
    assert extras["redis"] == {"redis"}
    assert extras["host"] == {"httpx"}
    assert extras["providers"] == {"anthropic", "httpx", "openai"}
    assert extras["all"] == (
        extras["mysql"] | extras["redis"] | extras["host"] | extras["providers"]
    )
    assert not core & extras["all"]


def test_runtime_uses_core_for_shared_ids_and_time() -> None:
    assert not (PACKAGE / "_ids.py").exists()
    assert not (PACKAGE / "_time.py").exists()

    imports = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    assert "from gewu_core.ids import" in imports
    assert "from gewu_core.time import" in imports


def _dependency_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()
