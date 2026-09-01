# Gewu Core Monorepo Guidelines

This repository is a business-neutral Agent Runtime monorepo built with `uv workspace`.

## Workspace

- `packages/gewu-core/`: Configuration, logging, database, Redis, HTTP, and shared infrastructure.
- `packages/agent-runtime/`: Business-neutral Agent Runtime, tools, workspaces, skills, scenes, and execution coordination.
- Subscribers may depend on Gewu. Gewu must never depend on a subscriber.

## Engineering Boundaries

- Runtime Core does not handle inbound HTTP or determine caller identity or business permissions.
- Runtime ownership is represented consistently by `subscriber_id + principal_id + principal_type`.
- Runtime Workspace expresses only logical paths, mounts, and capabilities. It does not model business organization hierarchies.
- `gewu-core` contains neither Agent execution rules nor subscriber business logic.

## Working Principles

- Think Before Coding: State assumptions, ambiguities, and tradeoffs before making changes. Ask when requirements or context are uncertain instead of guessing.
- Simplicity First: Use the smallest design and least code that solve the current problem. Do not add abstractions, configuration, or frameworks for hypothetical extensions.
- Surgical Changes: Change only what is required to achieve the goal. Do not opportunistically refactor, reformat, remove comments, or optimize unrelated code.
- Goal-Driven Execution: Translate the task into verifiable outcomes. When a reproduction test is practical, write it before the fix, then run the relevant checks.
- No Legacy Compatibility: Do not preserve legacy logic, data structures, or table designs. Ignore old data unless an explicit migration is requested.

## Development and Verification

- Python 3.12 is the minimum supported version. Use `apply_patch` for manual edits.
- Do not modify generated directories, real configuration, or unrelated lock files.
- Add tests before introducing new behavior, then run:

```bash
uv run black packages tests --check
uv run ruff check packages tests
uv run mypy
uv run pytest
uv lock --check
uv build --all-packages
```
