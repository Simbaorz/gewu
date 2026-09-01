# Gewu

Gewu is a business-neutral Python Agent Runtime monorepo. It provides in-process Agent execution,
a model and tool loop, conversation state, context compaction, Workspace VFS, skills, scenes, and
execution coordination without embedding subscriber identity, authorization, organization, or
resource-binding rules.

## Projects

| Project | Python package | Responsibility |
| --- | --- | --- |
| `packages/gewu-core` | `gewu-core` | Configuration, logging, database, Redis, HTTP, and shared infrastructure |
| `packages/agent-runtime` | `gewu-agent-runtime` | Agent execution, tools, workspaces, skills, scenes, and persistence protocols |

Subscribers integrate through `SubscriberRuntimeProvider -> PreparedAgentTurn -> AgentRuntime`.
Business-specific implementations live in separate subscriber repositories.

## Development

```bash
uv sync --all-packages --all-extras
uv run pytest
uv run mypy
uv build --all-packages
```
