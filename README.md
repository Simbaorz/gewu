# Gewu

**Gewu is a provider-neutral, embeddable Python runtime for building stateful server-side AI
agents.**

It owns the model and tool execution loop, durable conversation state, context compaction,
logical workspaces, skills, scenes, and run coordination. The host application keeps control of
identity, authorization, model selection, data access, and transport.

> **Project status:** Alpha. Gewu is under active development and its public API may change before
> the first stable release.

## Why Gewu

Gewu is designed for teams that already have a backend product and want to add Agent capabilities
without moving business policy into an Agent framework.

- **Host-controlled capabilities:** Every turn receives an already-authorized model, tool set,
  workspace, prompt, and optional skill or scene catalog.
- **Durable conversations:** Append-only messages, versioned state, idempotent runs, suspended Ask
  flows, and cumulative context compactions survive beyond one model call.
- **Logical Workspace VFS:** Authorized mounts expose logical paths without leaking physical
  storage or business organization structure into the Runtime.
- **Provider-neutral execution:** The core loop depends on model protocols, with optional adapters
  for OpenAI-compatible APIs, Anthropic, and China Unicom Open Service.
- **Production-oriented coordination:** MySQL is the reference fact store. Redis provides
  disposable state caching and distributed run leases.
- **Explicit ownership boundaries:** Runtime data is scoped by
  `subscriber_id + principal_id + principal_type`.

## Architecture

```text
HTTP / RPC / Worker
        |
        v
SubscriberRuntimeProvider
        |
        v
PreparedAgentTurn
        |
        v
AgentRuntime
  |-- AgentEngine           model and tool loop
  |-- RuntimeStore          conversations, messages, runs, and compactions
  |-- RunLease              cross-process run coordination
  `-- StateCache            disposable conversation-state cache
```

The dependency direction is intentional: subscriber applications may depend on Gewu, while Gewu
must never depend on subscriber identity, permission, organization, or resource-binding rules.

## Packages

| Project | Python package | Responsibility |
| --- | --- | --- |
| [`packages/agent-runtime`](packages/agent-runtime) | `gewu-agent-runtime` | Agent execution, tools, workspaces, skills, scenes, persistence contracts, and adapters |
| [`packages/gewu-core`](packages/gewu-core) | `gewu-core` | Configuration, logging, database, Redis, HTTP, concurrency, and shared infrastructure |

## Getting Started

Clone the repository and synchronize the workspace:

```bash
git clone https://github.com/Simbaorz/gewu.git
cd gewu
uv sync --all-packages --all-extras
```

## Minimal In-Process Example

The scripted model makes this example deterministic and requires no external API key:

```python
import asyncio

from gewu_agent_runtime import (
    AgentRuntime,
    PrincipalRef,
    PrincipalType,
    TurnBindings,
    TurnRequest,
)
from gewu_agent_runtime.engine import AssistantFinal
from gewu_agent_runtime.llm import ModelStreamChunk, ScriptedChatModel
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)


async def main() -> None:
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=InMemoryWorkspaceBackend(),
            )
        ]
    )
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="Hello from Gewu.")]])
    runtime = AgentRuntime(store=InMemoryRuntimeStore())

    session = await runtime.start_turn(
        TurnRequest(
            invoker=PrincipalRef(
                subscriber_id="example",
                principal_id="user-1",
                principal_type=PrincipalType.USER,
            ),
            content="Say hello.",
        ),
        TurnBindings(model=model, workspace=workspace),
    )

    async for event in session.stream():
        if isinstance(event, AssistantFinal):
            print(event.content)


asyncio.run(main())
```

For a production integration, the host replaces the in-memory implementations with authorized
model, workspace, tool, MySQL, and Redis bindings. See the
[`gewu-agent-runtime` package guide](packages/agent-runtime/README.md) for the ownership and
integration contracts.

## Runtime Boundary

Gewu intentionally does not:

- handle inbound HTTP or choose a transport protocol;
- authenticate callers or derive business permissions;
- select tenant resources from untrusted request fields;
- infer organization hierarchy from workspace paths; or
- treat prompts as an authorization boundary.

The host must resolve those policies before constructing `PreparedAgentTurn` or `TurnBindings`.

## Development

Gewu requires Python 3.12 or newer and uses a `uv` workspace.

```bash
uv sync --all-packages --all-extras
uv run black packages tests --check
uv run ruff check packages tests
uv run mypy
uv run pytest
uv lock --check
uv build --all-packages
```

## License

Gewu is released under the [MIT License](LICENSE).
