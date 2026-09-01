# Gewu

**Gewu (格物) is a provider-neutral, embeddable Python runtime for building stateful
server-side AI agents.**

Gewu owns the model and tool execution loop, context construction, durable conversation memory,
context compaction, logical workspaces, skills, scenes, and run coordination. The host application
keeps control of identity, authorization, model selection, data access, and transport.

> **Project status:** Alpha. Gewu is under active development and its public API may change before
> the first stable release.

## Why Gewu

Gewu is designed for teams that already have a backend product and want to add Agent capabilities
without moving business policy into an Agent framework.

- **Host-controlled capabilities:** Every turn receives an already-authorized model, tool set,
  workspace, prompt, and optional Skill or Scene catalog.
- **Durable conversation memory:** Conversations, append-only messages, versioned runtime state,
  runs, suspended Ask flows, and cumulative compactions survive beyond one model call.
- **Managed model context:** Gewu reconstructs provider-neutral model messages, preserves Tool call
  structure, accounts for Tool schemas and images, and protects the model context window.
- **Two-stage compaction:** Low-value historical Tool results can be removed from the model
  projection before older conversation prefixes are replaced by cumulative summaries.
- **Logical Workspace VFS:** Authorized mounts expose logical paths without leaking physical
  storage or business organization structure into the Runtime.
- **Provider-neutral execution:** The core loop depends on model protocols, with optional adapters
  for OpenAI-compatible APIs, Anthropic, and China Unicom Open Service.
- **Production-oriented coordination:** MySQL is the reference fact store. Redis provides
  disposable state caching and distributed run leases.
- **Explicit ownership boundaries:** Runtime data is scoped by
  `subscriber_id + principal_id + principal_type`.

## Core Runtime Flow

```text
Host application
  |-- authenticates the caller
  |-- authorizes the model, tools, workspace, Skills, and Scenes
  `-- creates TurnBindings
              |
              v
        AgentRuntime
              |
              |-- persists the input and Run
              |-- rebuilds context from summary + append-only history
              |-- applies Micro Compact and optional Full Compact
              |-- executes the model/Tool loop
              `-- persists Tool calls, results, output, and runtime state
```

The dependency direction is intentional: subscriber applications may depend on Gewu, while Gewu
must never depend on subscriber identity, permission, organization, or resource-binding rules.

## Context and Memory

### Context management

For each turn, Gewu rebuilds model input from durable Runtime records rather than treating an
in-memory message list as the source of truth. The context pipeline:

- starts from the latest committed compaction boundary;
- loads the remaining message history in bounded pages;
- restores user, assistant, Tool call, and Tool result structure;
- injects the system prompt and cumulative conversation summary;
- avoids duplicating the current input during persistent reconstruction;
- represents historical images as durable attachment references; and
- estimates messages, images, Tool definitions, and Tool arguments before the model call.

### Durable conversation memory

Gewu persists the operational memory required to continue a server-side Agent conversation:

- conversation metadata and an append-only message log;
- versioned conversation state;
- invocation Runs and idempotency records;
- pending Ask state for suspend-and-resume workflows;
- file-read and Skill invocation state; and
- immutable cumulative compaction generations.

MySQL is the reference durable implementation. An in-memory store is included for tests and
embedded development, while Redis can be used as a disposable state cache and distributed Run
lease backend.

In the current release, **memory means durable conversation history and Runtime state**. Gewu does
not yet provide embedding-based semantic memory, vector retrieval, user-profile extraction, or
cross-conversation recall.

### Context compaction

Gewu applies two complementary forms of compaction:

1. **Micro Compact** replaces older successful file, search, and command Tool results with compact
   placeholders in the model projection. The original append-only records remain unchanged.
2. **Full Compact** uses a host-authorized, no-Tool model to replace an older conversation prefix
   with a cumulative summary. Compactions are versioned, retain their sequence boundary and model
   metadata, and are committed atomically with related Runtime state transitions.

The default Full Compact policy triggers at 75% of the model context window, targets 50%, and
treats 90% as the safe hard limit. Full Compact is opt-in: the host must supply both a policy and a
compaction model or model provider.

```python
from gewu_agent_runtime import TurnBindings
from gewu_agent_runtime.compaction import CompactionPolicy

bindings = TurnBindings(
    model=authorized_model,
    workspace=authorized_workspace,
    tool_set=authorized_tool_set,
    compaction_policy=CompactionPolicy(),
    compaction_model=authorized_compaction_model,
)
```

## Built-in Tools

Gewu includes the following optional, business-neutral reference Tools. No Tool is globally
registered or enabled by default; the host selects an authorized `ToolSet` for each turn.

| Category | Tool | Purpose | Required host capability |
| --- | --- | --- | --- |
| Filesystem | `read` | Read a bounded range from a text file | Readable Workspace mount |
| Filesystem | `write` | Create or fully write a text file | Writable Workspace mount |
| Filesystem | `append` | Append text to a file | Writable Workspace mount |
| Filesystem | `edit` | Apply an exact string replacement | Writable Workspace mount |
| Filesystem | `delete` | Delete an exact file or directory path | Writable Workspace mount |
| Filesystem | `list` | List entries under a logical directory | Readable Workspace mount |
| Filesystem | `glob` | Find files by path pattern | Readable Workspace mount |
| Filesystem | `grep` | Search readable text with safe regular expressions | Readable Workspace mount |
| Execution | `bash` | Execute an approved non-interactive local command | Host-provided Bash executor |
| Interaction | `ask_user` | Ask structured questions and optionally suspend the Run | Host callback or Runtime suspension binding |
| Skills | `skill` | Load an authorized Skill into the conversation | Host-provided Skill catalog |

Filesystem Tools operate only through the authorized logical `WorkspaceSession`. `bash` does not
create a shell by itself, and `skill` cannot load entries outside the catalog supplied for the
current turn.

## Extending Gewu with Tools

### Native Python Tools

The current extension mechanism is an in-process Python `Tool`. The `@tool` decorator derives the
model-visible JSON Schema from Python type annotations and Google-style `Args:` documentation.
The resulting Tool is explicitly added to a per-turn `ToolSet` without global registration.

```python
from gewu_agent_runtime.builtins import FILE_TOOLS
from gewu_agent_runtime.tools import ToolResult, ToolSetBuilder, tool


@tool(name="lookup_order", category="orders", allow_parallel=True)
async def lookup_order(order_id: str) -> ToolResult:
    """Look up an order through an application-owned service.

    Args:
        order_id: Stable order identifier.
    """

    # `order_service` is an application-owned, already-authorized dependency.
    order = await order_service.get_order(order_id)
    return ToolResult(output={"order": order})


authorized_tool_set = (
    ToolSetBuilder(name="support-agent", version="v1")
    .extend(FILE_TOOLS)
    .add(lookup_order)
    .build()
)
```

The host remains responsible for authenticating the caller, authorizing the operation, binding
credentials, and deciding which Tools are visible in a given turn.

### Model Context Protocol

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) client support is **not implemented
in the current release**. Gewu does not yet connect to MCP servers, discover remote Tools, or route
Tool calls over stdio or Streamable HTTP.

Until an MCP client adapter is available, external services must be exposed through native Python
Tools and injected into `TurnBindings.tool_set`. MCP support is a natural future adapter boundary:
remote Tool definitions can be converted into Gewu `Tool` values and then pass through the same
per-turn authorization and execution pipeline.

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
  |-- ConversationContextBuilder   persistent context reconstruction
  |-- CompactionService            Micro and Full Compact
  |-- AgentEngine                  provider-neutral model and Tool loop
  |-- RuntimeStore                 conversations, messages, Runs, state, compactions
  |-- RunLease                     cross-process Run coordination
  `-- StateCache                   disposable conversation-state cache
```

## Packages

| Project | Python package | Responsibility |
| --- | --- | --- |
| [`packages/agent-runtime`](packages/agent-runtime) | `gewu-agent-runtime` | Agent execution, context, memory, compaction, Tools, Workspaces, Skills, Scenes, persistence contracts, and adapters |
| [`packages/gewu-core`](packages/gewu-core) | `gewu-core` | Configuration, logging, database, Redis, HTTP, concurrency, and shared infrastructure |

## Getting Started

Gewu requires Python 3.12 or newer and uses a `uv` workspace.

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
model, Workspace, Tool, MySQL, and Redis bindings. See the
[`gewu-agent-runtime` package guide](packages/agent-runtime/README.md) for the ownership and
integration contracts.

## Runtime Boundary

Gewu intentionally does not:

- handle inbound HTTP or choose a transport protocol;
- authenticate callers or derive business permissions;
- select tenant resources from untrusted request fields;
- infer organization hierarchy from Workspace paths; or
- treat prompts or Tool descriptions as an authorization boundary.

The host must resolve those policies before constructing `PreparedAgentTurn` or `TurnBindings`.

## Development

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
