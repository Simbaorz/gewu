# Gewu

[English](README.md) | [简体中文](README.zh-CN.md)

**Gewu (格物) is a provider-neutral, embeddable Python runtime for building governed,
stateful, server-side AI agents.**

Gewu turns an authorized model, ToolSet, logical Workspace, Skill and Scene catalog, and
persistence backend into a durable Agent conversation. It owns the model and Tool loop, context
construction, conversation memory, context compaction, suspend-and-resume interaction, and Run
coordination. The host application keeps control of identity, authorization, organization policy,
model access, business data, and transport.

Gewu is especially suited to self-hosted organizational knowledge Agents. A host can expose a
curated Wiki as a logical Workspace, use Scenes to define knowledge scopes, use Skills to supply
repeatable working methods, and inject authorized Tools for live facts. The Agent can then inspect
the relevant knowledge and obtain current data without moving business rules or credentials into
the Runtime.

The broader project goal is to make capable, controllable knowledge Agents practical for
enterprises without requiring a general-purpose desktop Agent or a vector database by default.

> **Project status:** Alpha. Gewu is under active development and its public API may change before
> the first stable release.

## Where Gewu Fits

Gewu is the Runtime layer, not a complete knowledge-base product or SaaS application.

```text
Web / Admin / API / Messaging channels
                  |
                  v
Host application
  |-- authenticates users and resolves organization policy
  |-- authorizes models, Scenes, Skills, Tools, and Workspace mounts
  |-- manages Wiki content and live business-data services
  `-- prepares one Agent turn
                  |
                  v
Gewu Agent Runtime
  |-- executes the model and Tool loop
  |-- constructs and compacts model context
  |-- persists conversations, messages, Runs, and Runtime state
  |-- projects Scenes and Skills into the conversation
  `-- coordinates concurrent and suspended execution
                  |
       +----------+-----------+
       |          |           |
       v          v           v
     Models   Workspace    Authorized Tools
                            (live facts/actions)
```

This boundary is intentional. Subscriber applications may depend on Gewu, while Gewu must never
depend on subscriber identity systems, business permissions, organization hierarchies, Wiki
schemas, or data APIs.

## Core Capabilities

| Capability | What Gewu provides |
| --- | --- |
| Agent execution | A streaming, provider-neutral model and Tool loop with structured lifecycle events, bounded iterations, and explicit completion, suspension, and failure states |
| Durable conversations | Append-only messages, versioned conversation state, invocation Runs, idempotency records, attachments, pending Ask state, and immutable compaction generations |
| Context management | Persistent context reconstruction, Tool-call preservation, multimodal accounting, token estimation, and safe context-window limits |
| Context compaction | Micro Compact for low-value historical Tool results and model-generated Full Compact for cumulative conversation summaries |
| Logical Workspace VFS | Capability-scoped logical mounts that hide physical storage and can combine multiple host-authorized knowledge sources in one Agent view |
| Scenes | An authorized knowledge scope with a logical root, description, and an optional required Skill workflow |
| Skills | Reusable instructions loaded into the conversation only when authorized and needed |
| Tools | Eleven optional business-neutral Tools for files, search, controlled execution, user clarification, and Skill loading, plus native Python Tool extension |
| Human interaction | Structured clarification through `ask_user`, including persistent suspension and exact Run resumption |
| Ownership and coordination | Consistent `subscriber_id + principal_id + principal_type` ownership, MySQL persistence, Redis state caching, and distributed Run leases |
| Model integration | Protocol-based execution with optional adapters for OpenAI-compatible APIs, Anthropic, and China Unicom Open Service |

## Knowledge, Workflows, and Live Facts

Gewu deliberately separates three different kinds of context:

```text
Scene       where the Agent should work
Skill       how the Agent should approach the work
Workspace   the knowledge and evidence the Agent may inspect
Tool        the live facts or actions the host permits
```

### Scenes as knowledge scopes

A Scene points to an authorized logical path in the current Workspace. When the host selects a
Scene, Gewu adds a Meta Message describing the Scene name, root path, description, and bound
workflow. Gewu does not copy the entire Scene into the model context. The Agent is instructed to
discover the relevant entry points and read only the evidence needed for the current question.

If a Scene requires a Skill, the Runtime tells the Agent to load it before handling the Scene.
This allows an application to pair a body of knowledge with a repeatable method without baking
that method into a global system prompt.

### Skills as reusable working methods

Skills contain task-specific instructions and are resolved from a catalog already filtered by
the host. Gewu lists only authorized Skills, loads full Skill content on demand through the
`skill` Tool, records invocation state, and keeps Skill content aligned with the active model
context.

### Wiki knowledge without a mandatory vector index

For curated Wiki or Markdown knowledge, the host can mount the content into the Workspace and let
the Agent navigate it with `list`, `glob`, `grep`, and `read`. This preserves directory, document,
and section structure and makes edits immediately visible without a mandatory chunking,
embedding, or re-indexing pipeline.

Gewu is not a built-in RAG engine and does not claim that file navigation replaces semantic
retrieval for every corpus. A host may inject full-text search, vector retrieval, reranking, or
any hybrid retrieval strategy as an authorized Tool when the knowledge scale or format requires
it.

### Live business facts stay outside the Runtime

Operational facts such as orders, inventory, customers, metrics, or tickets belong to the host
application. The host exposes them through narrow, authorized Tools rather than teaching Gewu a
business schema or giving the model unrestricted database access. This keeps business data
access, credentials, row-level policy, and audit rules in the system that owns them.

## Durable Context and Memory

### Persistent context reconstruction

For every turn, Gewu rebuilds model input from durable Runtime records rather than treating an
in-memory message list as the source of truth. The context pipeline:

- starts from the latest committed compaction boundary;
- loads the remaining history in bounded pages;
- restores user, assistant, Tool call, and Tool result structure;
- injects the system prompt, cumulative summary, and authorized Meta Messages;
- avoids duplicating the current input during reconstruction;
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

MySQL is the reference durable implementation. An in-memory Store is included for tests and
embedded development. Redis is an optional, disposable state cache and distributed Run lease
backend; it is not the conversation fact store.

In the current release, **memory means durable conversation history and Runtime state**. Gewu does
not yet provide embedding-based semantic memory, user-profile extraction, or cross-conversation
recall.

### Two-stage context compaction

1. **Micro Compact** replaces older successful file, search, and command Tool results with compact
   placeholders in the model projection. Original append-only records remain unchanged.
2. **Full Compact** uses a host-authorized, no-Tool model to replace an older conversation prefix
   with a cumulative summary. Compactions are versioned, retain their sequence boundary and model
   metadata, and are committed atomically with related Runtime state transitions.

The default Full Compact policy triggers at 75% of the model context window, targets 50%, and
treats 90% as the safe hard limit. Full Compact is opt-in: the host must provide both a policy and
a compaction model or model provider.

## Authorized Workspaces and Ownership

`WorkspaceSession` presents one already-authorized logical filesystem view to the Agent. A host
can combine tenant, team, shared, user, or application-owned content as separate mounts and assign
read-only or read-write capability to each mount. Longest-prefix routing and logical paths keep
backend storage details out of model-visible Tool calls.

Gewu does not interpret a tenant, province, city, team, or user hierarchy. The host resolves any
inheritance or downward-sharing policy first, then constructs the Workspace and catalogs visible
for the current turn. Runtime records are scoped consistently by
`subscriber_id + principal_id + principal_type`, and persistence operations reject cross-subscriber
access.

## Runtime Flow

```text
Host application
  |-- authenticates the caller
  |-- authorizes the model, Tools, Workspace, Skills, and Scene
  `-- creates TurnBindings
              |
              v
        AgentRuntime
              |
              |-- persists the input and Run
              |-- rebuilds context from summary + append-only history
              |-- prepares Scene, Skill, and Tool context
              |-- applies Micro Compact and optional Full Compact
              |-- executes the streaming model and Tool loop
              `-- persists Tool calls, results, output, and Runtime state
```

## Built-in Tools

No Tool is globally registered or enabled by default. The host explicitly selects an authorized
`ToolSet` for each turn.

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

Filesystem Tools operate only through the authorized `WorkspaceSession`. `bash` does not create a
shell by itself: it is only a Tool contract and requires a host-provided executor. A knowledge
application can omit it entirely. The `skill` Tool cannot load entries outside the catalog
supplied for the current turn.

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
Tools and injected into `TurnBindings.tool_set`. MCP is the intended standard adapter boundary for
future third-party Tool integration; it will still pass through the same host authorization and
per-turn ToolSet model.

## Architecture and Packages

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

### Minimal in-process example

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

## Intentional Boundaries

Gewu intentionally does not:

- provide an end-user Web or administration product;
- handle inbound HTTP or choose a transport protocol;
- authenticate callers or derive organization and business permissions;
- manage Wiki authoring, document ingestion, or business schemas;
- select tenant resources from untrusted request fields;
- infer organization hierarchy from Workspace paths;
- provide built-in vector retrieval or semantic long-term memory;
- connect to MCP servers in the current release; or
- treat prompts, Meta Messages, or Tool descriptions as authorization boundaries.

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
