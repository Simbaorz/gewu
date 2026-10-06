# Gewu

[English](README.md) | [简体中文](README.zh-CN.md)

**Gewu (格物) is a Python runtime for running AI agents inside enterprise backend
applications.**

It connects models, Tools, knowledge Workspaces, and conversation storage, and manages the full
lifecycle of an Agent request—from start and Tool use to completion, suspension, resumption, and
persistence. Your application continues to own user identity, business authorization, knowledge
management, and data APIs. Gewu runs only the capabilities that the application has authorized.

If you are building enterprise knowledge Q&A, customer support, or a business assistant, Gewu can
serve as the Agent foundation underneath the product.

> **Project status:** Alpha. Gewu is under active development and its public API may change before
> the first stable release.

## What Problem Does It Solve?

Calling a model once is easy. Running an Agent inside a real enterprise system requires much more
than a model API provides:

- Which model, Tools, and knowledge may this request use?
- How should Tool Calls, Tool Results, and images be saved and restored on the next turn?
- How can a long conversation stay within the model context window without simply discarding its
  history?
- How should a suspended Run resume after the user supplies missing information?
- How can concurrent requests avoid modifying the same conversation at the same time?
- How can a conversation continue after a service restart or on another worker?

Gewu collects these shared concerns in an embeddable Runtime so that the host application can stay
focused on its own business and product.

## A Concrete Knowledge Q&A Request

Suppose a user asks:

> Does this order meet our refund policy?

The host application first makes the business decisions: identify the user, select an authorized
“After-sales Policy” Scene, mount the relevant Wiki, and provide an order lookup Tool that already
enforces data permissions. It then passes those capabilities to Gewu.

```text
User question
   |
   v
Host application
  |-- authenticates the user and resolves permissions
  |-- selects the “After-sales Policy” Scene
  |-- mounts a readable policy Wiki
  |-- injects an authorized order lookup Tool
  `-- selects the model for this turn
   |
   v
Gewu
  |-- persists the input and creates a Run
  |-- reconstructs context from durable history
  |-- tells the model which Scene and Skill apply
  |-- lets the Agent inspect refund rules with grep/read
  |-- lets the Agent obtain current order facts through the Tool
  |-- asks the user for missing information through ask_user when needed
  `-- persists Tool Calls, Tool Results, the answer, and conversation state
```

In this flow:

- the Wiki contains rules, explanations, and data definitions;
- Tools provide current facts such as orders, inventory, customers, metrics, or tickets; and
- Gewu lets the model use both within a controlled boundary and keeps the conversation running
  reliably.

Gewu does not understand what an “order” or “refund” means. The same Runtime can support HR
policies, product manuals, equipment maintenance, internal support, and other enterprise domains.

## What Gewu Provides

### 1. A complete Agent execution loop

Gewu provides a streaming model and Tool loop that:

- sends authorized Tool schemas to the model;
- receives streaming text and Tool Calls;
- executes Tools and returns their results to the model for continued reasoning;
- preserves Assistant, Tool Call, and Tool Result structure;
- enforces maximum iterations and safe context usage; and
- emits structured events for text deltas, Tool use, final answers, suspension, and errors.

The core execution loop depends only on model protocols. Gewu currently includes adapters for
OpenAI-compatible APIs, Anthropic, and China Unicom Open Service, and a host can implement its own
model binding.

### 2. Server-side conversations and Run state

Gewu does not treat an in-process message list as the source of truth. It can persist:

- conversations and append-only messages;
- every invocation Run and its lifecycle;
- Tool Calls, Tool Results, and attachments;
- versioned conversation state;
- idempotency records;
- pending Ask state while waiting for a user; and
- context compaction generations.

MySQL is the reference fact store. Redis provides a disposable state cache and distributed Run
leases. Losing Redis cannot delete completed conversations or Runs. An in-memory Store is included
for tests and embedded development.

### 3. Context management and two-stage compaction

At the start of every turn, Gewu reconstructs model context from durable records and restores
messages, images, Tool Calls, and Tool Results in their correct structure. Before calling the
model, it accounts for messages, images, Tool schemas, and Tool arguments.

When a conversation grows, Gewu offers two stages of compaction:

1. **Micro Compact** shortens older file, search, and command results in the model projection
   without changing the original message records.
2. **Full Compact** uses a separate, Tool-free model to turn an older conversation prefix into a
   cumulative summary and commits it atomically as a versioned generation.

The database therefore retains the complete history while the model sees a context that remains
within usable limits.
Full Compact is opt-in: the host must explicitly provide a compaction policy and compaction model
or model provider.

In the current release, Gewu uses “memory” to mean durable conversation history and Runtime state.
It does not yet include user-profile extraction, cross-conversation semantic recall, or
embedding-based long-term memory.

### 4. A logical Workspace built for Agents

`WorkspaceSession` is a capability-scoped logical filesystem. The model sees logical paths and
does not need to know whether knowledge is physically stored in a local directory, object storage,
a database, or another service.

The host can combine multiple mounts in one turn and assign read-only or read-write access to each
one. For example, an application can assemble company-wide knowledge, team knowledge, and a user
workspace into one authorized view before handing it to Gewu.

Gewu does not interpret an enterprise organization hierarchy or decide how knowledge is inherited
or shared downward. The host resolves those policies first; Gewu enforces only the resulting
Workspace capabilities.

### 5. Scenes and Skills

These objects answer different questions:

| Object | Purpose | Example |
| --- | --- | --- |
| Scene | Where should the Agent work? | After-sales policy, finance policy, equipment manual |
| Skill | How should the Agent perform this kind of work? | Check policy first, obtain current facts, then provide a conclusion with evidence |

A Scene has a name, description, and Workspace root path, and may require a Skill. When a user
selects a Scene, Gewu adds its name, entry path, and visible bound Skill name to the current
context as a Meta Message. Scene descriptions and workflow instructions are not included in this
reminder. The host's system prompt defines how to use Scenes and their bound Skills. Gewu does
not put the entire knowledge directory into the prompt; the Agent discovers and reads evidence
on demand through file Tools.

Skills come from a catalog that the host has already filtered. Gewu lists only authorized Skills
and loads full instructions only when the model invokes the `skill` Tool.

### 6. Ownership boundaries for multiple subscribers and users

Runtime ownership is represented consistently as:

```text
subscriber_id + principal_id + principal_type
```

This gives multi-tenant, multi-user services a stable persistence scope and prevents one operation
from crossing `subscriber_id`. The host application still defines the actual tenant, department,
team, role, and data-permission model.

### 7. Suspend-and-resume user clarification

An Agent can ask structured questions through `ask_user`. If the current request cannot receive an
immediate answer, Gewu saves the pending Ask state and suspends the Run. A matching user answer can
later resume that exact Run instead of restarting the whole turn.

## How Knowledge Enters Gewu

Curated Wiki or Markdown knowledge can be mounted into the Workspace so that the Agent can navigate
it with `list`, `glob`, `grep`, and `read`. This preserves directory, document, and section structure,
and changes do not require a mandatory chunking and embedding pipeline.

Gewu is not a built-in RAG engine, and it does not require all knowledge to use file navigation. If
your corpus needs full-text search, vector retrieval, reranking, or another retrieval method, expose
that capability as a Tool and authorize it per turn like any other business Tool.

## Built-in Tools

Gewu currently includes eleven optional, business-neutral Tools. They are not globally registered
or enabled by default. The host explicitly builds a `ToolSet` for every turn.

| Category | Tool | Purpose |
| --- | --- | --- |
| Files | `read` | Read a bounded range from a text file |
| Files | `write` | Create or fully write a text file |
| Files | `append` | Append text |
| Files | `edit` | Apply an exact text replacement |
| Files | `delete` | Delete an exact file or directory |
| Discovery | `list` | List a logical directory |
| Discovery | `glob` | Find files by path pattern |
| Discovery | `grep` | Search text with safe regular expressions |
| Execution | `bash` | Call a host-provided non-interactive command executor |
| Interaction | `ask_user` | Ask structured questions and optionally suspend the Run |
| Skills | `skill` | Load an authorized Skill |

File Tools can operate only through the current `WorkspaceSession`. `bash` is a contract, not a
shell created by Gewu; it works only when the host supplies a Bash executor. An enterprise
knowledge application can omit it entirely.

## Adding a Business Tool

The current extension mechanism uses `@tool` to wrap an application service as a Python Tool and
add it to the ToolSet for the current turn:

```python
from gewu_agent_runtime.builtins import FILE_TOOLS
from gewu_agent_runtime.tools import ToolResult, ToolSetBuilder, tool


@tool(name="lookup_order", category="orders")
async def lookup_order(order_id: str) -> ToolResult:
    """Return an order that the current user is allowed to view.

    Args:
        order_id: Order identifier.
    """

    order = await authorized_order_service.get(order_id)
    return ToolResult(output={"order": order})


tool_set = (
    ToolSetBuilder(name="after-sales", version="v1")
    .extend(FILE_TOOLS)
    .add(lookup_order)
    .build()
)
```

`authorized_order_service`, database credentials, and data permissions belong to the host. Gewu
only calls the injected Tool and never bypasses it to access the business database directly.

### MCP status

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) client support is **not implemented
in the current release**. Gewu cannot yet connect directly to MCP servers, discover remote Tools,
or execute Tool Calls over stdio or Streamable HTTP.

External capabilities currently need a native Python Tool wrapper. A future MCP adapter will enter
the same per-turn authorization and Tool execution path instead of bypassing host permissions.

## Gewu and Host Responsibilities

| Gewu owns | The host application owns |
| --- | --- |
| Model and Tool execution loop | HTTP, WebSocket, messaging channels, and UI |
| Conversation, message, Run, and state persistence | Login, users, organizations, and roles |
| Context reconstruction, accounting, and compaction | Business authorization for models and Tools |
| Logical Workspace capability execution | Wiki editing, ingestion, versioning, and publishing |
| Runtime projection of Scenes and Skills | Scene and Skill management and authorization |
| Ask suspension and resumption | Business databases and external service integration |
| Run leases and concurrency protection | Audit, operations, and product policy |

Gewu never treats a prompt, Meta Message, or Tool description as a security boundary. Every trusted
permission must be resolved before the host creates `PreparedAgentTurn` or `TurnBindings`.

## Getting Started

Gewu requires Python 3.12 or newer and uses a `uv` workspace.

```bash
git clone https://github.com/Simbaorz/gewu.git
cd gewu
uv sync --all-packages --all-extras
```

This minimal example uses an in-memory Store and a scripted model, so it requires no API key:

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

In a production integration, the host supplies authorized model, Workspace, and ToolSet bindings,
plus the required MySQL and Redis adapters. See the
[`gewu-agent-runtime` package guide](packages/agent-runtime/README.md) for detailed contracts.

## Packages

| Project | Python package | Responsibility |
| --- | --- | --- |
| [`packages/agent-runtime`](packages/agent-runtime) | `gewu-agent-runtime` | Agent execution, context, memory, compaction, Tools, Workspaces, Skills, Scenes, persistence contracts, and adapters |
| [`packages/gewu-core`](packages/gewu-core) | `gewu-core` | Configuration, logging, database, Redis, HTTP, concurrency, and shared infrastructure |

## Not Included Today

- A ready-to-use Web chat or administration product;
- user, organization, and business authorization systems;
- Wiki editing and general-purpose document ingestion;
- a built-in vector knowledge base or RAG pipeline;
- semantic long-term memory and cross-conversation user profiles;
- an MCP client; or
- Subagent orchestration.

These capabilities can be implemented by a product above Gewu or connected through Tools, but they
should not be understood as features already included in the current release.

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
