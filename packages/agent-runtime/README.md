# Gewu Agent Runtime

`gewu-agent-runtime` is a reusable in-process Python Agent Runtime. The host prepares one turn by
injecting an authorized model, tool set, workspace session, resource catalogs, and persistence
implementations. The runtime does not resolve business identity or permissions.

## Ownership Boundary

The package owns the model/tool loop and these durable records:

- `agent_conversation` and its append-only message log
- versioned conversation state and immutable cumulative compactions
- invocation runs, including invoker identity, idempotency, lifecycle, and model snapshot

The host owns authentication, authorization, model selection, data-source access, wiki/skill
catalog filtering, and construction of workspace mounts. A host may use its own users or service
principals; the Runtime only enforces that one operation cannot cross `subscriber_id`.

Authorized Skill catalogs must also resolve subscriber-specific same-name precedence before they
cross the Runtime boundary. The Runtime accepts only unique model-visible names and stable asset
keys and rejects duplicates as a host contract error; it never derives precedence from catalog
order or organization metadata.

MySQL is the fact source. Redis adapters are disposable optimizations for state caching and
distributed run leases. Losing Redis cannot erase conversations or completed runs.

## Package Layers

```text
domain / llm / tools / workspace       neutral values and protocols
engine                                 persistence-free model/tool loop
runtime / context / compaction         persistent turn orchestration
builtins                               optional Ask, file, search, and Skill tools
adapters/mysql                         SQLAlchemy fact store and agent_* tables
adapters/redis                         TTL run lease and state cache
```

Install Core only:

```bash
uv add gewu-agent-runtime
```

Install only the adapters required by the host process:

```bash
uv add "gewu-agent-runtime[mysql,redis]"
```

The available extras are `mysql`, `redis`, `providers`, and `host`; `all` remains a convenience for
deployments that intentionally use every adapter.

## Minimal In-Process Use

```python
from gewu_agent_runtime import AgentRuntime, PrincipalRef, PrincipalType, TurnBindings, TurnRequest
from gewu_agent_runtime.persistence import InMemoryRuntimeStore

runtime = AgentRuntime(store=InMemoryRuntimeStore())
session = await runtime.start_turn(
    TurnRequest(
        invoker=PrincipalRef(
            subscriber_id="subscriber-a",
            principal_id="user-42",
            principal_type=PrincipalType.USER,
        ),
        content="Analyze the workspace",
    ),
    TurnBindings(
        model=authorized_model,
        workspace=authorized_workspace,
        tool_set=authorized_tool_set,
        prompt=assembled_prompt,
    ),
)

async for event in session.stream():
    await transport.send(event)
```

For schema-managed deployments, import `AgentRuntimeBase.metadata` into migrations. The
`create_schema(engine)` helper is intended for tests and simple deployments.

## Subscriber Prompts and Scene Context

`build_system_prompt(static_sections=...)` replaces the Runtime's default static sections with
subscriber-provided rules. Omitting the argument preserves the default prompt; an empty sequence
removes its static sections. After the static/dynamic boundary, actual Workspace context comes
first, followed by extra dynamic sections and then supplied memory. Empty sections are omitted,
including an empty Workspace context. Current timestamps are not automatically injected.

A selected Scene reminder contains its name, logical entry path, and visible bound Skill name
when available. It does not include the Scene description or workflow instructions. The default
system prompt defines how to apply the selected Scene, load its bound Skill, and inspect its
files. Hosts replacing the static prompt own these rules. Scenes are selected explicitly by
the user; the Runtime does not list all Scenes for autonomous model selection. A default Scene
is used only when supplied by the host, without overriding the user's explicit selection.

Model invocation failures emit `MODEL_CALL_FAILED` diagnostics only at DEBUG level. They include
run, conversation, and request identifiers, the error code, underlying HTTP status when available,
and exception types and code locations, without request or response contents or exception messages.
