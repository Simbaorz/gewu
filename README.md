# Gewu（格物）

Gewu 是业务中立的 Python Agent Runtime Monorepo。它提供进程内 Agent 执行、模型与工具循环、
会话状态、上下文压缩、Workspace VFS、Skill、Scene 和执行协调，不包含任何订阅方的身份、
权限、组织和资源绑定规则。

## Projects

| 工程 | Python 包 | 职责 |
| --- | --- | --- |
| `packages/gewu-core` | `gewu-core` | 配置、日志、数据库、Redis、HTTP 和通用基础设施 |
| `packages/agent-runtime` | `gewu-agent-runtime` | Agent 执行、Tool、Workspace、Skill、Scene 和持久化协议 |

订阅方通过 `SubscriberRuntimeProvider -> PreparedAgentTurn -> AgentRuntime` 接入，具体业务实现
位于各自独立的订阅方仓库中。

## Development

```bash
uv sync --all-packages --all-extras
uv run pytest
uv run mypy
uv build --all-packages
```
