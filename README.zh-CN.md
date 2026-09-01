# Gewu

[English](README.md) | [简体中文](README.zh-CN.md)

**Gewu（格物）是一个模型提供商无关、可嵌入的 Python 运行时，用于构建有状态的服务端 AI Agent。**

Gewu 负责模型与 Tool 执行循环、上下文构建、持久化会话记忆、上下文压缩、逻辑
Workspace、Skill、Scene 与 Run 协调。宿主应用继续掌控身份、授权、模型选择、数据访问与
传输协议。

> **项目状态：** Alpha。Gewu 正在积极开发中，在首个稳定版本发布前，公开 API 可能发生
> 变化。

## 为什么选择 Gewu

Gewu 面向已经拥有后端产品、希望加入 Agent 能力，但不希望将业务策略迁入 Agent 框架的
团队。

- **由宿主控制能力：** 每个 Turn 都接收已经完成授权的模型、ToolSet、Workspace、Prompt，
  以及可选的 Skill 或 Scene Catalog。
- **持久化会话记忆：** Conversation、仅追加消息、带版本的 Runtime State、Run、可暂停的
  Ask 流程和累计压缩均可跨越单次模型调用持续存在。
- **受管理的模型上下文：** Gewu 重建与模型提供商无关的消息，保留 Tool Call 结构，在计算
  上下文时包含 Tool Schema 与图片，并保护模型上下文窗口。
- **两级压缩：** 先从模型投影中清除低价值的历史 Tool Result，再用累计摘要替换更早的会话
  前缀。
- **逻辑 Workspace VFS：** 通过已授权挂载提供逻辑路径，不向 Runtime 泄露物理存储或业务
  组织结构。
- **模型提供商无关的执行：** 核心循环只依赖模型协议，并提供 OpenAI 兼容 API、Anthropic
  和中国联通开放服务的可选适配器。
- **面向生产的协调机制：** MySQL 是参考事实存储，Redis 提供可丢弃的 State Cache 与分布式
  Run Lease。
- **明确的所有权边界：** Runtime 数据使用
  `subscriber_id + principal_id + principal_type` 确定作用域。

## Runtime 核心流程

```text
宿主应用
  |-- 认证调用者
  |-- 授权模型、Tool、Workspace、Skill 与 Scene
  `-- 创建 TurnBindings
              |
              v
        AgentRuntime
              |
              |-- 持久化输入与 Run
              |-- 根据摘要和仅追加历史重建上下文
              |-- 执行 Micro Compact 与可选的 Full Compact
              |-- 执行模型与 Tool 循环
              `-- 持久化 Tool Call、结果、输出与 Runtime State
```

依赖方向是刻意设计的：订阅方应用可以依赖 Gewu，但 Gewu 不得依赖订阅方的身份、权限、
组织结构或资源绑定规则。

## 上下文与记忆

### 上下文管理

每个 Turn 中，Gewu 都会根据持久化 Runtime 记录重建模型输入，而不是将内存中的消息列表视为
事实来源。上下文流水线会：

- 从最新一次已提交的压缩边界开始；
- 通过有界分页加载剩余消息历史；
- 恢复 User、Assistant、Tool Call 与 Tool Result 结构；
- 注入 System Prompt 与累计会话摘要；
- 在持久化重建期间避免重复加入当前输入；
- 将历史图片表示为持久化附件引用；
- 在调用模型前估算消息、图片、Tool 定义与 Tool 参数所占的 Token。

### 持久化会话记忆

Gewu 持久化继续运行服务端 Agent 会话所需的操作记忆：

- Conversation 元数据与仅追加消息日志；
- 带版本的 Conversation State；
- Invocation Run 与幂等记录；
- 用于暂停和恢复流程的 Pending Ask State；
- 文件读取状态与 Skill 调用状态；
- 不可变的累计压缩版本。

MySQL 是参考持久化实现。项目提供用于测试和嵌入式开发的内存 Store；Redis 可作为可丢弃的
State Cache 与分布式 Run Lease 后端。

在当前版本中，**记忆指持久化会话历史与 Runtime State**。Gewu 尚未提供基于 Embedding 的
语义记忆、向量检索、用户画像提取或跨会话召回。

### 上下文压缩

Gewu 提供两种互补的压缩方式：

1. **Micro Compact** 将较早且执行成功的文件、搜索和命令类 Tool Result 替换为模型投影中的
   简短占位符，原始的仅追加记录不会被修改。
2. **Full Compact** 使用宿主授权且不携带 Tool 的模型，将较早的会话前缀替换为累计摘要。
   压缩记录带版本，保留序列边界与模型元数据，并与相关 Runtime State 变更原子提交。

默认 Full Compact 策略在模型上下文窗口使用率达到 75% 时触发，以 50% 为压缩目标，并将
90% 视为安全硬限制。Full Compact 需要显式启用：宿主必须同时提供压缩策略和压缩模型或模型
Provider。

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

## 内置 Tool

Gewu 提供以下可选、业务中立的参考 Tool。任何 Tool 都不会被全局注册或默认启用；宿主为每个
Turn 选择已经授权的 `ToolSet`。

| 分类 | Tool | 用途 | 所需宿主能力 |
| --- | --- | --- | --- |
| 文件系统 | `read` | 读取文本文件的有界范围 | 可读 Workspace Mount |
| 文件系统 | `write` | 创建或完整写入文本文件 | 可写 Workspace Mount |
| 文件系统 | `append` | 向文件追加文本 | 可写 Workspace Mount |
| 文件系统 | `edit` | 执行精确字符串替换 | 可写 Workspace Mount |
| 文件系统 | `delete` | 删除精确的文件或目录路径 | 可写 Workspace Mount |
| 文件系统 | `list` | 列出逻辑目录中的条目 | 可读 Workspace Mount |
| 文件系统 | `glob` | 按路径模式查找文件 | 可读 Workspace Mount |
| 文件系统 | `grep` | 使用安全正则表达式搜索可读文本 | 可读 Workspace Mount |
| 命令执行 | `bash` | 执行已批准的非交互式本地命令 | 宿主提供的 Bash Executor |
| 用户交互 | `ask_user` | 提出结构化问题，并可暂停 Run | 宿主回调或 Runtime Suspension Binding |
| Skill | `skill` | 将已授权 Skill 加载到会话 | 宿主提供的 Skill Catalog |

文件系统 Tool 只能通过已授权的逻辑 `WorkspaceSession` 工作。`bash` 本身不会创建 Shell，
`skill` 也无法加载当前 Turn 所提供 Catalog 之外的内容。

## 使用 Tool 扩展 Gewu

### Python 原生 Tool

当前的扩展机制是进程内 Python `Tool`。`@tool` 装饰器根据 Python 类型标注与 Google 风格的
`Args:` 文档生成模型可见的 JSON Schema。生成的 Tool 必须显式加入每个 Turn 的 `ToolSet`，
不会发生全局注册。

```python
from gewu_agent_runtime.builtins import FILE_TOOLS
from gewu_agent_runtime.tools import ToolResult, ToolSetBuilder, tool


@tool(name="lookup_order", category="orders", allow_parallel=True)
async def lookup_order(order_id: str) -> ToolResult:
    """通过应用自己管理的服务查询订单。

    Args:
        order_id: 稳定的订单标识。
    """

    # `order_service` 是应用持有并已经完成授权的依赖。
    order = await order_service.get_order(order_id)
    return ToolResult(output={"order": order})


authorized_tool_set = (
    ToolSetBuilder(name="support-agent", version="v1")
    .extend(FILE_TOOLS)
    .add(lookup_order)
    .build()
)
```

宿主仍然负责认证调用者、授权操作、绑定凭据，并决定当前 Turn 对模型可见的 Tool。

### Model Context Protocol

当前版本**尚未实现** [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) Client。
Gewu 目前不会连接 MCP Server、发现远程 Tool，也不会通过 stdio 或 Streamable HTTP 转发
Tool Call。

在 MCP Client Adapter 可用之前，外部服务需要通过 Python 原生 Tool 暴露，并注入
`TurnBindings.tool_set`。MCP 是自然的未来适配边界：远程 Tool 定义可以被转换成 Gewu
`Tool`，然后进入相同的逐 Turn 授权和执行流水线。

## 架构

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
  |-- ConversationContextBuilder   持久化上下文重建
  |-- CompactionService            Micro Compact 与 Full Compact
  |-- AgentEngine                  模型提供商无关的模型与 Tool 循环
  |-- RuntimeStore                 Conversation、Message、Run、State 与 Compaction
  |-- RunLease                     跨进程 Run 协调
  `-- StateCache                   可丢弃的 Conversation State Cache
```

## 包结构

| 工程 | Python 包 | 职责 |
| --- | --- | --- |
| [`packages/agent-runtime`](packages/agent-runtime) | `gewu-agent-runtime` | Agent 执行、上下文、记忆、压缩、Tool、Workspace、Skill、Scene、持久化契约与适配器 |
| [`packages/gewu-core`](packages/gewu-core) | `gewu-core` | 配置、日志、数据库、Redis、HTTP、并发与共享基础设施 |

## 快速开始

Gewu 要求 Python 3.12 或更高版本，并使用 `uv workspace`。

```bash
git clone https://github.com/Simbaorz/gewu.git
cd gewu
uv sync --all-packages --all-extras
```

## 最小进程内示例

以下示例使用 Scripted Model，因此结果确定且不需要外部 API Key：

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

生产集成中，宿主需要将内存实现替换为已经授权的模型、Workspace、Tool、MySQL 与 Redis
Binding。所有权和集成契约请参阅
[`gewu-agent-runtime` 包指南](packages/agent-runtime/README.md)。

## Runtime 边界

Gewu 有意不负责：

- 处理入站 HTTP 或选择传输协议；
- 认证调用者或推导业务权限；
- 根据不可信请求字段选择租户资源；
- 根据 Workspace 路径推断组织层级；
- 将 Prompt 或 Tool Description 当作授权边界。

宿主必须在构造 `PreparedAgentTurn` 或 `TurnBindings` 前解决这些策略。

## 开发与验证

```bash
uv sync --all-packages --all-extras
uv run black packages tests --check
uv run ruff check packages tests
uv run mypy
uv run pytest
uv lock --check
uv build --all-packages
```

## 许可证

Gewu 使用 [MIT License](LICENSE) 开源。
