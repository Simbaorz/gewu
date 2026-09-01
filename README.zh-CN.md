# Gewu

[English](README.md) | [简体中文](README.zh-CN.md)

**Gewu（格物）是一套模型提供商无关、可嵌入的 Python Agent Runtime，用于构建受治理、
有状态、运行于服务端的 AI Agent。**

Gewu 将宿主已经授权的模型、ToolSet、逻辑 Workspace、Skill 与 Scene Catalog，以及持久化
后端组织成可持续运行的 Agent 会话。它负责模型与 Tool 执行循环、上下文构建、会话记忆、
上下文压缩、可暂停与恢复的用户交互，以及 Run 协调；身份、权限、组织策略、模型访问、
业务数据与传输协议仍由宿主应用掌控。

Gewu 尤其适合构建可私有化部署的组织知识 Agent。宿主可以将经过整理的 Wiki 暴露为逻辑
Workspace，用 Scene 划分知识范围，用 Skill 提供可复用的工作方法，再通过授权 Tool 接入实时
事实。Agent 由此可以按需查阅相关知识、获取当前数据，而无需将业务规则或凭据放入 Runtime。

这套体系的整体目标，是让企业能够以可控的方式构建具备实际能力的知识 Agent，默认不依赖
通用桌面 Agent，也不强制引入向量数据库。

> **项目状态：** Alpha。Gewu 正在积极开发中，在首个稳定版本发布前，公开 API 可能发生
> 变化。

## Gewu 在体系中的位置

Gewu 是 Runtime 层，不是一套完整的知识库产品或 SaaS 应用。

```text
Web / Admin / API / 消息渠道
                  |
                  v
宿主应用
  |-- 认证用户并解析组织策略
  |-- 授权模型、Scene、Skill、Tool 与 Workspace Mount
  |-- 管理 Wiki 内容与实时业务数据服务
  `-- 准备一次 Agent Turn
                  |
                  v
Gewu Agent Runtime
  |-- 执行模型与 Tool 循环
  |-- 构建并压缩模型上下文
  |-- 持久化 Conversation、Message、Run 与 Runtime State
  |-- 将 Scene 与 Skill 投影到会话上下文
  `-- 协调并发执行和暂停中的 Run
                  |
       +----------+-----------+
       |          |           |
       v          v           v
      模型     Workspace      授权 Tool
                            （实时事实与操作）
```

这条边界是刻意设计的：订阅方应用可以依赖 Gewu，但 Gewu 不得反向依赖订阅方的身份系统、
业务权限、组织层级、Wiki Schema 或数据接口。

## 核心能力

| 能力 | Gewu 提供的内容 |
| --- | --- |
| Agent 执行 | 流式、模型提供商无关的模型与 Tool 循环，包含结构化生命周期事件、有界迭代，以及明确的完成、暂停与失败状态 |
| 持久化会话 | 仅追加 Message、带版本的 Conversation State、Invocation Run、幂等记录、Attachment、Pending Ask State 与不可变压缩版本 |
| 上下文管理 | 从持久化记录重建上下文，保留 Tool Call 结构，统计多模态内容与 Token，并保护模型上下文窗口 |
| 上下文压缩 | 用 Micro Compact 清理低价值历史 Tool Result，再用模型生成的 Full Compact 维护累计会话摘要 |
| 逻辑 Workspace VFS | 使用能力约束的逻辑 Mount 隐藏物理存储，并将多个经宿主授权的知识来源组合为同一个 Agent 视图 |
| Scene | 带有逻辑根路径、描述以及可选 Required Skill 工作流程的授权知识范围 |
| Skill | 仅在获得授权且实际需要时加载到会话中的可复用工作指令 |
| Tool | 十一个可选、业务中立的文件、搜索、受控执行、用户澄清与 Skill Tool，以及 Python 原生 Tool 扩展机制 |
| 用户交互 | 通过 `ask_user` 进行结构化澄清，并支持持久化暂停和精确恢复原 Run |
| 所有权与协调 | 统一的 `subscriber_id + principal_id + principal_type` 所有权、MySQL 持久化、Redis State Cache 与分布式 Run Lease |
| 模型接入 | 基于协议的模型执行，以及 OpenAI 兼容 API、Anthropic 和中国联通开放服务的可选适配器 |

## 知识、工作方法与实时事实

Gewu 有意将几种不同性质的上下文分开：

```text
Scene       Agent 应当在哪里工作
Skill       Agent 应当怎样开展工作
Workspace   Agent 可以查阅哪些知识与证据
Tool        宿主允许 Agent 获取哪些实时事实或执行哪些操作
```

### Scene 作为知识范围

Scene 指向当前 Workspace 中一个已经授权的逻辑路径。宿主选择 Scene 后，Gewu 会加入一条
Meta Message，向模型说明 Scene 名称、根路径、描述与绑定的工作流程。Gewu 不会把整个 Scene
一次性复制进模型上下文，而是要求 Agent 发现相关入口，只读取解决当前问题所需的证据。

如果 Scene 绑定了必需的 Skill，Runtime 会要求 Agent 在处理该 Scene 前加载它。这样，上层
应用可以把一组知识与稳定的工作方法组合起来，而不必把所有方法固化到全局 System Prompt。

### Skill 作为可复用工作方法

Skill 保存任务相关的工作指令，并由宿主已经完成权限过滤的 Catalog 提供。Gewu 只向模型列出
获得授权的 Skill，通过 `skill` Tool 按需加载完整内容，记录调用状态，并保持 Skill 内容与
当前有效模型上下文一致。

### 不强制向量索引的 Wiki 知识

对于经过整理的 Wiki 或 Markdown 知识，宿主可以将其挂载到 Workspace，让 Agent 使用
`list`、`glob`、`grep` 与 `read` 主动浏览。这种方式保留目录、文档与章节结构，并让内容修改
立即可见，不强制经过切片、Embedding 和重新索引流水线。

Gewu 不是内置的 RAG 引擎，也不认为文件导航能够取代所有语料上的语义检索。当知识规模或
文档格式需要时，宿主可以将全文检索、向量召回、重排或任意混合检索策略作为授权 Tool 注入
Runtime。

### 实时业务事实留在 Runtime 之外

订单、库存、客户、指标、工单等实时事实属于宿主应用。宿主通过边界明确、已经授权的 Tool
暴露这些能力，而不是让 Gewu 理解具体业务 Schema，或让模型不受限制地访问数据库。业务数据
访问、凭据、行级权限和审计策略仍由真正拥有这些数据的系统负责。

## 持久化上下文与记忆

### 从持久化记录重建上下文

每个 Turn 中，Gewu 都会根据持久化 Runtime 记录重建模型输入，而不是将内存消息列表视为
事实来源。上下文流水线会：

- 从最新一次已提交的压缩边界开始；
- 通过有界分页加载剩余消息历史；
- 恢复 User、Assistant、Tool Call 与 Tool Result 结构；
- 注入 System Prompt、累计摘要与已经授权的 Meta Message；
- 在重建期间避免重复加入当前输入；
- 将历史图片表示为持久化 Attachment 引用；
- 在调用模型前估算消息、图片、Tool 定义与 Tool 参数占用的 Token。

### 持久化会话记忆

Gewu 持久化继续运行服务端 Agent 会话所需的操作记忆：

- Conversation 元数据与仅追加 Message 日志；
- 带版本的 Conversation State；
- Invocation Run 与幂等记录；
- 用于暂停和恢复流程的 Pending Ask State；
- 文件读取状态与 Skill 调用状态；
- 不可变的累计压缩版本。

MySQL 是参考持久化实现。项目提供用于测试和嵌入式开发的内存 Store。Redis 是可选、可丢弃的
State Cache 与分布式 Run Lease 后端，不是会话事实存储。

在当前版本中，**记忆指持久化会话历史与 Runtime State**。Gewu 尚未提供基于 Embedding 的
语义记忆、用户画像提取或跨会话召回。

### 两级上下文压缩

1. **Micro Compact** 将较早且执行成功的文件、搜索和命令类 Tool Result 替换为模型投影中的
   简短占位符，原始的仅追加记录不会被修改。
2. **Full Compact** 使用宿主授权且不携带 Tool 的模型，将较早的会话前缀替换为累计摘要。
   压缩记录带版本，保留序列边界与模型元数据，并与相关 Runtime State 变更原子提交。

默认 Full Compact 策略在模型上下文窗口使用率达到 75% 时触发，以 50% 为压缩目标，并将
90% 视为安全硬限制。Full Compact 需要显式启用：宿主必须同时提供压缩策略和压缩模型或模型
Provider。

## 授权 Workspace 与所有权

`WorkspaceSession` 向 Agent 提供一个已经完成授权的逻辑文件系统视图。宿主可以把租户、团队、
共享、用户或应用持有的内容作为不同 Mount 组合起来，并为每个 Mount 分配只读或读写能力。
最长前缀路由和逻辑路径让模型可见的 Tool Call 不需要了解后端物理存储。

Gewu 不解释租户、省、市、团队或用户等组织层级。宿主先解析继承或向下共享策略，再为当前
Turn 构造实际可见的 Workspace 与 Catalog。Runtime 记录统一使用
`subscriber_id + principal_id + principal_type` 确定作用域，持久化操作拒绝跨订阅方访问。

## Runtime 执行流程

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
              |-- 准备 Scene、Skill 与 Tool 上下文
              |-- 执行 Micro Compact 与可选的 Full Compact
              |-- 执行流式模型与 Tool 循环
              `-- 持久化 Tool Call、结果、输出与 Runtime State
```

## 内置 Tool

任何 Tool 都不会被全局注册或默认启用；宿主需要为每个 Turn 显式选择已经授权的 `ToolSet`。

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
| 命令执行 | `bash` | 执行已经批准的非交互式本地命令 | 宿主提供的 Bash Executor |
| 用户交互 | `ask_user` | 提出结构化问题，并可暂停 Run | 宿主回调或 Runtime Suspension Binding |
| Skill | `skill` | 将已经授权的 Skill 加载到会话 | 宿主提供的 Skill Catalog |

文件系统 Tool 只能通过已经授权的 `WorkspaceSession` 工作。`bash` 本身不会创建 Shell，它只是
一份 Tool 契约，必须绑定宿主提供的 Executor；知识应用完全可以不启用它。`skill` Tool 也无法
加载当前 Turn 所提供 Catalog 之外的内容。

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
`TurnBindings.tool_set`。MCP 是未来接入第三方 Tool 的标准适配边界，但它仍会遵循相同的宿主
授权和逐 Turn ToolSet 模型。

## 架构与包结构

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

### 最小进程内示例

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

## 有意保留的边界

Gewu 有意不负责：

- 提供面向最终用户的 Web 或管理后台产品；
- 处理入站 HTTP 或选择传输协议；
- 认证调用者或推导组织与业务权限；
- 管理 Wiki 编辑、文档导入或业务 Schema；
- 根据不可信请求字段选择租户资源；
- 根据 Workspace 路径推断组织层级；
- 提供内置向量检索或语义长期记忆；
- 在当前版本连接 MCP Server；
- 将 Prompt、Meta Message 或 Tool Description 当作授权边界。

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
