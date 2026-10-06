# Gewu

[English](README.md) | [简体中文](README.zh-CN.md)

**Gewu（格物）是一套用于在企业后端运行 AI Agent 的 Python Runtime。**

它把模型、Tool、知识 Workspace 和会话存储连接起来，负责一次 Agent 请求从开始、调用工具、
产生回答，到暂停、恢复和持久化的完整过程。你的应用仍然负责用户身份、业务权限、知识管理和
数据接口，Gewu 只运行已经授权给它的能力。

如果你正在搭建企业知识问答、智能客服或业务助手，Gewu 可以作为其中的 Agent 底座。

> **项目状态：** Alpha。Gewu 正在积极开发中，在首个稳定版本发布前，公开 API 可能发生
> 变化。

## 它解决什么问题

一次模型调用很容易，但把 Agent 放进真实的企业系统，还需要处理很多模型 API 不负责的事情：

- 这一轮允许使用哪个模型、哪些 Tool、哪部分知识？
- Tool Call、Tool Result 和图片怎样完整保存，并在下一轮恢复？
- 会话越来越长时，怎样控制上下文而不直接丢掉历史？
- 用户补充信息后，怎样恢复之前暂停的 Run？
- 多个用户同时请求时，怎样避免同一会话被并发修改？
- 服务重启或请求转移到另一台机器后，怎样继续原来的会话？

Gewu 将这些通用问题收敛到一个可嵌入的 Runtime 中，让上层应用专注于自己的业务和产品。

## 一次实际的知识问答怎样运行

假设用户问：

> 这笔订单符合公司的退款规则吗？

上层应用先完成业务判断：识别当前用户，选择他有权访问的“售后规则” Scene，挂载相应 Wiki，
再提供一个已经做过权限校验的订单查询 Tool。随后把这些能力交给 Gewu。

```text
用户问题
   |
   v
宿主应用
  |-- 确认用户身份与权限
  |-- 选择“售后规则” Scene
  |-- 挂载可读的规则 Wiki
  |-- 注入已授权的订单查询 Tool
  `-- 选择本轮模型
   |
   v
Gewu
  |-- 保存输入并创建 Run
  |-- 根据持久化历史重建上下文
  |-- 告诉模型当前 Scene 和需要遵循的 Skill
  |-- 让 Agent 使用 grep/read 查阅退款规则
  |-- 让 Agent 调用订单 Tool 获取实时状态
  |-- 必要时通过 ask_user 向用户补充提问
  `-- 保存 Tool Call、Tool Result、回答和会话状态
```

在这个过程中：

- Wiki 提供规则、说明和数据口径；
- Tool 提供订单、库存、客户、指标等实时事实；
- Gewu 负责让模型在受控范围内使用它们，并让会话可靠地持续下去。

Gewu 不理解“订单”或“退款”是什么。同样的 Runtime 也可以用于人事制度、产品手册、设备运维、
内部支持等场景。

## Gewu 具备哪些能力

### 1. 完整的 Agent 执行循环

Gewu 提供流式的模型与 Tool 循环：

- 把已经授权的 Tool Schema 发送给模型；
- 接收流式文本和 Tool Call；
- 执行 Tool，并把结果送回模型继续推理；
- 保留 Assistant、Tool Call、Tool Result 的完整结构；
- 限制最大迭代次数和安全上下文用量；
- 通过结构化事件返回增量文本、Tool 调用、最终回答、暂停或错误。

核心执行只依赖模型协议。当前提供 OpenAI 兼容 API、Anthropic 和中国联通开放服务适配器，
也可以接入宿主自己的模型实现。

### 2. 服务端会话与运行状态

Gewu 不把进程内的消息数组当作会话事实。它可以持久化：

- Conversation 与仅追加 Message；
- 每次 Invocation Run 及其生命周期；
- Tool Call、Tool Result 和 Attachment；
- 带版本的 Conversation State；
- 幂等记录；
- 等待用户回答的 Pending Ask；
- Context Compaction 版本。

MySQL 是参考事实存储；Redis 用于可丢弃的 State Cache 和分布式 Run Lease。Redis 丢失不会
删除已经完成的会话和 Run。项目也提供内存 Store，方便测试和进程内开发。

### 3. 上下文管理与两级压缩

每个 Turn 开始时，Gewu 都会从持久化记录重建模型上下文，并正确恢复消息、图片、Tool Call
与 Tool Result。调用模型前还会计算消息、图片、Tool Schema 和参数所占用的上下文。

当会话变长时，Gewu 提供两级处理：

1. **Micro Compact**：从模型视图中缩短较早的文件、搜索和命令结果，但不修改原始消息记录。
2. **Full Compact**：使用单独的、无 Tool 模型把较早的会话整理为累计摘要，并以带版本的方式
   原子提交。

因此，数据库中保留完整历史，模型看到的上下文则保持在可用范围内。
Full Compact 默认不启用，宿主需要显式提供压缩策略和压缩模型或 Model Provider。

当前 Gewu 所说的“记忆”，是持久化会话历史与 Runtime State。它尚未内置用户画像提取、跨会话
语义召回或基于 Embedding 的长期记忆。

### 4. 面向 Agent 的逻辑 Workspace

`WorkspaceSession` 是一个受权限约束的逻辑文件系统。模型只能看到逻辑路径，不需要知道知识
实际存放在本机目录、对象存储、数据库还是其他系统。

宿主可以在一次 Turn 中组合多个 Mount，并分别指定只读或读写权限。例如，上层应用可以把企业
公共知识、部门知识和用户空间组合成一个视图，再交给 Gewu 使用。

Gewu 本身不解释企业组织层级，也不决定知识如何继承或向下共享。宿主先完成这些权限计算，
Gewu 只执行最终得到的 Workspace 能力。

### 5. Scene 与 Skill

这两个对象解决不同问题：

| 对象 | 作用 | 例子 |
| --- | --- | --- |
| Scene | 告诉 Agent 当前应当在哪一组知识中工作 | 售后规则、财务制度、设备维修手册 |
| Skill | 告诉 Agent 应当怎样完成一类任务 | 先核对规则，再获取事实，最后给出结论和依据 |

Scene 包含名称、描述和 Workspace 根路径，也可以绑定一个必需的 Skill。用户选择 Scene 后，
Gewu 会通过 Meta Message 把名称、入口路径和可见的绑定 Skill 名称加入当前上下文，
其中不包含 Scene 描述或工作流指令。宿主的 System Prompt 负责定义 Scene 和绑定 Skill 的使用规则。
Gewu 不会把整个知识目录一次性塞给模型；Agent 使用文件 Tool 按需发现和阅读证据。

Skill 来自宿主已经过滤过的 Catalog。Gewu 只列出当前用户有权使用的 Skill，并在模型实际调用
`skill` Tool 时加载完整内容。

### 6. 面向多订阅方与多用户的运行边界

Runtime 中的所有者统一表示为：

```text
subscriber_id + principal_id + principal_type
```

它为多租户、多用户服务提供稳定的持久化作用域，避免一次操作跨越 `subscriber_id`。具体的
租户、部门、团队、角色和数据权限模型仍由宿主应用定义。

### 7. 可暂停和恢复的用户澄清

Agent 可以通过 `ask_user` 提出结构化问题。如果当前请求无法立即获得用户回答，Gewu 会保存
Pending Ask State、暂停 Run，并在收到匹配的回答后精确恢复，而不是重新开始整轮对话。

## 知识怎样进入 Gewu

对于已经整理成 Wiki 或 Markdown 的知识，可以把它挂载到 Workspace，让 Agent 使用 `list`、
`glob`、`grep` 和 `read` 主动查找。这样能够保留目录、文档和章节结构，内容修改后也不必强制
重新切片和向量化。

Gewu 不是一套内置 RAG 引擎，也不要求所有知识都只能通过文件查找。如果你的语料需要全文检索、
向量召回、重排或其他检索方式，可以把相应能力包装成 Tool，和普通业务 Tool 一样按 Turn 授权。

## 内置 Tool

Gewu 当前提供 11 个可选、业务中立的 Tool。它们不会被全局注册或默认启用，宿主必须为每个
Turn 显式构造 `ToolSet`。

| 分类 | Tool | 用途 |
| --- | --- | --- |
| 文件 | `read` | 有界读取文本文件 |
| 文件 | `write` | 创建或完整写入文本文件 |
| 文件 | `append` | 追加文本 |
| 文件 | `edit` | 精确替换文本 |
| 文件 | `delete` | 删除指定文件或目录 |
| 查找 | `list` | 列出逻辑目录 |
| 查找 | `glob` | 按路径模式查找文件 |
| 查找 | `grep` | 使用安全正则搜索文本 |
| 执行 | `bash` | 调用宿主提供的非交互命令执行器 |
| 交互 | `ask_user` | 向用户提出结构化问题并可暂停 Run |
| Skill | `skill` | 加载已经授权的 Skill |

文件 Tool 只能通过当前 `WorkspaceSession` 操作。`bash` 只是接口契约，不会自行创建 Shell；
只有宿主提供 Bash Executor 后才能使用，企业知识应用也可以完全不启用它。

## 接入自己的业务 Tool

当前可以使用 `@tool` 把应用服务包装为 Python Tool，再加入本轮的 `ToolSet`：

```python
from gewu_agent_runtime.builtins import FILE_TOOLS
from gewu_agent_runtime.tools import ToolResult, ToolSetBuilder, tool


@tool(name="lookup_order", category="orders")
async def lookup_order(order_id: str) -> ToolResult:
    """查询当前用户有权查看的订单。

    Args:
        order_id: 订单编号。
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

`authorized_order_service`、数据库凭据和数据权限都属于宿主。Gewu 只调用已经注入的 Tool，
不会绕过它直接访问业务数据库。

### MCP 状态

当前版本**尚未实现** [Model Context Protocol（MCP）](https://modelcontextprotocol.io/) Client，
还不能直接连接 MCP Server、发现远程 Tool，或通过 stdio、Streamable HTTP 执行 Tool Call。

现阶段外部能力需要包装成 Python Tool。未来 MCP Adapter 也会进入相同的逐 Turn 授权和 Tool
执行流程，而不是绕过宿主权限。

## Gewu 与宿主应用的边界

| Gewu 负责 | 宿主应用负责 |
| --- | --- |
| 模型与 Tool 执行循环 | HTTP、WebSocket、消息渠道和 UI |
| 会话、Message、Run 与 State 持久化 | 登录、用户、组织与角色 |
| 上下文重建、计量和压缩 | 模型与 Tool 的业务授权 |
| 逻辑 Workspace 能力执行 | Wiki 编辑、导入、版本与发布 |
| Scene/Skill 的 Runtime 投影 | Scene/Skill 的管理与权限过滤 |
| Ask 暂停和恢复 | 业务数据库与外部系统接入 |
| Run Lease 与并发保护 | 审计、运营和产品策略 |

Gewu 不会把 Prompt、Meta Message 或 Tool Description 当作安全边界。所有可信权限都必须在宿主
创建 `PreparedAgentTurn` 或 `TurnBindings` 之前确定。

## 快速开始

Gewu 要求 Python 3.12 或更高版本，并使用 `uv workspace`。

```bash
git clone https://github.com/Simbaorz/gewu.git
cd gewu
uv sync --all-packages --all-extras
```

以下最小示例使用内存 Store 和 Scripted Model，不需要 API Key：

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

生产集成需要由宿主提供已经授权的模型、Workspace、ToolSet，以及所需的 MySQL 和 Redis
Binding。详细契约请参阅 [`gewu-agent-runtime` 包文档](packages/agent-runtime/README.md)。

## 包结构

| 工程 | Python 包 | 职责 |
| --- | --- | --- |
| [`packages/agent-runtime`](packages/agent-runtime) | `gewu-agent-runtime` | Agent 执行、上下文、记忆、压缩、Tool、Workspace、Skill、Scene、持久化契约与适配器 |
| [`packages/gewu-core`](packages/gewu-core) | `gewu-core` | 配置、日志、数据库、Redis、HTTP、并发与共享基础设施 |

## 当前没有提供的能力

- 可直接使用的 Web 聊天与管理后台；
- 用户、组织和业务权限系统；
- Wiki 编辑和通用文档导入产品；
- 内置向量知识库与 RAG 流水线；
- 语义长期记忆和跨会话用户画像；
- MCP Client；
- Subagent 编排。

这些能力可以由上层产品实现或通过 Tool 接入，但不应被理解为 Gewu 当前已经提供。

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
