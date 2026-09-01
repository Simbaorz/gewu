# Gewu 核心 Monorepo 约定

本仓库是基于 `uv workspace` 的业务中立 Agent Runtime Monorepo。

## Workspace

- `packages/gewu-core/`：配置、日志、数据库、Redis、HTTP 与通用基础设施。
- `packages/agent-runtime/`：业务中立的 Agent Runtime、Tool、Workspace、Skill、Scene 与执行协调。
- 订阅方只能依赖 Gewu，Gewu 不得反向依赖订阅方。

## 工程边界

- Runtime Core 不处理入站 HTTP，不决定调用者身份或业务权限。
- Runtime 所有者统一使用 `subscriber_id + principal_id + principal_type`。
- Runtime Workspace 只表达逻辑路径、挂载和能力，不感知业务组织层级。
- `gewu-core` 不包含 Agent 执行规则或订阅方业务。

## 开发与验证

- Python 最低版本为 3.12；手工编辑使用 `apply_patch`。
- 不修改生成目录、真实配置或无关锁文件。
- 新行为先补测试，再运行：

```bash
uv run black packages tests --check
uv run ruff check packages tests
uv run mypy
uv run pytest
uv lock --check
uv build --all-packages
```
