"""Business-data Tool mechanics over a subscriber-authorized capability."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

import pytest

import gewu_agent_runtime.builtins.business_data as business_data_module
from gewu_agent_runtime.builtins import (
    BusinessDataQueryExecution,
    BusinessDataQueryOutput,
    BusinessDataQueryRequest,
    business_data_tool,
    query_business_data_template,
)
from gewu_agent_runtime.tools import ToolContext
from gewu_agent_runtime.workspace import WorkspaceSession


class MemoryBusinessDataCapability:
    def __init__(self, output: BusinessDataQueryOutput) -> None:
        self.output = output
        self.requests: list[BusinessDataQueryRequest] = []

    async def query(self, request: BusinessDataQueryRequest) -> BusinessDataQueryOutput:
        self.requests.append(request)
        return self.output


class FailingBusinessDataCapability:
    async def query(self, request: BusinessDataQueryRequest) -> BusinessDataQueryOutput:
        del request
        raise RuntimeError("business-data-private-secret")


def _runtime(workspace: WorkspaceSession) -> ToolContext:
    return ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )


def test_business_data_model_contract_exactly_matches_subscriber() -> None:
    payload = {
        "description": query_business_data_template.description,
        "input_schema": query_business_data_template.input_schema,
        "category": query_business_data_template.category,
        "writes": query_business_data_template.writes_workspace,
        "allow_parallel": query_business_data_template.allow_parallel,
        "retry": query_business_data_template.retry_on_failure,
        "max_retries": query_business_data_template.max_retries,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert digest == "3ef9ec3a9d26ffc67e36c01ed865661199f6aec33043fbf4cdb530ae654943c9"


async def test_unbound_business_data_tool_preserves_subscriber_error_projection(
    workspace: WorkspaceSession,
) -> None:
    result = await query_business_data_template.execute(
        {
            "raw_sql": "select * from users",
            "purpose": "verify user state",
        },
        _runtime(workspace),
    )

    assert result.is_error is True
    assert isinstance(result.output, BusinessDataQueryOutput)
    assert result.output.execution.success is False
    assert result.output.execution.message == result.output.error
    assert result.output_payload() == {
        "success": False,
        "summary": "查询失败：Business data query runtime configuration is unavailable.",
        "purpose": "verify user state",
        "error": "Business data query runtime configuration is unavailable.",
    }


async def test_business_data_validates_limits_redacts_and_projects_result(
    workspace: WorkspaceSession,
) -> None:
    capability = MemoryBusinessDataCapability(
        BusinessDataQueryOutput(
            success=True,
            execution=BusinessDataQueryExecution(success=True, status="1"),
            columns=["id", "phone"],
            rows=[
                {"id": 1, "phone": "13800000000"},
                {"id": 2, "phone": "13900000000"},
                {"id": 3, "phone": "13700000000"},
            ],
        )
    )
    tool_value = business_data_tool(
        capability,
        database_key="default_db",
        row_limit=2,
        request_id_factory=lambda: "request-1",
    )

    result = await tool_value.execute(
        {
            "raw_sql": "select id, phone from users limit 1000;",
            "purpose": "verify user state",
        },
        _runtime(workspace),
    )

    assert capability.requests[0].raw_sql == "select id, phone from users limit 2"
    assert result.is_error is False
    assert isinstance(result.output, BusinessDataQueryOutput)
    assert result.output.database_key == "default_db"
    assert result.output.rows == [
        {"id": 1, "phone": "***"},
        {"id": 2, "phone": "***"},
    ]
    assert result.output.truncated is True
    assert result.output_payload() == {
        "success": True,
        "summary": "查询成功，返回 2 行业务数据。 结果已截断。",
        "purpose": "verify user state",
        "columns": ["id", "phone"],
        "rows": [
            {"id": 1, "phone": "***"},
            {"id": 2, "phone": "***"},
        ],
        "row_count": 2,
        "truncated": True,
    }
    assert result.raw_output_payload()["request_id"] == "request-1"


async def test_business_data_rejects_non_read_only_or_multiple_sql(
    workspace: WorkspaceSession,
) -> None:
    capability = MemoryBusinessDataCapability(BusinessDataQueryOutput(success=True))
    tool_value = business_data_tool(capability)

    mutation = await tool_value.execute(
        {"raw_sql": "update users set active = 1", "purpose": "change"},
        _runtime(workspace),
    )
    multiple = await tool_value.execute(
        {"raw_sql": "select 1; select 2", "purpose": "check"},
        _runtime(workspace),
    )
    dangerous_cte = await tool_value.execute(
        {"raw_sql": "with changed as (delete from users) select 1", "purpose": "check"},
        _runtime(workspace),
    )

    assert "only allows SELECT or WITH" in mutation.output_payload()["error"]
    assert "multiple SQL statements" in multiple.output_payload()["error"]
    assert "forbidden keyword" in dangerous_cte.output_payload()["error"]
    assert capability.requests == []


async def test_business_data_preserves_small_limit_and_clamps_large_limit(
    workspace: WorkspaceSession,
) -> None:
    capability = MemoryBusinessDataCapability(BusinessDataQueryOutput(success=True))
    tool_value = business_data_tool(capability, row_limit=20)

    await tool_value.execute(
        {"raw_sql": "select * from users limit 5", "purpose": "small"},
        _runtime(workspace),
    )
    await tool_value.execute(
        {"raw_sql": "select * from users limit 1000", "purpose": "large"},
        _runtime(workspace),
    )

    assert [request.raw_sql for request in capability.requests] == [
        "select * from users limit 5",
        "select * from users limit 20",
    ]


async def test_business_data_capability_failure_returns_exception_body_like_subscriber(
    workspace: WorkspaceSession,
) -> None:
    result = await business_data_tool(
        FailingBusinessDataCapability(),
        database_key="default_db",
        request_id_factory=lambda: "request-1",
    ).execute(
        {"raw_sql": "select 1", "purpose": "verify"},
        _runtime(workspace),
    )

    assert result.is_error is True
    assert result.output_payload() == {
        "success": False,
        "summary": "查询失败：Business data query failed: business-data-private-secret",
        "purpose": "verify",
        "error": "Business data query failed: business-data-private-secret",
    }
    assert result.raw_output_payload()["database_key"] == "default_db"
    assert result.raw_output_payload()["request_id"] == "request-1"


async def test_business_data_result_obeys_serialized_byte_budget(
    workspace: WorkspaceSession,
) -> None:
    capability = MemoryBusinessDataCapability(
        BusinessDataQueryOutput(
            success=True,
            execution=BusinessDataQueryExecution(success=True),
            columns=["id", "content"],
            rows=[{"id": index, "content": f"row-{index}-" + "x" * 700} for index in range(5)],
        )
    )
    result = await business_data_tool(
        capability,
        row_limit=10,
        max_result_bytes=2048,
    ).execute(
        {"raw_sql": "select id, content from records", "purpose": "verify records"},
        _runtime(workspace),
    )

    assert isinstance(result.output, BusinessDataQueryOutput)
    assert 0 < result.output.row_count < 5
    assert result.output.truncated is True
    assert (
        len(
            json.dumps(
                result.raw_output_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        <= 2048
    )
    assert (
        len(
            json.dumps(
                result.output_payload(),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        <= 2048
    )


async def test_business_data_oversized_metadata_returns_stable_bounded_error(
    workspace: WorkspaceSession,
) -> None:
    marker = "oversized-column-" + "m" * 10_000
    capability = MemoryBusinessDataCapability(
        BusinessDataQueryOutput(
            success=True,
            execution=BusinessDataQueryExecution(success=True),
            columns=[marker],
        )
    )
    result = await business_data_tool(
        capability,
        max_result_bytes=2048,
    ).execute(
        {"raw_sql": "select content from records", "purpose": "verify records"},
        _runtime(workspace),
    )

    assert result.is_error is True
    assert isinstance(result.output, BusinessDataQueryOutput)
    assert result.output.error == ("Business data query result exceeds the configured byte limit.")
    assert marker not in json.dumps(result.raw_output_payload(), ensure_ascii=False)


async def test_business_data_oversized_single_cell_never_leaves_tool(
    workspace: WorkspaceSession,
) -> None:
    marker = "must-not-leave-tool-" + "z" * 10_000
    capability = MemoryBusinessDataCapability(
        BusinessDataQueryOutput(
            success=True,
            execution=BusinessDataQueryExecution(success=True),
            columns=["content"],
            rows=[{"content": marker}],
        )
    )

    result = await business_data_tool(
        capability,
        max_result_bytes=2048,
    ).execute(
        {"raw_sql": "select content from records", "purpose": "verify records"},
        _runtime(workspace),
    )

    assert isinstance(result.output, BusinessDataQueryOutput)
    assert result.output.success is True
    assert result.output.rows == []
    assert result.output.row_count == 0
    assert result.output.truncated is True
    assert marker not in json.dumps(result.raw_output_payload(), ensure_ascii=False)
    assert marker not in json.dumps(result.output_payload(), ensure_ascii=False)


async def test_business_data_projection_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
    workspace: WorkspaceSession,
) -> None:
    capability = MemoryBusinessDataCapability(
        BusinessDataQueryOutput(
            success=True,
            execution=BusinessDataQueryExecution(success=True),
            columns=["id"],
            rows=[{"id": 1}],
        )
    )
    original_normalize = business_data_module._normalize_output

    def slow_normalize(*args: object, **kwargs: object) -> BusinessDataQueryOutput:
        time.sleep(0.05)
        return original_normalize(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(business_data_module, "_normalize_output", slow_normalize)
    task = asyncio.create_task(
        business_data_tool(capability).execute(
            {"raw_sql": "select id from records", "purpose": "verify records"},
            _runtime(workspace),
        )
    )
    heartbeats = 0
    while not task.done():
        await asyncio.sleep(0.005)
        heartbeats += 1
    result = await task

    assert result.is_error is False
    assert heartbeats >= 3
