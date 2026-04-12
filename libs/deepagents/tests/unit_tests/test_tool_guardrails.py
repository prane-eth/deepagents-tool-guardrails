"""Unit tests for tool guardrails middleware."""

from types import SimpleNamespace
from typing import Any, cast

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents.middleware.tool_guardrails import ToolGuardrailsMiddleware


def _request() -> ToolCallRequest:
    return cast(
        "ToolCallRequest",
        SimpleNamespace(
            tool_call={
                "id": "tc1",
                "name": "execute",
                "args": {"command": "ls"},
                "type": "tool_call",
            }
        ),
    )


def _handler(request: ToolCallRequest) -> ToolMessage:
    _ = request
    return ToolMessage(content="ok", name="execute", tool_call_id="tc1")


class TestToolGuardrailsMiddleware:
    def test_wrap_tool_call_supports_async_input_guardrail(self) -> None:
        async def allow_input(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return True

        middleware = ToolGuardrailsMiddleware(agent_name="deep-agent", tool_input_guardrails=[allow_input])

        result = middleware.wrap_tool_call(_request(), _handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"

    async def test_awrap_tool_call_supports_async_input_guardrail(self) -> None:
        async def deny_input(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return False

        async def handler(request: ToolCallRequest) -> ToolMessage:
            _ = request
            return ToolMessage(content="ok", name="execute", tool_call_id="tc1")

        middleware = ToolGuardrailsMiddleware(agent_name="deep-agent", tool_input_guardrails=[deny_input])

        result = await middleware.awrap_tool_call(_request(), handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "input guardrail" in str(result.content)

    async def test_awrap_tool_call_supports_async_output_guardrail(self) -> None:
        async def deny_output(tool_result: ToolMessage | Command[Any], agent_name: str) -> bool:
            _ = (tool_result, agent_name)
            return False

        async def handler(request: ToolCallRequest) -> ToolMessage:
            _ = request
            return ToolMessage(content="ok", name="execute", tool_call_id="tc1")

        middleware = ToolGuardrailsMiddleware(agent_name="deep-agent", tool_output_guardrails=[deny_output])

        result = await middleware.awrap_tool_call(_request(), handler)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "output guardrail" in str(result.content)

    async def test_awrap_tool_call_accepts_mixed_sync_and_async_guardrails(
        self,
    ) -> None:
        def allow_input(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return True

        async def allow_output(tool_result: ToolMessage | Command[Any], agent_name: str) -> bool:
            _ = (tool_result, agent_name)
            return True

        async def handler(request: ToolCallRequest) -> ToolMessage:
            _ = request
            return ToolMessage(content="ok", name="execute", tool_call_id="tc1")

        middleware = ToolGuardrailsMiddleware(
            agent_name="deep-agent",
            tool_input_guardrails=[allow_input],
            tool_output_guardrails=[allow_output],
        )

        result = await middleware.awrap_tool_call(_request(), handler)

        assert isinstance(result, ToolMessage)
        assert result.content == "ok"
