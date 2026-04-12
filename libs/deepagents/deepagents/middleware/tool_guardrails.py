"""Tool call guardrails middleware for deep agents."""

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeAlias

from langchain.agents.middleware.types import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

GuardrailResult: TypeAlias = bool | Awaitable[bool]
ToolInputGuardrail: TypeAlias = Callable[[ToolCallRequest, str], GuardrailResult]
ToolOutputGuardrail: TypeAlias = Callable[[ToolMessage | Command[Any], str], GuardrailResult]


class ToolGuardrailsMiddleware(AgentMiddleware):
    """Apply tool input/output guardrail checks around tool execution.

    Each input guardrail is called with `(request, agent_name)`.
    Each output guardrail is called with `(tool_result, agent_name)`.

    Returning `False` blocks the tool call and returns an error `ToolMessage`.
    """

    def __init__(
        self,
        *,
        agent_name: str,
        tool_input_guardrails: Sequence[ToolInputGuardrail] | None = None,
        tool_output_guardrails: Sequence[ToolOutputGuardrail] | None = None,
    ) -> None:
        """Initialize guardrails middleware.

        Args:
            agent_name: Name of the agent this middleware is attached to.
            tool_input_guardrails: Guardrails to validate tool call payloads.
            tool_output_guardrails: Guardrails to validate tool results.
        """
        super().__init__()
        self._agent_name = agent_name
        self._tool_input_guardrails = list(tool_input_guardrails or [])
        self._tool_output_guardrails = list(tool_output_guardrails or [])

    @staticmethod
    async def _await_guardrail(result: Awaitable[bool]) -> bool:
        return bool(await result)

    @staticmethod
    def _resolve_sync_guardrail_result(result: GuardrailResult) -> bool:
        if not inspect.isawaitable(result):
            return bool(result)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(ToolGuardrailsMiddleware._await_guardrail(result))

        msg = "Async guardrails are not supported in sync tool execution. Use async agent invocation instead."
        raise RuntimeError(msg)

    @staticmethod
    async def _resolve_async_guardrail_result(result: GuardrailResult) -> bool:
        if inspect.isawaitable(result):
            return bool(await result)
        return bool(result)

    def _run_input_guardrails(self, request: ToolCallRequest) -> bool:
        return all(self._resolve_sync_guardrail_result(guardrail(request, self._agent_name)) for guardrail in self._tool_input_guardrails)

    async def _arun_input_guardrails(self, request: ToolCallRequest) -> bool:
        for guardrail in self._tool_input_guardrails:
            if not await self._resolve_async_guardrail_result(guardrail(request, self._agent_name)):
                return False
        return True

    def _run_output_guardrails(self, result: ToolMessage | Command[Any]) -> bool:
        return all(self._resolve_sync_guardrail_result(guardrail(result, self._agent_name)) for guardrail in self._tool_output_guardrails)

    async def _arun_output_guardrails(self, result: ToolMessage | Command[Any]) -> bool:
        for guardrail in self._tool_output_guardrails:
            if not await self._resolve_async_guardrail_result(guardrail(result, self._agent_name)):
                return False
        return True

    def _blocked_message(self, *, request: ToolCallRequest, phase: str) -> ToolMessage:
        return ToolMessage(
            content=(f"Tool call blocked by {phase} guardrail for agent '{self._agent_name}'. Adjust the request or update guardrail policy."),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run input guardrails before execution and output guardrails after."""
        if self._tool_input_guardrails and not self._run_input_guardrails(request):
            return self._blocked_message(request=request, phase="input")

        result = handler(request)

        if self._tool_output_guardrails and not self._run_output_guardrails(result):
            return self._blocked_message(request=request, phase="output")

        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async variant of tool guardrail checks."""
        if self._tool_input_guardrails and not await self._arun_input_guardrails(request):
            return self._blocked_message(request=request, phase="input")

        result = await handler(request)

        if self._tool_output_guardrails and not await self._arun_output_guardrails(result):
            return self._blocked_message(request=request, phase="output")

        return result
