"""Unit tests for deepagents.graph module."""

from unittest.mock import patch
from typing import Any

from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents._version import __version__
from deepagents.graph import create_deep_agent
from deepagents.middleware.tool_guardrails import ToolGuardrailsMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel


class TestCreateDeepAgentMetadata:
    """Tests for metadata on the compiled graph."""

    def test_versions_metadata_contains_sdk_version(self) -> None:
        """`create_deep_agent` should attach SDK version in metadata.versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        versions = agent.config["metadata"]["versions"]
        assert versions["deepagents"] == __version__

    def test_ls_integration_metadata_preserved(self) -> None:
        """`ls_integration` should still be present alongside versions."""
        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
        agent = create_deep_agent(model=model)
        assert agent.config is not None
        assert agent.config["metadata"]["ls_integration"] == "deepagents"


class TestCreateDeepAgentGuardrails:
    """Tests for tool guardrails passthrough behavior."""

    def test_passes_tool_guardrails_to_main_agent_constructor(self) -> None:
        """Top-level guardrails should be forwarded to the main `create_agent` call."""

        def allow_tool_inputs(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return True

        def allow_tool_outputs(tool_result: ToolMessage | Command[Any], agent_name: str) -> bool:
            _ = (tool_result, agent_name)
            return True

        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

        class _FakeCompiledAgent:
            def __init__(self) -> None:
                self.config: dict | None = None

            def with_config(self, config: dict) -> "_FakeCompiledAgent":
                self.config = config
                return self

        fake_compiled_agent = _FakeCompiledAgent()

        with patch("deepagents.graph.create_agent", return_value=fake_compiled_agent) as mock_create_agent:
            create_deep_agent(
                model=model,
                tool_input_guardrails=[allow_tool_inputs],
                tool_output_guardrails=[allow_tool_outputs],
            )

        assert mock_create_agent.called
        _, kwargs = mock_create_agent.call_args
        guardrail_middleware = next((m for m in kwargs["middleware"] if isinstance(m, ToolGuardrailsMiddleware)), None)
        assert guardrail_middleware is not None
        assert guardrail_middleware._tool_input_guardrails == [allow_tool_inputs]
        assert guardrail_middleware._tool_output_guardrails == [allow_tool_outputs]

    def test_subagent_guardrails_inherit_and_override_defaults(self) -> None:
        """Top-level guardrails should flow to subagents unless subagent overrides them."""

        def default_input(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return True

        def default_output(tool_result: ToolMessage | Command[Any], agent_name: str) -> bool:
            _ = (tool_result, agent_name)
            return True

        def override_input(request: ToolCallRequest, agent_name: str) -> bool:
            _ = (request, agent_name)
            return True

        model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))

        class _FakeCompiledAgent:
            def __init__(self) -> None:
                self.config: dict | None = None

            def with_config(self, config: dict) -> "_FakeCompiledAgent":
                self.config = config
                return self

        with (
            patch("deepagents.graph.create_agent", return_value=_FakeCompiledAgent()),
            patch("deepagents.middleware.subagents.create_agent") as mock_subagent_create_agent,
        ):
            create_deep_agent(
                model=model,
                tools=[],
                subagents=[
                    {
                        "name": "custom",
                        "description": "Custom subagent",
                        "system_prompt": "You are custom.",
                        "model": model,
                        "tools": [],
                        "tool_input_guardrails": [override_input],
                    }
                ],
                tool_input_guardrails=[default_input],
                tool_output_guardrails=[default_output],
            )

        assert len(mock_subagent_create_agent.call_args_list) == 2
        general_purpose_call = mock_subagent_create_agent.call_args_list[0]
        custom_call = mock_subagent_create_agent.call_args_list[1]

        gp_guardrail_middleware = next(
            (m for m in general_purpose_call.kwargs["middleware"] if isinstance(m, ToolGuardrailsMiddleware)),
            None,
        )
        custom_guardrail_middleware = next((m for m in custom_call.kwargs["middleware"] if isinstance(m, ToolGuardrailsMiddleware)), None)

        assert gp_guardrail_middleware is not None
        assert gp_guardrail_middleware._tool_input_guardrails == [default_input]
        assert gp_guardrail_middleware._tool_output_guardrails == [default_output]
        assert custom_guardrail_middleware is not None
        assert custom_guardrail_middleware._tool_input_guardrails == [override_input]
        assert custom_guardrail_middleware._tool_output_guardrails == [default_output]
