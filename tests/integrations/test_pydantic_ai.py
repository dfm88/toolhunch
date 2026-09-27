import importlib
import sys

import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolSearchReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from toolhunch import BM25Retriever, ToolSearchPipeline
from toolhunch.integrations.pydantic_ai import catalog_from_tool_defs, reveal_strategy

pytestmark = pytest.mark.anyio


def deferred(name: str, description: str, *properties: str) -> ToolDefinition:
    schema = {"type": "object", "properties": {p: {"type": "string"} for p in properties}}
    return ToolDefinition(name=name, description=description, parameters_json_schema=schema, defer_loading=True)


TOOLS = [
    deferred("get_weather", "Get the weather forecast for a city.", "city"),
    deferred("send_email", "Send an email message.", "to", "body"),
    deferred("get_diary_day", "Read one day of the diary.", "date"),
]


def test_catalog_from_tool_defs() -> None:
    card = catalog_from_tool_defs(TOOLS[:1], source="local")["get_weather"]
    assert (card.id, card.name, card.description, card.parameters, card.source) == snapshot(
        (
            "get_weather",
            "get_weather",
            "Get the weather forecast for a city.",
            {"type": "object", "properties": {"city": {"type": "string"}}},
            "local",
        )
    )


async def test_tool_search_reveals_what_the_pipeline_ranks() -> None:
    seen_tools: list[list[str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(sorted(tool.name for tool in info.function_tools))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"queries": ["weather forecast"]})])
        return ModelResponse(parts=[TextPart("done")])

    pipeline = ToolSearchPipeline(BM25Retriever(), k=2)
    agent = Agent(
        FunctionModel(model),
        toolsets=[ExternalToolset(TOOLS)],
        capabilities=[ToolSearch(strategy=reveal_strategy(pipeline))],
    )

    result = await agent.run("what's the weather in Milan?")

    returns = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart)
    ]
    assert [r.content["discovered_tools"] for r in returns] == [[{"name": "get_weather"}]]
    assert seen_tools == [["search_tools"], ["get_weather", "search_tools"]]  # revealed on the next request


def test_missing_extra_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in [m for m in sys.modules if m.startswith("toolhunch.integrations.pydantic_ai")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    with pytest.raises(ImportError, match=r"toolhunch\[pydantic-ai\]"):
        importlib.import_module("toolhunch.integrations.pydantic_ai")
