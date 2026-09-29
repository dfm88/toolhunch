import importlib
import sys
from collections.abc import Sequence
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolSearchReturnPart,
    UserContent,
)
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


BOOKING_TOOLS = [
    deferred("book_table", "Book a table at a restaurant.", "restaurant", "guests"),
    deferred("book_flight", "Book a flight between two airports.", "origin", "destination"),
]
IMAGE = ImageUrl(url="https://example.com/menu.png")


async def search_once(prompt: str | Sequence[UserContent], decider: Any) -> list[list[str]]:
    """Run an agent on `prompt` that searches for "book a table" once; return the tools each request offered."""
    offered: list[list[str]] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.append(sorted(tool.name for tool in info.function_tools))
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("search_tools", {"queries": ["book a table"]})])
        return ModelResponse(parts=[TextPart("done")])

    pipeline = ToolSearchPipeline(BM25Retriever(), decider=decider, k=2, top_n=1)
    agent = Agent(
        FunctionModel(model),
        toolsets=[ExternalToolset(BOOKING_TOOLS)],
        capabilities=[ToolSearch(strategy=reveal_strategy(pipeline))],
    )
    await agent.run(prompt)
    return offered


async def test_reveal_gives_the_decider_the_user_prompt(recording_decider: Any) -> None:
    decider = recording_decider(order=["book_flight", "book_table"])

    offered = await search_once("Book a table for two", decider)

    assert decider.states == ["Request: Book a table for two\nSearch queries: book a table"]
    assert offered == [["search_tools"], ["book_flight", "search_tools"]]  # the decider's pick, not the retriever's


@pytest.mark.parametrize(
    ("prompt", "state"),
    [
        pytest.param(["Book this", IMAGE], "Request: Book this\nSearch queries: book a table", id="text-and-image"),
        pytest.param(
            ["Book this", IMAGE, TextContent("for two", metadata={"lang": "en"})],
            "Request: Book this\nfor two\nSearch queries: book a table",
            id="text-parts-are-joined-by-newlines",
        ),
        pytest.param([IMAGE], "Search queries: book a table", id="no-text-at-all"),
    ],
)
async def test_reveal_uses_the_text_parts_of_a_multimodal_prompt(
    recording_decider: Any, prompt: list[UserContent], state: str
) -> None:
    decider = recording_decider()

    await search_once(prompt, decider)

    assert decider.states == [state]


async def test_reveal_reveals_nothing_when_the_decider_abstains(recording_decider: Any) -> None:
    decider = recording_decider(abstain=True)

    offered = await search_once("Book a table for two", decider)

    assert decider.candidates == [["book_table", "book_flight"]]  # it was asked, and had candidates to rank
    assert offered == [["search_tools"], ["search_tools"]]  # the next request lists only search_tools


def test_missing_extra_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in [m for m in sys.modules if m.startswith("toolhunch.integrations.pydantic_ai")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    with pytest.raises(ImportError, match=r"toolhunch\[pydantic-ai\]"):
        importlib.import_module("toolhunch.integrations.pydantic_ai")
