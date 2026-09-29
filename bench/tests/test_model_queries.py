import hashlib
import importlib.metadata
import json
from datetime import date
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from toolhunch_bench.datasets.model_queries import (
    WRITER_INSTRUCTIONS,
    WrittenQueries,
    load_model_queries,
    model_queries_path,
    save_model_queries,
    search_tools_definition,
    write_queries,
    writer_block,
)
from toolhunch_bench.datasets.toolret import ToolRetTask

BOOKING = ToolRetTask(
    "yelp_query_0", "yelp", "Book a table for two tonight", "Given a `yelp` task", frozenset({"yelp_tool_0"})
)
RAIN = ToolRetTask(
    "weather_query_0", "weather", "Will it rain in Rome?", "Given a `weather` task", frozenset({"weather_tool_0"})
)


def calling_search(*calls: list[str]) -> FunctionModel:
    """A model whose response makes one parallel `search_tools` call per list of queries."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart("search_tools", {"queries": queries}) for queries in calls],
            usage=RequestUsage(input_tokens=100, output_tokens=40),
        )

    return FunctionModel(respond)


@pytest.mark.anyio
async def test_the_first_response_search_calls_become_the_queries() -> None:
    model = calling_search(["restaurant booking", "reserve table"], ["reserve table", "table reservation"])

    written, usage = await write_queries([BOOKING], model=model)

    assert written == [
        WrittenQueries(
            task_id="yelp_query_0",
            queries=("restaurant booking", "reserve table", "table reservation"),
            fallback=False,
        )
    ]
    # The run stops at the first response: one request, billed once, and no tool call is answered.
    assert (usage.requests, usage.input_tokens, usage.output_tokens, usage.tool_calls) == (1, 100, 40, 0)


@pytest.mark.anyio
async def test_a_first_response_without_search_falls_back_to_the_request() -> None:
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("Sure, I can book that.")]))
    answers: dict[str, str] = {}

    written, _ = await write_queries([BOOKING], model=model, answers=answers)

    assert written == [WrittenQueries(task_id="yelp_query_0", queries=("Book a table for two tonight",), fallback=True)]
    assert answers == {"yelp_query_0": "Sure, I can book that."}  # what the model said instead of searching


@pytest.mark.anyio
async def test_blank_or_malformed_search_arguments_count_as_no_search() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart("search_tools", {"queries": ["", "   "]}),
                ToolCallPart("search_tools", {"queries": "a bare string"}),
                ToolCallPart("search_tools", {"queries": [1, None]}),
                ToolCallPart("search_tools", '{"queries": ["cut off'),
            ]
        )

    written, _ = await write_queries([BOOKING], model=FunctionModel(respond))

    assert written == [WrittenQueries(task_id="yelp_query_0", queries=("Book a table for two tonight",), fallback=True)]


@pytest.mark.anyio
async def test_the_model_is_offered_search_tools_alone_with_the_fixed_instructions() -> None:
    offered: list[tuple[list[str], str | None, str]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = messages
        assert isinstance(request, ModelRequest)
        [prompt] = request.parts
        assert isinstance(prompt, UserPromptPart)
        offered.append(([tool.name for tool in info.function_tools], info.instructions, str(prompt.content)))
        return ModelResponse(parts=[TextPart("done")])

    await write_queries([RAIN], model=FunctionModel(respond))

    # The placeholder deferred tool stays off the request: only the tool search reaches the model.
    assert offered == [(["search_tools"], WRITER_INSTRUCTIONS, "Will it rain in Rome?")]


def test_the_writer_block_records_everything_that_shaped_the_queries() -> None:
    definition = search_tools_definition()

    block = writer_block(model="gpt-5.4-mini-2026-03-17", definition=definition, run_date=date(2026, 9, 29))

    canonical = json.dumps(
        {"description": definition.description, "parameters": definition.parameters_json_schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    assert definition.name == "search_tools"
    assert definition.parameters_json_schema["required"] == ["queries"]
    assert block == {
        "model": "gpt-5.4-mini-2026-03-17",
        "settings": {"openai_reasoning_effort": "none"},
        "pydantic_ai": importlib.metadata.version("pydantic-ai-slim"),
        "instructions": "You are a helpful assistant. Use your tools to complete the user's request.",
        "search_tools_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "date": "2026-09-29",
    }


def test_model_queries_round_trip_and_missing_ids_are_rejected(tmp_path: Path) -> None:
    written = [
        WrittenQueries(task_id="yelp_query_0", queries=("restaurant booking", "reserve table"), fallback=False),
        WrittenQueries(task_id="weather_query_0", queries=("Will it rain in Rome?",), fallback=True),
    ]
    path = model_queries_path(tmp_path / "toolret-pilot-50.json")
    other = ToolRetTask("email_query_0", "email", "mail my boss", "Given an `email` task", frozenset({"email_tool_0"}))

    save_model_queries(path, written, writer={"model": "writer-model", "date": "2026-09-29"})

    assert path.name == "toolret-pilot-50.model-queries.json"
    assert json.loads(path.read_text()) == snapshot(
        {
            "writer": {"model": "writer-model", "date": "2026-09-29"},
            "queries": {
                "yelp_query_0": {"queries": ["restaurant booking", "reserve table"], "fallback": False},
                "weather_query_0": {"queries": ["Will it rain in Rome?"], "fallback": True},
            },
        }
    )
    assert load_model_queries(path, [BOOKING, RAIN]) == {entry.task_id: entry for entry in written}
    with pytest.raises(ValueError, match="email_query_0"):
        load_model_queries(path, [BOOKING, other])
