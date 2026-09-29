"""Model-written search queries: what an agent would type into its tool search, frozen once per task file.

For each task a Pydantic AI agent with the `ToolSearch` capability reads the request. Only its first response
is used: the queries of its `search_tools` calls. The run stops there, before any tool runs, so no query is
ever answered and no tool is ever revealed. The queries are frozen next to the task file, in
`<task file stem>.model-queries.json`, together with what produced them.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, cast

import tiktoken
from genai_prices import Usage, calc_price
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RunUsage

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date
    from pathlib import Path

    from pydantic_ai import RunContext
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.tools import ToolDefinition

    from toolhunch_bench.datasets.toolret import ToolRetTask

__all__ = [
    "WRITER_INSTRUCTIONS",
    "WRITER_MODEL",
    "WRITER_SETTINGS",
    "WriterEstimate",
    "WrittenQueries",
    "estimate_writer_cost",
    "load_model_queries",
    "model_queries_path",
    "save_model_queries",
    "search_tools_definition",
    "write_queries",
    "writer_block",
]

WRITER_MODEL = "gpt-5.4-mini-2026-03-17"
# On Chat Completions this model takes function tools only without reasoning: any other effort is an HTTP 400.
WRITER_SETTINGS: dict[str, Any] = {"openai_reasoning_effort": "none"}
WRITER_INSTRUCTIONS = "You are a helpful assistant. Use your tools to complete the user's request."

_OUTPUT_TOKENS_PER_TASK = 300  # an upper bound, not a measurement: the writer does not reason, so a response is short
_FRAMING_TOKENS = 50  # what the API adds around the messages and the tool


@dataclass(frozen=True, slots=True)
class WrittenQueries:
    """What the writer searched for on one task.

    Attributes:
        task_id: The ToolRet task id.
        queries: The queries of every `search_tools` call in the first response, in order, each once. When the
            response made no search, the request alone.
        fallback: Whether the response made no search, so that `queries` is the request.
    """

    task_id: str
    queries: tuple[str, ...]
    fallback: bool


@dataclass(frozen=True, slots=True)
class WriterEstimate:
    """What writing the queries of `tasks` tasks is expected to bill; the output tokens are an upper bound."""

    tasks: int
    input_tokens: int
    output_tokens: int
    usd: float


class _Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queries: Annotated[tuple[str, ...], Field(min_length=1)]
    fallback: bool


class _QueriesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    writer: dict[str, Any]
    queries: dict[str, _Entry]


def placeholder() -> None:
    """A tool that does nothing."""


def unused(ctx: RunContext[object], queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
    """The search strategy: never reached, because the run stops at the first response."""
    return []


def _writer_agent(model: Model) -> Agent[object, str]:
    # On Chat Completions `ToolSearch` offers `search_tools` only while some tool is deferred, and keeps that tool off
    # the request, so a placeholder is all the agent needs. Neither it nor the strategy runs: the run stops at the
    # first response.
    return Agent(
        model,
        instructions=WRITER_INSTRUCTIONS,
        tools=[Tool(placeholder, defer_loading=True)],
        capabilities=[ToolSearch(strategy=unused)],
    )


def search_tools_definition() -> ToolDefinition:
    """The `search_tools` tool as the writer is offered it: Pydantic AI's own description and parameters.

    It is read from the tools of a request to a `FunctionModel`, so no model is paid and nothing private is
    imported. It runs its own event loop: call it from synchronous code.
    """
    offered: list[ToolDefinition] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        offered.extend(info.function_tools)
        return ModelResponse(parts=[TextPart("ok")])

    asyncio.run(_writer_agent(FunctionModel(respond)).run("probe"))
    [definition] = [tool for tool in offered if tool.name == ToolSearch.function_tool_name]
    return definition


def _definition_sha256(definition: ToolDefinition) -> str:
    canonical = json.dumps(
        {"description": definition.description, "parameters": definition.parameters_json_schema},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def writer_block(*, model: str, definition: ToolDefinition, run_date: date) -> dict[str, Any]:
    """The `writer` block of a queries file: everything that shaped the queries, so that a change shows.

    `search_tools_sha256` is the SHA-256 of the canonical JSON (sorted keys, compact) of the description and
    the parameter schema of `definition`, which is what the model reads of `search_tools`.
    """
    return {
        "model": model,
        "settings": dict(WRITER_SETTINGS),
        "pydantic_ai": importlib.metadata.version("pydantic-ai-slim"),
        "instructions": WRITER_INSTRUCTIONS,
        "search_tools_sha256": _definition_sha256(definition),
        "date": run_date.isoformat(),
    }


def _searched(response: ModelResponse) -> tuple[str, ...]:
    """The queries of the `search_tools` calls in `response`, in order, each once; blank ones are not queries."""
    found: dict[str, None] = {}
    for call in response.tool_calls:
        if call.tool_name != ToolSearch.function_tool_name:
            continue
        queries = call.args_as_dict().get("queries")
        if isinstance(queries, list):
            found.update(
                dict.fromkeys(
                    query for query in cast("list[object]", queries) if isinstance(query, str) and query.strip()
                )
            )
    return tuple(found)


async def write_queries(
    tasks: Sequence[ToolRetTask],
    *,
    model: Model,
    settings: ModelSettings | None = None,
    usage: RunUsage | None = None,
    answers: dict[str, str] | None = None,
) -> tuple[list[WrittenQueries], RunUsage]:
    """Ask the writer agent what it would search for on each task, and stop at its first response.

    Tasks run one after the other, so the results follow `tasks`. A first response with no search falls back
    to the request, flagged.

    Args:
        tasks: The tasks whose requests the agent reads.
        model: The writer model.
        settings: Settings for every request, on top of the model's own.
        usage: Where the usage of each request is added as its task ends, so that a caller keeps what a run
            that fails midway had billed. By default a new one.
        answers: Where the text of a first response with no search is kept, by task id, so that a caller can
            see what the model said instead; empty when the response had no text. By default nothing is kept.

    Returns:
        One entry per task, and the usage of the whole run.
    """
    agent = _writer_agent(model)
    total = usage if usage is not None else RunUsage()
    written: list[WrittenQueries] = []
    for task in tasks:
        queries: tuple[str, ...] = ()
        async with agent.iter(task.query, model_settings=settings) as run:
            async for node in run:
                if Agent.is_call_tools_node(node):
                    queries = _searched(node.model_response)
                    if not queries and answers is not None:
                        answers[task.id] = node.model_response.text or ""
                    total.incr(run.usage)
                    break  # leaving here runs no tool and asks the model nothing more
        written.append(WrittenQueries(task_id=task.id, queries=queries or (task.query,), fallback=not queries))
    return written, total


def estimate_writer_cost(tasks: Sequence[ToolRetTask], *, model: str, definition: ToolDefinition) -> WriterEstimate:
    """Estimate what `write_queries` bills on `tasks`, priced by genai-prices for OpenAI.

    A request is counted as its instructions, the `search_tools` definition and the task's request, in tiktoken
    `o200k_base` tokens, plus a fixed framing. A response is counted as a fixed number of tokens, an upper bound
    that a writer without reasoning should stay well under, so the estimate leans high.
    """
    encoding = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        return len(encoding.encode(text, disallowed_special=()))

    tool = json.dumps(
        {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.parameters_json_schema,
        }
    )
    per_request = count(WRITER_INSTRUCTIONS) + count(tool) + _FRAMING_TOKENS
    input_tokens = sum(per_request + count(task.query) for task in tasks)
    output_tokens = _OUTPUT_TOKENS_PER_TASK * len(tasks)
    price = calc_price(
        Usage(input_tokens=input_tokens, output_tokens=output_tokens), model_ref=model, provider_id="openai"
    )
    return WriterEstimate(
        tasks=len(tasks), input_tokens=input_tokens, output_tokens=output_tokens, usd=float(price.total_price)
    )


def model_queries_path(task_file: Path) -> Path:
    """Where the queries of `task_file` are frozen: `<directory>/<stem>.model-queries.json` beside it."""
    return task_file.with_name(f"{task_file.stem}.model-queries.json")


def save_model_queries(path: Path, written: Sequence[WrittenQueries], *, writer: Mapping[str, Any]) -> None:
    """Write `written`, in order, with the `writer` block (see `writer_block`) to `path`."""
    contents = _QueriesFile(
        writer=dict(writer),
        queries={entry.task_id: _Entry(queries=entry.queries, fallback=entry.fallback) for entry in written},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_model_queries(path: Path, tasks: Sequence[ToolRetTask]) -> dict[str, WrittenQueries]:
    """The frozen queries of `tasks`, by task id in the order of `tasks`, from the file at `path`.

    Raises:
        ValueError: The file is not a queries file, or has no entry for one of `tasks`.
    """
    stored = _QueriesFile.model_validate_json(path.read_text(encoding="utf-8")).queries
    if missing := [task.id for task in tasks if task.id not in stored]:
        raise ValueError(f"{path} has no queries for {len(missing)} task ids, first {missing[0]!r}")
    return {
        task.id: WrittenQueries(task_id=task.id, queries=stored[task.id].queries, fallback=stored[task.id].fallback)
        for task in tasks
    }
