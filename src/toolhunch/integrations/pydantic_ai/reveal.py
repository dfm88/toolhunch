"""Reveal: toolhunch as the ranking behind Pydantic AI's `ToolSearch` capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from toolhunch.integrations.pydantic_ai.catalog import catalog_from_tool_defs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import ToolSearchFunc
    from pydantic_ai.tools import ToolDefinition

    from toolhunch.pipeline import ToolSearchPipeline

__all__ = ["reveal_strategy"]


def reveal_strategy[AgentDepsT](pipeline: ToolSearchPipeline) -> ToolSearchFunc[AgentDepsT]:
    """A search function for `ToolSearch(strategy=...)` that ranks deferred tools with `pipeline`.

    Pydantic AI calls it with the model's `search_tools` queries and every searchable deferred tool,
    then reveals the returned names. It runs locally, and through the client-executed native tool
    search on Anthropic and OpenAI Responses, which keeps the prompt cache intact.

    `ToolSearch(max_results=...)` (default 10) truncates the list after this function returns: set
    the pipeline's `k` to the same value.
    """

    async def search(ctx: RunContext[AgentDepsT], queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        result = await pipeline.search(queries, catalog_from_tool_defs(tools))
        return result.ids

    return search
