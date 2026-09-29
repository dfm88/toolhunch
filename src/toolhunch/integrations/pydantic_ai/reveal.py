"""Reveal: toolhunch as the ranking behind Pydantic AI's `ToolSearch` capability."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai.messages import TextContent

from toolhunch.integrations.pydantic_ai.catalog import catalog_from_tool_defs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai import RunContext
    from pydantic_ai.capabilities import ToolSearchFunc
    from pydantic_ai.messages import UserContent
    from pydantic_ai.tools import ToolDefinition

    from toolhunch.pipeline import ToolSearchPipeline

__all__ = ["reveal_strategy"]


def reveal_strategy[AgentDepsT](pipeline: ToolSearchPipeline) -> ToolSearchFunc[AgentDepsT]:
    """A search function for `ToolSearch(strategy=...)` that ranks deferred tools with `pipeline`.

    Pydantic AI calls it with the model's `search_tools` queries and every searchable deferred tool,
    then reveals the returned names. It runs locally, and through the client-executed native tool
    search on Anthropic and OpenAI Responses, which keeps the prompt cache intact.

    With a decider, the pipeline also reads the run's user prompt as the request: a string as it is, a
    multimodal prompt as its text parts joined by newlines. When the decider abstains, nothing is revealed.

    `ToolSearch(max_results=...)` (default 10) truncates the list after this function returns. Without a
    decider, set the pipeline's `k` to the same value; with one, `k` is how many candidates it ranks, so
    set `top_n` to the same value instead.
    """

    async def search(ctx: RunContext[AgentDepsT], queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
        result = await pipeline.search(queries, catalog_from_tool_defs(tools), context=_prompt_text(ctx.prompt))
        return [] if result.abstained else result.ids

    return search


def _prompt_text(prompt: str | Sequence[UserContent] | None) -> str | None:
    """A run's user prompt as text: a string as it is, a multimodal prompt's text parts joined by newlines."""
    if prompt is None or isinstance(prompt, str):
        return prompt
    texts = [part if isinstance(part, str) else part.content for part in prompt if isinstance(part, str | TextContent)]
    return "\n".join(texts) if texts else None
