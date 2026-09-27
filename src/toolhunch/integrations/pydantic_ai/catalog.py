"""Pydantic AI `ToolDefinition`s → toolhunch cards."""

from __future__ import annotations

from typing import TYPE_CHECKING

from toolhunch.cards import ToolCard, ToolCatalog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pydantic_ai.tools import ToolDefinition

__all__ = ["catalog_from_tool_defs"]


def catalog_from_tool_defs(tool_defs: Sequence[ToolDefinition], *, source: str | None = None) -> ToolCatalog:
    """Build a catalog whose card ids are the tool names (unique within an agent)."""
    return ToolCatalog(
        ToolCard(
            name=tool_def.name,
            description=tool_def.description or "",
            parameters=tool_def.parameters_json_schema,
            source=source,
        )
        for tool_def in tool_defs
    )
