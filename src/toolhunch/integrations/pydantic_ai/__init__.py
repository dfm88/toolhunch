"""Pydantic AI integration: tool catalogs from `ToolDefinition`s and a `ToolSearch` strategy.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

from toolhunch import BM25Retriever, ToolSearchPipeline
from toolhunch.integrations.pydantic_ai import reveal_strategy

pipeline = ToolSearchPipeline(BM25Retriever(), k=10)
agent = Agent(..., capabilities=[ToolSearch(strategy=reveal_strategy(pipeline))])
```
"""

try:
    import pydantic_ai  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as error:
    raise ImportError(
        "toolhunch.integrations.pydantic_ai needs Pydantic AI: pip install 'toolhunch[pydantic-ai]'"
    ) from error

from toolhunch.integrations.pydantic_ai.catalog import catalog_from_tool_defs
from toolhunch.integrations.pydantic_ai.reveal import reveal_strategy

__all__ = ["catalog_from_tool_defs", "reveal_strategy"]
