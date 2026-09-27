from collections.abc import Sequence

import pytest

from toolhunch import Retrieval, RetrievalUsage, ScoredCard, ToolCard, ToolCatalog, ToolSearchPipeline

pytestmark = pytest.mark.anyio


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        self.calls.append((list(queries), k))
        cards = [ToolCard(name="search", id="web.search"), ToolCard(name="search", id="docs.search")]
        return Retrieval(tuple(ScoredCard(card, 1.0) for card in cards), RetrievalUsage(query_tokens=3))


async def test_search_runs_the_retriever_and_traces_the_stage() -> None:
    retriever = StubRetriever()

    result = await ToolSearchPipeline(retriever, k=2).search(["find docs"], ToolCatalog([]))

    assert retriever.calls == [(["find docs"], 2)]
    assert result.ids == ["web.search", "docs.search"]
    assert result.names == ["search", "search"]
    [stage] = result.trace
    assert (stage.stage, stage.candidates, stage.usage) == ("retrieve", 2, RetrievalUsage(query_tokens=3))
    assert stage.seconds >= 0


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ToolSearchPipeline(StubRetriever(), k=0)
