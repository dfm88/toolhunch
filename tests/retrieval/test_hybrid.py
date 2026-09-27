from collections.abc import Sequence

import pytest

from toolhunch import Retrieval, RetrievalUsage, ScoredCard, ToolCard, ToolCatalog
from toolhunch.retrieval import HybridRetriever, reciprocal_rank_fusion

pytestmark = pytest.mark.anyio


class FixedRetriever:
    def __init__(self, ids: str, usage: RetrievalUsage) -> None:
        self.result = Retrieval(tuple(ScoredCard(ToolCard(name=i), 1.0) for i in ids), usage)
        self.asked_k: list[int] = []

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        self.asked_k.append(k)
        return self.result


async def test_fuses_rankings_and_sums_usage() -> None:
    lexical = FixedRetriever("abc", RetrievalUsage())
    dense = FixedRetriever("cab", RetrievalUsage(index_tokens=7, query_tokens=2))

    result = await HybridRetriever([lexical, dense], weights=[1.0, 2.0]).retrieve(["q"], ToolCatalog([]), k=2)

    expected = reciprocal_rank_fusion([lexical.result.matches, dense.result.matches], k=2, weights=[1.0, 2.0])
    assert [s.card.id for s in result.matches] == [s.card.id for s in expected] == ["c", "a"]
    assert result.usage == RetrievalUsage(index_tokens=7, query_tokens=2)
    assert lexical.asked_k == dense.asked_k == [100]  # each side ranks deep before fusion


def test_configuration_errors() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetriever([])
    with pytest.raises(ValueError, match="weights"):
        HybridRetriever([FixedRetriever("a", RetrievalUsage())], weights=[1.0, 1.0])
