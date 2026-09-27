"""The tool search pipeline: stage 1 today, stage 2 (a decider) in a later release."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from toolhunch.retrieval.base import check_k

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch.cards import ToolCatalog
    from toolhunch.retrieval.base import RetrievalUsage, Retriever, ScoredCard

__all__ = ["SearchResult", "StageTrace", "ToolSearchPipeline"]


@dataclass(frozen=True, slots=True)
class StageTrace:
    """What one stage did: how long it took, how many candidates it kept, what it cost."""

    stage: str
    seconds: float
    candidates: int
    usage: RetrievalUsage


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The ranked matches of one search and a trace of every stage that produced them."""

    matches: tuple[ScoredCard, ...]
    trace: tuple[StageTrace, ...]

    @property
    def ids(self) -> list[str]:
        """Card ids, best first."""
        return [match.card.id for match in self.matches]

    @property
    def names(self) -> list[str]:
        """Tool names, best first (may repeat when ids differ)."""
        return [match.card.name for match in self.matches]


class ToolSearchPipeline:
    """Ranks a catalog's cards for a list of queries.

    Stage 1 is a [`Retriever`][toolhunch.Retriever] (recall). A decision stage for precision and
    abstention will run after it and add its own entry to `SearchResult.trace`. The pipeline keeps
    no per-call state; retrievers cache their indexes by catalog fingerprint.
    """

    def __init__(self, retriever: Retriever, *, k: int = 10) -> None:
        """Return at most `k` matches per search."""
        check_k(k)
        self._retriever = retriever
        self._k = k

    async def search(self, queries: Sequence[str], catalog: ToolCatalog) -> SearchResult:
        """Search `catalog` for `queries`."""
        started = time.perf_counter()
        retrieval = await self._retriever.retrieve(queries, catalog, k=self._k)
        stage = StageTrace(
            stage="retrieve",
            seconds=time.perf_counter() - started,
            candidates=len(retrieval.matches),
            usage=retrieval.usage,
        )
        return SearchResult(matches=retrieval.matches, trace=(stage,))
