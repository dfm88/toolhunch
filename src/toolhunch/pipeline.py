"""The tool search pipeline: a retriever for recall, then an optional decider for precision and abstention."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from toolhunch.decision.planner import build_state
from toolhunch.retrieval.base import check_k, clean_queries

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch.cards import ToolCatalog
    from toolhunch.decision.base import DecisionUsage
    from toolhunch.decision.decider import Decider, Decision
    from toolhunch.retrieval.base import RetrievalUsage, Retriever, ScoredCard

__all__ = ["SearchResult", "StageTrace", "ToolSearchPipeline"]


@dataclass(frozen=True, slots=True)
class StageTrace:
    """What one stage did: how long it took, how many candidates it kept, what it cost.

    Attributes:
        stage: `"retrieve"` or `"decide"`.
        seconds: Wall time of the stage, measured around it.
        candidates: How many candidates the stage returned, before `top_n` cuts the final list.
        usage: What the stage consumed: a `RetrievalUsage` for `"retrieve"`, a `DecisionUsage` for `"decide"`.
    """

    stage: str
    seconds: float
    candidates: int
    usage: RetrievalUsage | DecisionUsage


@dataclass(frozen=True, slots=True)
class SearchResult:
    """The ranked matches of one search and a trace of every stage that produced them.

    Attributes:
        matches: The best `top_n` candidates, best first; all of them without a `top_n`. With a decider they follow
            its ranking and carry its scores, which are not comparable with retrieval scores, and they are kept when
            it abstained.
        trace: One entry per stage that ran, in order: `"retrieve"`, then `"decide"` when there is a decider.
        decision: What the decider concluded, with the exchanges it made and its threshold key. `None` without a
            decider.
    """

    matches: tuple[ScoredCard, ...]
    trace: tuple[StageTrace, ...]
    decision: Decision | None = None

    @property
    def ids(self) -> list[str]:
        """Card ids, best first."""
        return [match.card.id for match in self.matches]

    @property
    def names(self) -> list[str]:
        """Tool names, best first (may repeat when ids differ)."""
        return [match.card.name for match in self.matches]

    @property
    def abstained(self) -> bool:
        """Whether the decider said that no candidate fits; always `False` without a decider."""
        return self.decision is not None and self.decision.abstained


class ToolSearchPipeline:
    """Ranks a catalog's cards for a list of queries.

    Stage 1 is a [`Retriever`][toolhunch.Retriever] (recall): it finds `k` candidates. Stage 2 is an optional
    [`Decider`][toolhunch.Decider] (precision and abstention): it re-ranks the candidates for the request and can say
    that none fits, and adds its own entry to `SearchResult.trace`. A failure of the decider is raised as it is,
    never replaced by the retriever's order. Without a decider, a search returns the retriever's candidates.
    The pipeline keeps no per-call state; retrievers cache their indexes by catalog fingerprint.
    """

    def __init__(
        self, retriever: Retriever, *, decider: Decider | None = None, k: int = 10, top_n: int | None = None
    ) -> None:
        """Return at most `top_n` matches per search, chosen among `k` candidates.

        Args:
            retriever: Stage 1.
            decider: Stage 2; without one the retriever's order stands.
            k: How many candidates the retriever finds, and so how many the decider ranks.
            top_n: How many matches a search returns; all `k` when `None`.

        Raises:
            ValueError: `k` or `top_n` is below 1.
        """
        check_k(k)
        if top_n is not None:
            check_k(top_n, name="top_n")
        self._retriever = retriever
        self._decider = decider
        self._k = k
        self._top_n = top_n

    async def search(self, queries: Sequence[str], catalog: ToolCatalog, *, context: str | None = None) -> SearchResult:
        """Search `catalog` for `queries`.

        Args:
            queries: What to search for.
            catalog: The tools to rank.
            context: The request the queries come from. The decider reads it beside the queries, and cuts it in the
                middle when it is longer than its model takes. Ignored without a decider.

        Raises:
            DecisionError: The decider's model failed or answered something unusable.
            ValueError: The decider cannot ask about these candidates within its model's limits.
        """
        started = time.perf_counter()
        retrieval = await self._retriever.retrieve(queries, catalog, k=self._k)
        retrieve = StageTrace(
            stage="retrieve",
            seconds=time.perf_counter() - started,
            candidates=len(retrieval.matches),
            usage=retrieval.usage,
        )
        if self._decider is None:
            return SearchResult(matches=retrieval.matches[: self._top_n], trace=(retrieve,))

        state = build_state(context, clean_queries(queries))
        started = time.perf_counter()
        decision = await self._decider.decide(state, retrieval.matches)
        decide = StageTrace(
            stage="decide",
            seconds=time.perf_counter() - started,
            candidates=len(decision.ranked),
            usage=decision.usage,
        )
        return SearchResult(matches=decision.ranked[: self._top_n], trace=(retrieve, decide), decision=decision)
