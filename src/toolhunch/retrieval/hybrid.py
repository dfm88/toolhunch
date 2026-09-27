"""Hybrid retrieval: several retrievers fused by reciprocal rank."""

from __future__ import annotations

from typing import TYPE_CHECKING

from toolhunch.retrieval.base import FUSION_DEPTH, Retrieval, RetrievalUsage, check_k, reciprocal_rank_fusion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch.cards import ToolCatalog
    from toolhunch.retrieval.base import Retriever

__all__ = ["HybridRetriever"]


class HybridRetriever:
    """Runs several retrievers and fuses their rankings with reciprocal rank fusion.

    Typically lexical + dense: BM25 catches exact identifiers, embeddings catch paraphrases. RRF
    uses only positions, so the retrievers' incomparable scores never mix. Retrievers run one after
    the other; usage is summed.
    """

    def __init__(
        self,
        retrievers: Sequence[Retriever],
        *,
        weights: Sequence[float] | None = None,
        rrf_k: int = 60,
        depth: int = FUSION_DEPTH,
    ) -> None:
        """Fuse `retrievers`, each asked for `max(k, depth)` matches, with optional `weights`."""
        if not retrievers:
            raise ValueError("HybridRetriever needs at least one retriever")
        if weights is not None and len(weights) != len(retrievers):
            raise ValueError(f"{len(weights)} weights for {len(retrievers)} retrievers")
        self._retrievers = tuple(retrievers)
        self._weights = None if weights is None else tuple(weights)
        self._rrf_k = rrf_k
        self._depth = depth

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Query every retriever, then fuse; see [`Retriever.retrieve`][toolhunch.Retriever.retrieve]."""
        check_k(k)
        results = [await retriever.retrieve(queries, catalog, k=max(k, self._depth)) for retriever in self._retrievers]
        matches = reciprocal_rank_fusion(
            [result.matches for result in results], k=k, rrf_k=self._rrf_k, weights=self._weights
        )
        usage = sum((result.usage for result in results), RetrievalUsage())
        return Retrieval(matches=tuple(matches), usage=usage)
