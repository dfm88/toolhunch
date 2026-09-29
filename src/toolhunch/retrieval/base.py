"""The retriever protocol and the pieces every retriever shares."""

from __future__ import annotations

import heapq
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from toolhunch.cards import ToolCard, ToolCatalog

__all__ = [
    "FUSION_DEPTH",
    "IndexCache",
    "Retrieval",
    "RetrievalUsage",
    "Retriever",
    "ScoredCard",
    "check_k",
    "clean_queries",
    "fuse_query_rankings",
    "reciprocal_rank_fusion",
    "top_k",
]

FUSION_DEPTH = 100
"""How deep each ranking goes before fusion: `max(k, FUSION_DEPTH)`."""


@dataclass(frozen=True, slots=True)
class ScoredCard:
    """A card and its score. Scores are comparable only within one retrieval."""

    card: ToolCard
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalUsage:
    """Paid resources one retrieval consumed.

    Attributes:
        index_tokens: Embedding tokens spent indexing cards during this call (first call per catalog).
        query_tokens: Embedding tokens spent on the queries.
    """

    index_tokens: int = 0
    query_tokens: int = 0

    def __add__(self, other: RetrievalUsage) -> RetrievalUsage:
        return RetrievalUsage(
            index_tokens=self.index_tokens + other.index_tokens,
            query_tokens=self.query_tokens + other.query_tokens,
        )


@dataclass(frozen=True, slots=True)
class Retrieval:
    """What a retriever returns: matches, best first, and the usage they cost."""

    matches: tuple[ScoredCard, ...] = ()
    usage: RetrievalUsage = RetrievalUsage()


class Retriever(Protocol):
    """Stage 1: find the cards most likely to answer the queries."""

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Return at most `k` matches for `queries` from `catalog`, best first.

        Several queries (Pydantic AI's `search_tools` sends a list) are ranked separately and fused.
        Queries without a letter or digit are ignored; with none left the result is empty. A single
        `str` is rejected with `TypeError` by the built-in retrievers.
        """
        ...


def clean_queries(queries: Sequence[str]) -> list[str]:
    """Strip queries, drop duplicates and queries without a letter or digit, keep the original order.

    A query of punctuation alone ("???") would still reach an embedder and return arbitrary cards.

    Raises:
        TypeError: `queries` is a single `str`, which would be searched character by character.
    """
    if isinstance(queries, str):
        raise TypeError("queries must be a list of strings, not a str: wrap a single query in a list")
    return list(dict.fromkeys(query.strip() for query in queries if any(char.isalnum() for char in query)))


def check_k(k: int, *, name: str = "k") -> None:
    """Reject a non-positive result size; `name` is what the error calls it."""
    if k < 1:
        raise ValueError(f"{name} must be at least 1, got {k}")


def top_k(scored: Iterable[ScoredCard], k: int) -> list[ScoredCard]:
    """The `k` best matches, ordered by score descending, then by card id."""
    return heapq.nsmallest(k, scored, key=lambda item: (-item.score, item.card.id))


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredCard]],
    *,
    k: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[ScoredCard]:
    """Fuse rankings by reciprocal rank: a card scores `sum(weight / (rrf_k + rank))`, ranks from 1.

    Only positions matter, not the original scores, so rankings on different scales fuse safely.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError(f"{len(weights)} weights for {len(rankings)} rankings")
    fused: dict[str, float] = {}
    cards: dict[str, ToolCard] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, scored in enumerate(ranking, start=1):
            fused[scored.card.id] = fused.get(scored.card.id, 0.0) + weight / (rrf_k + rank)
            cards[scored.card.id] = scored.card
    return top_k((ScoredCard(cards[card_id], score) for card_id, score in fused.items()), k)


def fuse_query_rankings(rankings: Sequence[Sequence[ScoredCard]], *, k: int) -> list[ScoredCard]:
    """One ranking per query → the final list: as is for one query, RRF for several."""
    if len(rankings) == 1:
        return list(rankings[0][:k])
    return reciprocal_rank_fusion(rankings, k=k)


class IndexCache[T]:
    """A small LRU map from catalog fingerprint to a built index."""

    def __init__(self, *, max_entries: int = 8) -> None:
        """Keep at most `max_entries` indexes."""
        self._max_entries = max_entries
        self._entries: OrderedDict[str, T] = OrderedDict()

    def get(self, fingerprint: str) -> T | None:
        """Return the index for `fingerprint`, if cached, and mark it recently used."""
        entry = self._entries.get(fingerprint)
        if entry is not None:
            self._entries.move_to_end(fingerprint)
        return entry

    def put(self, fingerprint: str, index: T) -> None:
        """Cache `index`, evicting the least recently used entry beyond the limit."""
        self._entries[fingerprint] = index
        self._entries.move_to_end(fingerprint)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
