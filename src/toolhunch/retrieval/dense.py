"""Dense retrieval: cosine similarity between query and card embeddings."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from typing import TYPE_CHECKING

from toolhunch.cards import default_search_text
from toolhunch.retrieval.base import (
    FUSION_DEPTH,
    IndexCache,
    Retrieval,
    RetrievalUsage,
    ScoredCard,
    check_k,
    clean_queries,
    fuse_query_rankings,
    top_k,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch.cards import SearchText, ToolCard, ToolCatalog
    from toolhunch.retrieval.embedders import Embedder

__all__ = ["DenseRetriever"]


@dataclass(frozen=True, slots=True)
class _Index:
    cards: tuple[ToolCard, ...]
    vectors: tuple[array[float], ...]  # unit length, one per card


class DenseRetriever:
    """Ranks cards by cosine similarity between query and search-text embeddings.

    Scoring is exact brute force in pure Python (`math.sumprod` over `array('f')`): about 5-13 ms
    per query for 1,000 cards and 0.2-0.6 s for 44,000 (1536 dimensions, measured 2026-09-27).
    Card vectors are kept per `(model_id, text)`, so a changed catalog embeds only its new cards;
    that store grows with every distinct text the retriever sees.
    Texts over the embedder's input limit must be cut by the embedder
    ([`OpenAIEmbedder`][toolhunch.OpenAIEmbedder] does, see `max_input_bytes`): a rejected card fails
    the whole index build.
    """

    def __init__(
        self, embedder: Embedder, *, search_text: SearchText = default_search_text, cache_size: int = 8
    ) -> None:
        """Use `embedder` for cards and queries; index `search_text(card)`."""
        self._embedder = embedder
        self._search_text = search_text
        self._indexes: IndexCache[_Index] = IndexCache(max_entries=cache_size)
        self._vectors: dict[tuple[str, str], array[float]] = {}

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Rank `catalog` for each query and fuse; see [`Retriever.retrieve`][toolhunch.Retriever.retrieve]."""
        check_k(k)
        cleaned = clean_queries(queries)
        if not cleaned or not len(catalog):
            return Retrieval()
        index, index_tokens = await self._index(catalog)
        batch = await self._embedder.embed(cleaned, kind="query")
        depth = max(k, FUSION_DEPTH)
        rankings = [
            top_k(
                (
                    ScoredCard(card, math.sumprod(query, vector))
                    for card, vector in zip(index.cards, index.vectors, strict=True)
                ),
                depth,
            )
            for query in (_unit(vector) for vector in batch.vectors)
        ]
        return Retrieval(
            matches=tuple(fuse_query_rankings(rankings, k=k)),
            usage=RetrievalUsage(index_tokens=index_tokens, query_tokens=batch.input_tokens),
        )

    async def _index(self, catalog: ToolCatalog) -> tuple[_Index, int]:
        if (cached := self._indexes.get(catalog.fingerprint)) is not None:
            return cached, 0
        model_id = self._embedder.model_id
        texts = [self._search_text(card) for card in catalog]
        missing = [text for text in dict.fromkeys(texts) if (model_id, text) not in self._vectors]
        tokens = 0
        if missing:
            batch = await self._embedder.embed(missing, kind="document")
            tokens = batch.input_tokens
            for text, vector in zip(missing, batch.vectors, strict=True):
                self._vectors[model_id, text] = _unit(vector)
        index = _Index(cards=catalog.cards, vectors=tuple(self._vectors[model_id, text] for text in texts))
        self._indexes.put(catalog.fingerprint, index)
        return index, tokens


def _unit(vector: Sequence[float]) -> array[float]:
    norm = math.sqrt(math.sumprod(vector, vector))
    return array("f", (value / norm for value in vector) if norm else vector)
