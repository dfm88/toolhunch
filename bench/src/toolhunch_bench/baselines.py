"""Baselines from other libraries, reimplemented as retrievers so every arm runs through the same pipeline.

`KeywordsRetriever` copies `keywords_search_fn` from `pydantic_ai/toolsets/_tool_search.py`
(pydantic-ai-slim 2.50.0): the algorithm behind `ToolSearch(strategy="keywords")` and the local
fallback of the default strategy. It is copied because that module is private;
`bench/tests/test_keywords_parity.py` runs the public path and fails when upstream changes.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from toolhunch import Retrieval, ScoredCard
from toolhunch.retrieval.base import check_k

if TYPE_CHECKING:
    from collections.abc import Sequence

    from toolhunch import ToolCard, ToolCatalog

__all__ = ["KeywordsRetriever"]

_TOKEN = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


class KeywordsRetriever:
    """Pydantic AI's keyword-overlap search as a [`Retriever`][toolhunch.Retriever].

    A card scores the number of distinct query terms (lowercase alphanumeric runs of all queries
    joined) found in its name or description; cards scoring 0 are dropped and the rest sorted by
    score, stable over the input order. As upstream, card terms are recomputed on every call, so
    latency is comparable. Upstream's input order is the order tools were registered; here it is a
    `random.Random(seed)` permutation of the catalog, so ties do not follow card ids.
    """

    def __init__(self, *, seed: int = 0, order: Sequence[str] | None = None) -> None:
        """Fix the input order with `seed`; `order` (card ids, for parity tests) overrides it."""
        self._seed = seed
        self._order = order
        self._orders: dict[str, list[ToolCard]] = {}

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Rank `catalog` by keyword overlap; see [`Retriever.retrieve`][toolhunch.Retriever.retrieve]."""
        check_k(k)
        terms = _terms(" ".join(queries))
        if not terms:
            return Retrieval()
        scored: list[tuple[int, ToolCard]] = []
        for card in self._input_order(catalog):
            score = len(terms & _terms(f"{card.name} {card.description}"))
            if score > 0:
                scored.append((score, card))
        scored.sort(key=lambda item: item[0], reverse=True)
        return Retrieval(matches=tuple(ScoredCard(card, float(score)) for score, card in scored[:k]))

    def _input_order(self, catalog: ToolCatalog) -> list[ToolCard]:
        if self._order is not None:
            return [catalog[card_id] for card_id in self._order]
        if (cards := self._orders.get(catalog.fingerprint)) is None:
            cards = list(catalog)
            random.Random(self._seed).shuffle(cards)
            self._orders[catalog.fingerprint] = cards
        return cards
