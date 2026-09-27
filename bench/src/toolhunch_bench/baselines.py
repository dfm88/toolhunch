"""Baselines from other libraries, reimplemented as retrievers so every arm runs through the same pipeline.

`KeywordsRetriever` copies `keywords_search_fn` from `pydantic_ai/toolsets/_tool_search.py`
(pydantic-ai-slim 2.50.0): the algorithm behind `ToolSearch(strategy="keywords")` and the local
fallback of the default strategy. It is copied because that module is private;
`bench/tests/test_keywords_parity.py` runs the public path and fails when upstream changes.

`BM25sToolRetRetriever` replays ToolRet's own BM25 baseline (`eval_bm25` in `toolret/eval.py`,
mangopy/tool-retrieval-benchmark @ d181f1c, 2025-03-30), so its numbers can sit next to the paper's.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Any

import bm25s  # pyright: ignore[reportMissingTypeStubs]

from toolhunch import Retrieval, ScoredCard
from toolhunch.retrieval.base import check_k

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from toolhunch import ToolCard, ToolCatalog

__all__ = ["BM25sToolRetRetriever", "KeywordsRetriever"]

_bm25s: Any = bm25s  # untyped; used only by BM25sToolRetRetriever

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


class BM25sToolRetRetriever:
    """ToolRet's BM25 baseline: bm25s exactly as `eval_bm25` calls it.

    `bm25s.BM25()` with its defaults (`k1=1.5`, `b=0.75`, Lucene variant) indexes each tool's raw
    JSON text tokenized with `stopwords="en"`; the query is tokenized with `bm25s.tokenize`'s
    defaults, which also drop English stop words, and repeated query terms count; a query left with
    no terms searches `"NONE"`. ToolRet pins bm25s 0.2.5, whose `tokenize` and `BM25` defaults are the
    ones used here. Two differences: the corpus is MTEB's shared catalog (the released script's
    `load_tools(task)` call does not run as written, so the paper's per-task corpus is uncertain), and
    zero-score results are dropped as in every other arm. Several queries are joined by a space.
    """

    def __init__(self, *, raw_text: Mapping[str, str]) -> None:
        """Index `raw_text[card.id]`, the corpus row's JSON text, for each card."""
        self._raw_text = raw_text
        self._indexes: dict[str, Any] = {}

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        """Rank `catalog` with bm25s; see [`Retriever.retrieve`][toolhunch.Retriever.retrieve]."""
        check_k(k)
        if not len(catalog):
            return Retrieval()
        tokens = _bm25s.tokenize(" ".join(queries), show_progress=False)
        if len(tokens.vocab) == 0:
            tokens = _bm25s.tokenize("NONE", stopwords=[], show_progress=False)
        positions, scores = self._index(catalog).retrieve(tokens, k=min(k, len(catalog)), show_progress=False)
        return Retrieval(
            matches=tuple(
                ScoredCard(catalog.cards[int(position)], float(score))
                for position, score in zip(positions[0], scores[0], strict=True)
                if score > 0
            )
        )

    def _index(self, catalog: ToolCatalog) -> Any:
        if (index := self._indexes.get(catalog.fingerprint)) is None:
            corpus = [self._raw_text[card.id] for card in catalog]
            index = _bm25s.BM25()
            index.index(_bm25s.tokenize(corpus, stopwords="en", show_progress=False), show_progress=False)
            self._indexes[catalog.fingerprint] = index
        return index
