"""Our BM25 against bm25s (`method="lucene"`), the implementation ToolRet's own baseline uses."""

import random
from typing import Any

import bm25s  # pyright: ignore[reportMissingTypeStubs]
import pytest

from toolhunch import BM25Retriever, ToolCard, ToolCatalog
from toolhunch.retrieval import TextAnalyzer

pytestmark = pytest.mark.anyio

VOCABULARY = [f"term{i}" for i in range(80)]
ZIPF = [1 / (rank + 1) for rank in range(len(VOCABULARY))]  # a few common terms, many rare ones


async def test_every_score_matches_bm25s_lucene() -> None:
    rng = random.Random(0)
    catalog = ToolCatalog(
        ToolCard(name=f"doc{i:03}", description=" ".join(rng.choices(VOCABULARY, ZIPF, k=rng.randint(3, 30))))
        for i in range(200)
    )
    analyzer = TextAnalyzer()
    ours = BM25Retriever(analyzer=analyzer, search_text=lambda card: card.description)
    library: Any = bm25s
    reference = library.BM25(k1=1.2, b=0.75, method="lucene")
    reference.index([analyzer.analyze(card.description) for card in catalog], show_progress=False)

    for _ in range(20):
        query = rng.sample(VOCABULARY, k=rng.randint(1, 4))  # distinct terms: bm25s would count repeats
        expected = reference.get_scores(query)
        result = await ours.retrieve([" ".join(query)], catalog, k=len(catalog))
        scores = {match.card.id: match.score for match in result.matches}
        for position, card in enumerate(catalog):
            assert scores.get(card.id, 0.0) == pytest.approx(float(expected[position]), rel=1e-4)  # bm25s: float32
