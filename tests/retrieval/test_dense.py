import re
from collections.abc import Sequence

import pytest

from toolhunch import ToolCard, ToolCatalog
from toolhunch.retrieval import DenseRetriever, EmbeddingBatch, EmbeddingKind

pytestmark = pytest.mark.anyio

AXES = ("weather", "email", "diary")


class FakeEmbedder:
    """One axis per keyword; one billed token per word. Records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[EmbeddingKind, list[str]]] = []

    @property
    def model_id(self) -> str:
        return "fake"

    async def embed(self, texts: Sequence[str], /, *, kind: EmbeddingKind) -> EmbeddingBatch:
        self.calls.append((kind, list(texts)))
        words = [re.findall(r"[a-z]+", text.lower()) for text in texts]
        return EmbeddingBatch(
            vectors=tuple(tuple(float(ws.count(axis)) for axis in AXES) for ws in words),
            input_tokens=sum(len(ws) for ws in words),
        )


WEATHER = ToolCard(name="get_weather", description="Weather forecast.")
EMAIL = ToolCard(name="send_email", description="Send an email.")
DIARY = ToolCard(name="get_diary_day", description="Read the diary.")
CATALOG = ToolCatalog([WEATHER, EMAIL, DIARY])


async def test_cosine_ranking_and_query_fusion() -> None:
    retriever = DenseRetriever(FakeEmbedder())

    result = await retriever.retrieve(["weather"], CATALOG, k=3)
    assert [(s.card.id, s.score) for s in result.matches] == [
        ("get_weather", pytest.approx(1.0)),
        ("get_diary_day", 0.0),  # orthogonal cards tie at 0 and fall back to id order
        ("send_email", 0.0),
    ]
    fused = await retriever.retrieve(["weather", "email"], CATALOG, k=2)
    assert [s.card.id for s in fused.matches] == ["get_weather", "send_email"]


async def test_cards_are_embedded_once_and_usage_is_split() -> None:
    embedder = FakeEmbedder()
    retriever = DenseRetriever(embedder)

    first = await retriever.retrieve(["weather"], CATALOG, k=1)
    second = await retriever.retrieve(["email"], CATALOG, k=1)
    await retriever.retrieve(["diary"], ToolCatalog([*CATALOG, ToolCard(name="ping")]), k=1)

    assert [(kind, len(texts)) for kind, texts in embedder.calls] == [
        ("document", 3),
        ("query", 1),
        ("query", 1),
        ("document", 1),  # a changed catalog re-embeds only the new card
        ("query", 1),
    ]
    assert (first.usage.index_tokens, first.usage.query_tokens) == (15, 1)  # card texts: 4 + 5 + 6 words
    assert (second.usage.index_tokens, second.usage.query_tokens) == (0, 1)


@pytest.mark.parametrize(("queries", "catalog"), [([" ", ""], CATALOG), (["weather"], ToolCatalog([]))])
async def test_nothing_to_search_makes_no_request(queries: list[str], catalog: ToolCatalog) -> None:
    embedder = FakeEmbedder()
    assert (await DenseRetriever(embedder).retrieve(queries, catalog, k=5)).matches == ()
    assert embedder.calls == []
