import math

import pytest

from toolhunch import ToolCard, ToolCatalog
from toolhunch.retrieval import BM25Retriever, TextAnalyzer

pytestmark = pytest.mark.anyio

CATALOG = ToolCatalog(
    [
        # analysed lengths: 5, 6 and 7 terms → avgdl 6
        ToolCard(name="get_weather", description="Forecast for a city.", parameters={"properties": {"city": {}}}),
        ToolCard(name="send_email", description="Send an email to a user.", parameters={"properties": {"body": {}}}),
        ToolCard(name="get_diary_day", description="Read the diary entry of a day."),
    ]
)


async def ids(
    retriever: BM25Retriever, queries: list[str], *, k: int = 10, catalog: ToolCatalog = CATALOG
) -> list[str]:
    return [scored.card.id for scored in (await retriever.retrieve(queries, catalog, k=k)).matches]


async def test_lucene_bm25_score() -> None:
    result = await BM25Retriever().retrieve(["weather city"], CATALOG, k=10)

    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))  # "weather" and "city" each occur in one card
    norm = 1.2 * (1 - 0.75 + 0.75 * 5 / 6)  # get_weather has 5 terms
    expected = idf * (1 / (1 + norm)) + idf * (2 / (2 + norm))  # "city" twice: description + property
    assert [(s.card.id, s.score) for s in result.matches] == [("get_weather", pytest.approx(expected, rel=1e-12))]
    assert result.usage.index_tokens == result.usage.query_tokens == 0


async def test_identifiers_and_multiple_queries() -> None:
    retriever = BM25Retriever()
    assert await ids(retriever, ["diary"]) == ["get_diary_day"]
    # one ranking per query, fused by RRF: both rank first, the tie falls back to id order
    assert await ids(retriever, ["send email", "weather", "weather "]) == ["get_weather", "send_email"]


@pytest.mark.parametrize("queries", [[], [" "], ["the of"], ["???"], ["zebra"]])
async def test_queries_without_usable_terms_return_nothing(queries: list[str]) -> None:
    assert await ids(BM25Retriever(), queries) == []


async def test_catalog_edges() -> None:
    retriever = BM25Retriever()
    assert await ids(retriever, ["weather"], catalog=ToolCatalog([])) == []
    assert await ids(retriever, ["get"], k=50) == ["get_weather", "get_diary_day"]  # shorter card wins
    with pytest.raises(ValueError, match="k must be"):
        await retriever.retrieve(["weather"], CATALOG, k=0)


class CountingAnalyzer(TextAnalyzer):
    calls: int = 0

    def analyze(self, text: str, /) -> list[str]:
        CountingAnalyzer.calls += 1
        return super(CountingAnalyzer, self).analyze(text)


async def test_index_is_built_once_per_catalog() -> None:
    CountingAnalyzer.calls = 0
    retriever = BM25Retriever(analyzer=CountingAnalyzer())
    await retriever.retrieve(["weather"], CATALOG, k=1)
    await retriever.retrieve(["email"], ToolCatalog(reversed(CATALOG.cards)), k=1)  # same fingerprint
    assert CountingAnalyzer.calls == len(CATALOG) + 2  # the cards once, then one call per query
