from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from toolhunch import BM25Retriever, Retrieval, RetrievalUsage, ScoredCard, ToolCard, ToolCatalog, ToolSearchPipeline
from toolhunch.decision import ChoiceDecider, DecisionError, DecisionUsage, ModelLimits

pytestmark = pytest.mark.anyio


class StubRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        self.calls.append((list(queries), k))
        cards = [ToolCard(name="search", id="web.search"), ToolCard(name="search", id="docs.search")]
        return Retrieval(tuple(ScoredCard(card, 1.0) for card in cards), RetrievalUsage(query_tokens=3))


async def test_search_runs_the_retriever_and_traces_the_stage() -> None:
    retriever = StubRetriever()

    result = await ToolSearchPipeline(retriever, k=2).search(["find docs"], ToolCatalog([]))

    assert retriever.calls == [(["find docs"], 2)]
    assert result.ids == ["web.search", "docs.search"]
    assert result.names == ["search", "search"]
    [stage] = result.trace
    assert (stage.stage, stage.candidates, stage.usage) == ("retrieve", 2, RetrievalUsage(query_tokens=3))
    assert stage.seconds >= 0


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="k must be"):
        ToolSearchPipeline(StubRetriever(), k=0)


@pytest.mark.parametrize("top_n", [0, -1])
def test_top_n_must_be_positive(top_n: int) -> None:
    with pytest.raises(ValueError, match=f"top_n must be at least 1, got {top_n}"):
        ToolSearchPipeline(StubRetriever(), top_n=top_n)


async def test_without_a_decider_nothing_changes() -> None:
    retriever = StubRetriever()

    result = await ToolSearchPipeline(retriever, k=2, top_n=1).search(
        ["find docs"], ToolCatalog([]), context="Where are the docs?"
    )

    assert result.ids == ["web.search"]  # top_n cuts the retriever's order
    assert result.decision is None
    assert not result.abstained
    assert [stage.stage for stage in result.trace] == ["retrieve"]
    assert retriever.calls == [(["find docs"], 2)]  # the context goes nowhere


async def test_the_decider_reorders_and_traces_its_stage(recording_decider: Any) -> None:
    decider = recording_decider(order=["docs.search", "web.search"], usage=DecisionUsage(1, 40, 1))

    result = await ToolSearchPipeline(StubRetriever(), decider=decider, k=2).search(
        ["find docs", "  "], ToolCatalog([]), context="Where are the docs?"
    )

    assert decider.states == ["Request: Where are the docs?\nSearch queries: find docs"]
    assert decider.candidates == [["web.search", "docs.search"]]  # what the retriever found, in its order
    assert result.ids == ["docs.search", "web.search"]
    assert result.decision is not None
    assert not result.abstained
    assert [stage.stage for stage in result.trace] == ["retrieve", "decide"]
    assert result.trace[0].usage == RetrievalUsage(query_tokens=3)
    assert result.trace[1].candidates == 2
    assert result.trace[1].usage == DecisionUsage(1, 40, 1)


async def test_the_decide_stage_is_timed_by_the_wall_clock(recording_decider: Any) -> None:
    # `Decision.seconds` is what the responses report (a replayed response repeats its first run's, here 0.0); the
    # trace is measured around the call.
    decider = recording_decider(delay=0.05)

    result = await ToolSearchPipeline(StubRetriever(), decider=decider, k=2).search(["find docs"], ToolCatalog([]))

    assert result.decision is not None
    assert result.decision.seconds == 0.0
    assert result.trace[1].seconds >= 0.04


async def test_top_n_cuts_the_ranking_the_decider_made(recording_decider: Any) -> None:
    decider = recording_decider(order=["docs.search", "web.search"])

    result = await ToolSearchPipeline(StubRetriever(), decider=decider, k=2, top_n=1).search(
        ["find docs"], ToolCatalog([])
    )

    assert result.ids == ["docs.search"]  # the decider's first, not the retriever's
    assert result.decision is not None
    assert len(result.decision.ranked) == 2  # the decision keeps every candidate
    assert result.trace[1].candidates == 2  # the stage trace is what the stage returned, before the cut


async def test_the_state_is_built_from_the_cleaned_queries(recording_decider: Any) -> None:
    decider = recording_decider()

    await ToolSearchPipeline(StubRetriever(), decider=decider, k=2).search(
        ["find docs", "find docs ", "???", " Find the docs "], ToolCatalog([])
    )

    assert decider.states == ["Search queries: find docs | Find the docs"]  # no context: no request line


async def test_abstention_is_reported_and_matches_are_kept(recording_decider: Any) -> None:
    decider = recording_decider(order=["docs.search"], abstain=True)

    result = await ToolSearchPipeline(StubRetriever(), decider=decider, k=2).search(["find docs"], ToolCatalog([]))

    assert result.abstained is True
    assert result.decision is not None
    assert result.decision.abstained is True
    assert result.ids == ["docs.search", "web.search"]  # the ranking is reported either way


async def test_decider_errors_propagate(recording_decider: Any) -> None:
    error = DecisionError("fake@test: the model is down")
    decider = recording_decider(error=error)

    with pytest.raises(DecisionError, match="the model is down") as raised:  # no fallback to the retrieval order
        await ToolSearchPipeline(StubRetriever(), decider=decider, k=2).search(["find docs"], ToolCatalog([]))

    assert raised.value is error


async def test_a_long_context_is_cut_through_the_pipeline(fake_model: Any) -> None:
    # A pasted document as the request, far over what the model takes in one text.
    model = fake_model({}, limits=ModelLimits(max_text_tokens=50, source="test", checked=date(2026, 9, 28)))
    pipeline = ToolSearchPipeline(StubRetriever(), decider=ChoiceDecider(model), k=2)

    result = await pipeline.search(["find docs"], ToolCatalog([]), context="word " * 500 + "tail-marker")

    assert result.decision is not None
    assert result.decision.state_cut is True
    [request] = model.requests
    assert request.state.startswith("Request: word")
    assert " … " in request.state  # the context is cut in the middle
    assert request.state.endswith("tail-marker\nSearch queries: find docs")  # its end and the queries line are whole
    assert result.ids == ["web.search", "docs.search"]  # and the candidates were ranked all the same


async def test_a_bare_string_is_rejected() -> None:
    catalog = ToolCatalog([ToolCard(name="get_weather", description="Weather forecast.")])
    with pytest.raises(TypeError, match="list"):  # a str is a Sequence[str]: it would search letter by letter
        await ToolSearchPipeline(BM25Retriever()).search("weather forecast", catalog)
