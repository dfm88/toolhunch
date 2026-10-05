import hashlib
import json
import re
from collections.abc import Mapping
from datetime import date
from typing import Any

import anyio
import httpx2
import pytest

from toolhunch import DetailLevel, ScoredCard, ToolCard
from toolhunch.decision import (
    LOGPROB_LIMITS,
    Abstention,
    Answer,
    BinaryAnswer,
    CandidatesDoNotFit,
    ChoiceAnswer,
    ChoiceDecider,
    ChoiceQuestion,
    Decision,
    DecisionError,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    Exchange,
    ModelLimits,
    ThresholdKey,
    jev,
)
from toolhunch.decision.planner import NONE_KEY, build_state, plan_question
from toolhunch.tokens import HeuristicTokenizer

pytestmark = pytest.mark.anyio

TOOLS_50 = [f"t{i:02d}" for i in range(50)]

# A head for the whole question and a window per option, like a local encoder's.
LAYA_LIKE = ModelLimits(max_question_tokens=240, max_option_tokens=12, source="test", checked=date(2026, 10, 5))


class Words:
    """A token is a word, so a token budget can be worked out by hand."""

    def count(self, text: str, /) -> int:
        return len(text.split())


def scored(*ids: str, names: Mapping[str, str] | None = None, description: str = "") -> list[ScoredCard]:
    """Candidates in retrieval order; a card is named after its id unless `names` says otherwise."""
    renames = names or {}
    return [
        ScoredCard(ToolCard(name=renames.get(card_id, card_id), description=description, id=card_id), 1.0 - rank / 100)
        for rank, card_id in enumerate(ids)
    ]


def options_of(request: DecisionRequest) -> dict[str, str]:
    """The options, by key, of the one question a request holds."""
    question = request.questions["tool"]
    assert isinstance(question, ChoiceQuestion)
    return dict(question.options)


def ranked_ids(decision: Decision) -> list[str]:
    return [match.card.id for match in decision.ranked]


async def test_candidates_are_reordered_by_probability(fake_model: Any) -> None:
    model = fake_model({"b": 5.0, "c": 3.0})
    candidates = scored("a", "b", "c", "d")

    decision = await ChoiceDecider(model).decide("Request: x", candidates)

    assert ranked_ids(decision) == ["b", "c", "a", "d"]  # a and d tie: retrieval order
    assert [match.score for match in decision.ranked] == pytest.approx([0.5, 0.3, 0.1, 0.1])
    assert decision.probabilities == pytest.approx({"a": 0.1, "b": 0.5, "c": 0.3, "d": 0.1})
    assert decision.probabilities["b"] == pytest.approx(0.5)
    assert (decision.none_probability, decision.abstained) == (None, False)
    assert decision.usage.requests == 1
    assert decision.seconds == pytest.approx(0.1)
    assert decision.state_cut is False
    # One question with the id "tool": the options in retrieval order, no reserved option, the state as given.
    [request] = model.requests
    assert request.state == "Request: x"
    assert list(request.questions) == ["tool"]
    assert list(options_of(request)) == ["a", "b", "c", "d"]
    [exchange] = decision.exchanges
    assert (exchange.round, exchange.request, exchange.detail) == (1, request, DetailLevel.FULL)
    assert exchange.response.usage == DecisionUsage(1, 10, 1)
    planned = plan_question(
        "Request: x",
        [match.card for match in candidates],
        reserved=False,
        limits=model.limits,
        tokenizer=HeuristicTokenizer(),
        max_detail=DetailLevel.FULL,
    )
    assert planned is not None
    assert exchange.estimated_input_tokens == planned.estimated_input_tokens


async def test_duplicate_names_map_back_to_their_cards(fake_model: Any) -> None:
    names = {"web.search": "search", "docs.search": "search", "x.none": "none"}
    candidates = scored("web.search", "docs.search", "x.none", "get", names=names)
    # The second "search" is keyed "search #2", and a tool called "none" is "none #2" beside the reserved option.
    model = fake_model({"search #2": 4.0, "none #2": 3.0})

    decision = await ChoiceDecider(model, abstention=Abstention()).decide("Request: docs", candidates)

    assert list(options_of(model.requests[0])) == ["search", "search #2", "none #2", "get", "none"]
    assert ranked_ids(decision) == ["docs.search", "x.none", "web.search", "get"]
    assert decision.probabilities == pytest.approx({"docs.search": 0.4, "x.none": 0.3, "web.search": 0.1, "get": 0.1})
    assert decision.none_probability == pytest.approx(0.1)  # the reserved option, not the tool named "none"
    assert decision.abstained is False

    # Without the reserved option, a tool called "none" is an ordinary tool, keyed "none".
    model = fake_model({"none": 5.0})
    decision = await ChoiceDecider(model, abstention=Abstention(reserved_option=False)).decide(
        "Request: x", scored("none", "get")
    )
    assert list(options_of(model.requests[0])) == ["none", "get"]
    assert ranked_ids(decision) == ["none", "get"]
    assert (decision.none_probability, decision.abstained) == (None, False)


async def test_duplicate_names_map_back_to_their_cards_in_every_round(fake_model: Any) -> None:
    # Twenty-five tools all called "search": each question keys its own options "search", "search #2", ..., so the
    # same key names a different card in every question.
    ids = [f"s{i:02d}" for i in range(25)]
    candidates = scored(*ids, names=dict.fromkeys(ids, "search"))
    model = fake_model({"search #3": 8.0}, limits=LOGPROB_LIMITS)

    decision = await ChoiceDecider(model).decide("Request: x", candidates)

    # Round one: chunk 0 holds s00, s02, ... and chunk 1 holds s01, s03, ...; "search #3" is s04 in the first and
    # s05 in the second. Each keeps it and its chunk's first card. In the final question, s00, s01, s04 and s05
    # are keyed "search" to "search #4", and "search #3" is s04.
    assert [list(options_of(exchange.request))[2] for exchange in decision.exchanges] == ["search #3"] * 3
    assert set(decision.probabilities) == {"s00", "s01", "s04", "s05"}
    assert [exchange.option_card_ids["search #3"] for exchange in decision.exchanges] == ["s04", "s05", "s04"]
    assert [list(exchange.option_card_ids.values()) for exchange in decision.exchanges] == [
        ids[::2],
        ids[1::2],
        ["s00", "s01", "s04", "s05"],
    ]
    assert ranked_ids(decision) == [
        "s04",
        "s00",
        "s01",
        "s05",
        *(i for i in ids if i not in {"s00", "s01", "s04", "s05"}),
    ]
    assert decision.probabilities["s04"] == pytest.approx(8 / 11)


async def test_candidates_with_the_same_id_are_rejected(fake_model: Any) -> None:
    model = fake_model({})
    with pytest.raises(ValueError, match="'a'"):
        await ChoiceDecider(model).decide("Request: x", scored("a", "b", "a"))
    assert model.requests == []


async def test_zero_probability_ties_keep_retrieval_order(fake_model: Any) -> None:
    # A Jev-style rounded distribution: one option takes everything and the others are exactly 0.0.
    model = fake_model({"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0})
    decision = await ChoiceDecider(model).decide("Request: x", scored("a", "b", "c", "d"))
    assert ranked_ids(decision) == ["c", "a", "b", "d"]
    assert [match.score for match in decision.ranked] == [1.0, 0.0, 0.0, 0.0]
    # Ties above zero keep it too.
    model = fake_model({"a": 0.0, "b": 0.0, "c": 1.0, "d": 1.0})
    decision = await ChoiceDecider(model).decide("Request: x", scored("a", "b", "c", "d"))
    assert ranked_ids(decision) == ["c", "d", "a", "b"]


async def test_a_jev_style_answer_ranks_through_the_real_adapter() -> None:
    # As the server sends it: two decimals that sum to 0.99, several exact zeros, the keys out of order.
    body = {
        "answers": {"tool": {"type": "choice", "probabilities": {"d": 0.0, "c": 0.99, "b": 0.0, "a": 0.0}}},
        "usage": {"input_tokens": 41, "output_tokens": 0},
    }
    sent: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=body)

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = jev("jev-1.13.0", api_key="not-a-real-key", http_client=client)

    decision = await ChoiceDecider(model).decide("Request: x", scored("a", "b", "c", "d"))

    assert ranked_ids(decision) == ["c", "a", "b", "d"]
    assert decision.probabilities == {"c": 1.0, "a": 0.0, "b": 0.0, "d": 0.0}  # renormalised over 0.99
    assert decision.usage == DecisionUsage(1, 41, 0)
    assert decision.key.model == "jev-1.13.0@api.typesafe.ai"
    assert decision.key.prompt_version == "tool-choice-v1"  # the server owns Jev's prompt: nothing to join
    [request] = sent
    question = json.loads(request.content)["questions"]["tool"]
    assert list(question["criteria"]) == ["a", "b", "c", "d"]


async def test_abstention_rules(fake_model: Any) -> None:
    async def decide(
        weights: Mapping[str, float], *, none_weight: float = 1.0, abstention: Abstention | None = None
    ) -> Decision:
        model = fake_model(weights, none_weight=none_weight)
        return await ChoiceDecider(model, abstention=abstention).decide("Request: x", scored("a", "b"))

    # The reserved option wins: abstains, and records how likely "none" was. The ranking is still reported.
    decision = await decide({"a": 3.0, "b": 1.0}, none_weight=6.0, abstention=Abstention())
    assert decision.abstained is True
    assert decision.none_probability == pytest.approx(0.6)
    assert decision.probabilities == pytest.approx({"a": 0.3, "b": 0.1})  # the tools only
    assert ranked_ids(decision) == ["a", "b"]
    # A tie with the best tool abstains too.
    decision = await decide({"a": 2.0, "b": 1.0}, none_weight=2.0, abstention=Abstention())
    assert decision.abstained is True
    assert decision.none_probability == pytest.approx(0.4)
    # A tool wins by more than the reserved option: answered.
    decision = await decide({"a": 6.0, "b": 1.0}, abstention=Abstention())
    assert decision.abstained is False
    assert decision.none_probability == pytest.approx(0.125)

    # Without the reserved option only the threshold decides, and "below" is strict.
    no_reserved = Abstention(threshold=0.6, reserved_option=False)
    decision = await decide({"a": 1.0, "b": 1.0}, abstention=no_reserved)  # best 0.5
    assert decision.abstained is True
    assert decision.none_probability is None
    decision = await decide({"a": 7.0, "b": 3.0}, abstention=no_reserved)  # best 0.7
    assert decision.abstained is False
    decision = await decide({"a": 1.0, "b": 1.0}, abstention=Abstention(threshold=0.5, reserved_option=False))
    assert decision.abstained is False  # best 0.5 is at the threshold, not below it
    # With the reserved option the threshold still applies, when "none" is unlikely but no tool stands out.
    decision = await decide({"a": 1.0, "b": 1.0}, none_weight=0.1, abstention=Abstention(threshold=0.6))
    assert decision.abstained is True
    assert decision.none_probability == pytest.approx(0.1 / 2.1)

    # No abstention never abstains, even when the best tool has 0.01.
    decision = await ChoiceDecider(fake_model({})).decide("Request: x", scored(*(f"t{i:02d}" for i in range(100))))
    assert decision.abstained is False
    assert decision.ranked[0].score == pytest.approx(0.01)


async def test_two_rounds_report_only_the_final_distribution(fake_model: Any) -> None:
    # Chunk i holds the cards at positions i, i + 3, ...; the winners sit deep in their chunks, and the final
    # question ranks them in the reverse of retrieval order.
    weights = {"t30": 2.0, "t45": 8.0, "t31": 4.0, "t46": 10.0, "t32": 6.0, "t47": 12.0}
    model = fake_model(weights, limits=LOGPROB_LIMITS)
    candidates = scored(*TOOLS_50)
    state = build_state("Book a table", ["restaurant"])

    decision = await ChoiceDecider(model, abstention=Abstention(), tokenizer=Words()).decide(state, candidates)

    finalists = ["t47", "t46", "t45", "t32", "t31", "t30"]
    assert len(model.requests) == 4
    assert decision.usage.requests == 4
    assert decision.seconds == pytest.approx(0.2)  # round one ran at the same time: its slowest call, then the final
    assert decision.server_seconds == pytest.approx(0.02)
    assert [exchange.round for exchange in decision.exchanges] == [1, 1, 1, 2]
    assert [len(options_of(exchange.request)) for exchange in decision.exchanges] == [17, 17, 16, 7]
    assert all(NONE_KEY not in options_of(exchange.request) for exchange in decision.exchanges[:3])
    final = options_of(decision.exchanges[3].request)
    assert list(final) == ["t30", "t31", "t32", "t45", "t46", "t47", NONE_KEY]  # retrieval order, "none" last
    assert {request.state for request in model.requests} == {state}
    assert decision.shape["questions"] == [[17, 17, 16], [7]]
    # Only the final question's distribution is reported.
    assert list(decision.probabilities) == finalists  # best first, not the retrieval order of the options
    assert decision.probabilities == pytest.approx({card: weight / 43 for card, weight in weights.items()})
    assert decision.none_probability == pytest.approx(1 / 43)
    assert decision.abstained is False
    # The finalists come first by probability; the cards eliminated in round one follow at 0.0, in retrieval order.
    assert ranked_ids(decision) == finalists + [card for card in TOOLS_50 if card not in finalists]
    assert [match.score for match in decision.ranked[:6]] == pytest.approx(
        [12 / 43, 10 / 43, 8 / 43, 6 / 43, 4 / 43, 2 / 43]
    )
    assert {match.score for match in decision.ranked[6:]} == {0.0}
    # Each exchange records what the planner estimated for its question.
    planned = plan_question(
        state,
        [match.card for match in candidates if match.card.id in finalists],
        reserved=True,
        limits=LOGPROB_LIMITS,
        tokenizer=Words(),
        max_detail=DetailLevel.FULL,
    )
    assert planned is not None
    assert decision.exchanges[3].estimated_input_tokens == planned.estimated_input_tokens
    assert decision.exchanges[3].detail is DetailLevel.FULL


async def test_the_finalists_kept_are_the_ones_the_model_can_take(fake_model: Any) -> None:
    # Ten finalists per chunk would overflow the twenty options: the plan keeps six, three chunks and "none" make 19.
    model = fake_model({}, limits=LOGPROB_LIMITS)

    decision = await ChoiceDecider(model, abstention=Abstention(), finalists_per_chunk=10).decide(
        "Request: x", scored(*TOOLS_50)
    )

    assert decision.shape["questions"] == [[17, 17, 16], [19]]
    assert len(decision.probabilities) == 18
    assert decision.shape["finalists_per_chunk"] == 10  # what was configured, not what the plan kept


async def test_a_one_card_chunk_passes_without_a_call(fake_model: Any) -> None:
    # Three candidates and two options per question: chunks [t00, t02] and [t01]; f is 1.
    model = fake_model(
        {"t00": 3.0}, limits=ModelLimits(max_options_per_choice=2, source="test", checked=date(2026, 9, 28))
    )

    decision = await ChoiceDecider(model).decide("Request: x", scored("t00", "t01", "t02"))

    assert [exchange.round for exchange in decision.exchanges] == [1, 2]
    assert [list(options_of(exchange.request)) for exchange in decision.exchanges] == [["t00", "t02"], ["t00", "t01"]]
    assert decision.shape["questions"] == [[2], [2]]  # the one-card chunk is not a question
    assert ranked_ids(decision) == ["t00", "t01", "t02"]  # t02 lost its chunk: last, at 0.0
    assert [match.score for match in decision.ranked] == pytest.approx([0.75, 0.25, 0.0])


async def test_round_one_asks_its_chunks_at_the_same_time(fake_model: Any) -> None:
    arrived = 0
    together = anyio.Event()

    async def wait_for_the_others(request: DecisionRequest) -> None:
        nonlocal arrived
        if NONE_KEY not in options_of(request):  # round one; the final question has the reserved option
            arrived += 1
            if arrived == 3:
                together.set()
            await together.wait()  # returns only once all three chunks are waiting: a serial loop would hang

    model = fake_model({}, limits=LOGPROB_LIMITS, before=wait_for_the_others)

    with anyio.fail_after(5):
        decision = await ChoiceDecider(model, abstention=Abstention()).decide("Request: x", scored(*TOOLS_50))

    assert decision.usage.requests == 4


@pytest.mark.parametrize("error_type", [DecisionError, ValueError])
async def test_a_failing_round_one_chunk_raises_the_models_error(fake_model: Any, error_type: type[Exception]) -> None:
    async def fail(request: DecisionRequest) -> None:
        first = next(iter(options_of(request)))  # chunk i starts with card ti
        if first == "t01":
            await anyio.sleep(0.05)  # fails after the third chunk does
            raise error_type("fake@test: the second chunk failed")
        if first == "t02":
            raise error_type("fake@test: the third chunk failed")

    decider = ChoiceDecider(fake_model({}, limits=LOGPROB_LIMITS, before=fail), abstention=Abstention())

    # The model's own error comes out, not an exception group, and the first failing chunk wins whatever the timing.
    with pytest.raises(error_type, match="the second chunk failed"):
        await decider.decide("Request: x", scored(*TOOLS_50))


async def test_finalists_that_do_not_fit_the_token_budget_raise(fake_model: Any) -> None:
    # Room for five one-word names per question: 40 candidates make eight chunks, and not even one finalist from
    # each fits the final question with the reserved option (4 + 11 + 8 x 6 + 13 = 76 tokens).
    budget = ModelLimits(
        max_options_per_choice=20, max_state_plus_question_tokens=45, source="test", checked=date(2026, 9, 28)
    )
    model = fake_model({}, limits=budget)
    decider = ChoiceDecider(model, abstention=Abstention(), tokenizer=Words())

    with pytest.raises(CandidatesDoNotFit, match="even one finalist") as raised:
        await decider.decide("Request: Book a table", scored(*TOOLS_50[:40]))
    assert raised.value.card_id is None
    assert len(model.requests) == 0  # refused before the first round, no longer after it


async def test_the_threshold_key_follows_the_payload_shape(fake_model: Any) -> None:
    async def decide(count: int, *, reserved: bool = True, detail: DetailLevel = DetailLevel.FULL) -> Decision:
        decider = ChoiceDecider(fake_model({}), abstention=Abstention(reserved_option=reserved), max_detail=detail)
        return await decider.decide("Request: x", scored(*TOOLS_50[:count]))

    base = await decide(20)
    assert base.shape == {
        "candidates": 20,
        "questions": [[21]],
        "reserved_option": True,
        "max_detail": "FULL",
        "finalists_per_chunk": 2,
        "questions_per_request": 1,
        "option_keys": "names-v1",
    }
    # Changing this recipe changes every stored threshold's key: pinned on purpose.
    assert base.key == ThresholdKey("fake@test", "tool-choice-v1", "db6d5d57dd5971bd", "choice")
    canonical = json.dumps(base.shape, sort_keys=True, separators=(",", ":")).encode()
    assert base.key.payload_shape == hashlib.sha256(canonical).hexdigest()[:16]
    assert str(base.key) == "fake@test|tool-choice-v1|db6d5d57dd5971bd|choice"

    # Each knob that changes what is asked changes the payload shape.
    variants = [
        await decide(50),
        await decide(20, reserved=False),
        await decide(20, detail=DetailLevel.BRIEF),
    ]
    assert len({base.key.payload_shape, *(variant.key.payload_shape for variant in variants)}) == 4
    assert variants[0].shape["questions"] == [[51]]
    assert variants[1].shape["questions"] == [[20]]
    assert variants[1].shape["reserved_option"] is False
    assert variants[2].shape["max_detail"] == "BRIEF"
    assert [variant.key.model for variant in variants] == ["fake@test"] * 3
    assert all(str(variant.key).count("|") == 3 for variant in variants)

    # A model's own prompt version is joined to ours.
    model = fake_model({}, prompt_version="letters-v1")
    versioned = await ChoiceDecider(model).decide("Request: x", scored("a", "b"))
    assert versioned.key.prompt_version == "tool-choice-v1+letters-v1"
    assert versioned.key.model == model.model_id
    assert versioned.key.question_kind == "choice"
    assert str(versioned.key).count("|") == 3

    # Two rounds record every question that was asked, per round.
    two_rounds = await ChoiceDecider(fake_model({}, limits=LOGPROB_LIMITS), abstention=Abstention()).decide(
        "Request: x", scored(*TOOLS_50)
    )
    assert two_rounds.shape["questions"] == [[17, 17, 16], [7]]
    assert two_rounds.key.payload_shape == "0fcf3645d5f1b6f9"


async def test_new_settings_enter_the_key_only_when_they_change_the_questions(fake_model: Any) -> None:
    candidates = scored(*TOOLS_50[:20])
    base = await ChoiceDecider(fake_model({}), abstention=Abstention()).decide("Request: x", candidates)
    assert base.key.payload_shape == "db6d5d57dd5971bd"  # unchanged: the thresholds measured so far still hold
    assert not {"min_detail", "budgets", "tokenizer"} & set(base.shape)

    # The floor changes what is asked. Six options and 60 words a question: a group of six described cards takes 79
    # at FULL and 67 at BRIEF but 49 at names, so without a floor round one asks two questions at names. With
    # BRIEF as the floor the groups shrink to four cards, 57 at FULL, and the final's three finalists keep it.
    squeezed = ModelLimits(
        max_options_per_choice=6, max_state_plus_question_tokens=60, source="test", checked=date(2026, 10, 5)
    )
    described = scored(*TOOLS_50[:12], description="Does a thing. Then another.")
    unfloored = await ChoiceDecider(fake_model({}, limits=squeezed), abstention=Abstention(), tokenizer=Words()).decide(
        "Request: x", described
    )
    floored = await ChoiceDecider(
        fake_model({}, limits=squeezed), abstention=Abstention(), tokenizer=Words(), min_detail=DetailLevel.BRIEF
    ).decide("Request: x", described)
    assert [exchange.round for exchange in unfloored.exchanges] == [1, 1, 2]
    assert {exchange.detail for exchange in unfloored.exchanges} == {DetailLevel.NAME}
    assert [exchange.round for exchange in floored.exchanges] == [1, 1, 1, 2]  # two rounds, more groups
    assert all(exchange.detail >= DetailLevel.BRIEF for exchange in floored.exchanges)
    assert floored.shape["min_detail"] == "BRIEF"
    assert "min_detail" not in unfloored.shape
    assert floored.key != unfloored.key != base.key

    budgeted = await ChoiceDecider(fake_model({}, limits=LAYA_LIKE), abstention=Abstention()).decide(
        "Request: x", candidates
    )
    assert budgeted.shape["budgets"] == {
        "max_options_per_choice": None,
        "max_request_tokens": None,
        "max_state_plus_question_tokens": None,
        "max_question_tokens": 240,
        "max_option_tokens": 12,
        "max_text_tokens": None,
    }
    assert budgeted.key != base.key

    counted = await ChoiceDecider(fake_model({}), abstention=Abstention(), tokenizer=Words()).decide(
        "Request: x", candidates
    )
    assert counted.shape["tokenizer"] == "Words"
    assert counted.key != base.key

    # The default tokenizer, named or not, is not a change; a floor above the ceiling is refused.
    named = ChoiceDecider(fake_model({}), abstention=Abstention(), tokenizer=HeuristicTokenizer())
    assert (await named.decide("Request: x", candidates)).key == base.key
    with pytest.raises(ValueError, match="min_detail=FULL is above max_detail=BRIEF"):
        ChoiceDecider(fake_model({}), min_detail=DetailLevel.FULL, max_detail=DetailLevel.BRIEF)


async def test_max_detail_caps_how_the_cards_are_described(fake_model: Any) -> None:
    described = "Does a thing. Then another."
    model = fake_model({})
    decision = await ChoiceDecider(model, max_detail=DetailLevel.NAME).decide(
        "Request: x", scored("a", "b", "c", description=described)
    )
    assert options_of(model.requests[0]) == {"a": "a", "b": "b", "c": "c"}
    assert decision.exchanges[0].detail is DetailLevel.NAME

    # Both rounds of a two-round search describe the cards at that level too.
    model = fake_model({}, limits=LOGPROB_LIMITS)
    decision = await ChoiceDecider(model, abstention=Abstention(), max_detail=DetailLevel.BRIEF).decide(
        "Request: x", scored(*TOOLS_50, description=described)
    )
    assert [exchange.detail for exchange in decision.exchanges] == [DetailLevel.BRIEF] * 4
    for exchange in decision.exchanges:
        texts = [text for key, text in options_of(exchange.request).items() if key != NONE_KEY]
        assert all(text.endswith(": Does a thing.") for text in texts)


async def test_the_rendered_detail_is_recorded_and_is_not_part_of_the_key(fake_model: Any) -> None:
    # The same search size and settings, cards short in one search and long in the other: the rendering differs.
    squeezed = ModelLimits(max_state_plus_question_tokens=1_000, source="test", checked=date(2026, 9, 28))
    model = fake_model({}, limits=squeezed)
    short = await ChoiceDecider(model).decide("Request: x", scored("a", "b", "c", description="Does a thing."))
    long = await ChoiceDecider(model).decide(
        "Request: x", scored("a", "b", "c", description="Does a thing. " + "word " * 800)
    )

    assert short.exchanges[0].detail is DetailLevel.FULL
    assert long.exchanges[0].detail is DetailLevel.BRIEF
    assert short.key == long.key


async def test_no_call_for_zero_or_one_candidate(fake_model: Any) -> None:
    # A state longer than the model takes is left alone when nothing is sent.
    tiny = ModelLimits(max_text_tokens=50, source="test", checked=date(2026, 9, 28))
    unfit_state = build_state("word " * 500, ["restaurant booking"])
    for abstention in (None, Abstention(), Abstention(reserved_option=False)):
        model = fake_model({}, limits=tiny)
        decision = await ChoiceDecider(model, abstention=abstention).decide(unfit_state, [])
        assert model.requests == []
        assert decision.ranked == ()
        assert decision.probabilities == {}
        assert decision.none_probability is None
        assert decision.abstained is (abstention is not None)
        assert (decision.state_cut, decision.exchanges) == (False, ())
        assert (decision.seconds, decision.server_seconds, decision.usage) == (0.0, 0.0, DecisionUsage())
        assert decision.shape["questions"] == []

    # One candidate without a reserved option: it is the answer, with probability 1.
    candidates = scored("a")
    for abstention in (None, Abstention(reserved_option=False)):
        model = fake_model({}, limits=tiny)
        decision = await ChoiceDecider(model, abstention=abstention).decide(unfit_state, candidates)
        assert model.requests == []
        assert decision.ranked == (ScoredCard(candidates[0].card, 1.0),)
        assert decision.probabilities == {"a": 1.0}
        assert (decision.none_probability, decision.abstained, decision.state_cut) == (None, False, False)
        assert (decision.exchanges, decision.seconds, decision.server_seconds) == ((), 0.0, 0.0)
        assert decision.shape["questions"] == []
    # Only a threshold above 1 can make a lone candidate abstain.
    decision = await ChoiceDecider(fake_model({}), abstention=Abstention(threshold=1.5, reserved_option=False)).decide(
        "Request: x", candidates
    )
    assert decision.abstained is True

    # With the reserved option there is something to ask: "a" or "none".
    model = fake_model({"a": 3.0})
    decision = await ChoiceDecider(model, abstention=Abstention()).decide("Request: x", candidates)
    assert [list(options_of(request)) for request in model.requests] == [["a", NONE_KEY]]
    assert decision.probabilities == pytest.approx({"a": 0.75})
    assert decision.none_probability == pytest.approx(0.25)
    assert decision.abstained is False
    assert decision.shape["questions"] == [[2]]


async def test_a_long_state_is_cut_and_flagged(fake_model: Any) -> None:
    model = fake_model({}, limits=ModelLimits(max_text_tokens=50, source="test", checked=date(2026, 9, 28)))
    state = build_state("word " * 500 + "tail-marker", ["restaurant booking"])

    decision = await ChoiceDecider(model).decide(state, scored("a", "b"))

    assert decision.state_cut is True
    sent = model.requests[0].state
    assert sent != state
    assert " … " in sent
    assert sent.startswith("Request: word")
    assert sent.endswith("tail-marker\nSearch queries: restaurant booking")  # the queries line survives, whole

    # The configured tokenizer sets the budget: 50 words less the 11 of the instructions is the most the state may take.
    by_words = fake_model({}, limits=ModelLimits(max_text_tokens=50, source="test", checked=date(2026, 9, 28)))
    await ChoiceDecider(by_words, tokenizer=Words()).decide(state, scored("a", "b"))
    assert 38 <= Words().count(by_words.requests[0].state) <= 39


@pytest.mark.parametrize(
    "answers",
    [
        pytest.param({"tool": ChoiceAnswer(probabilities={"a": 0.5, "ghost": 0.5})}, id="an-unknown-option"),
        pytest.param({"tool": ChoiceAnswer(probabilities={"a": 1.0})}, id="a-missing-option"),
        pytest.param({"tool": ChoiceAnswer(probabilities={"a": 0.4, "b": 0.4, "c": 0.2})}, id="an-extra-option"),
        pytest.param({"other": ChoiceAnswer(probabilities={"a": 0.5, "b": 0.5})}, id="no-answer-to-the-question"),
        pytest.param({"tool": BinaryAnswer(probability=0.5)}, id="not-a-choice-answer"),
        pytest.param({}, id="no-answers"),
    ],
)
async def test_answers_for_unknown_options_raise(fake_model: Any, answers: dict[str, Answer]) -> None:
    model = fake_model({}, answers=answers)
    with pytest.raises(DecisionError, match=rf"^{re.escape(model.model_id)}: "):
        await ChoiceDecider(model).decide("Request: x", scored("a", "b"))


def exchange_of(round_number: int, *, seconds: float, server: float | None) -> Exchange:
    question = ChoiceQuestion(instructions="q", options={"a": "a", "b": "b"})
    answer = ChoiceAnswer(probabilities={"a": 0.5, "b": 0.5})
    return Exchange(
        round=round_number,
        request=DecisionRequest(state="s", questions={"tool": question}),
        response=DecisionResponse(
            answers={"tool": answer}, usage=DecisionUsage(1, 10, 1), seconds=seconds, server_seconds=server, raw={}
        ),
        detail=DetailLevel.FULL,
        estimated_input_tokens=5,
    )


def decision_of(*exchanges: Exchange) -> Decision:
    return Decision(
        ranked=(),
        probabilities={},
        none_probability=None,
        abstained=False,
        key=ThresholdKey("m@h", "p", "s", "choice"),
        shape={},
        state_cut=False,
        exchanges=exchanges,
    )


def test_latency_is_the_critical_path_of_the_rounds() -> None:
    decision = decision_of(
        exchange_of(1, seconds=0.3, server=0.03),
        exchange_of(1, seconds=0.1, server=0.01),
        exchange_of(2, seconds=0.2, server=0.02),
    )
    assert decision.seconds == pytest.approx(0.5)  # the slowest call of round one, then the final: not 0.6, not 0.3
    assert decision.server_seconds == pytest.approx(0.05)
    assert decision.usage == DecisionUsage(3, 30, 3)
    # One response without a server time leaves the total unknown; the wall time is unaffected.
    partial = decision_of(exchange_of(1, seconds=0.3, server=0.03), exchange_of(2, seconds=0.2, server=None))
    assert partial.server_seconds is None
    assert partial.seconds == pytest.approx(0.5)


def test_sequential_seconds_add_every_call() -> None:
    decision = decision_of(
        exchange_of(1, seconds=0.1, server=None),
        exchange_of(1, seconds=0.3, server=None),
        exchange_of(2, seconds=0.2, server=None),
    )
    assert decision.seconds == pytest.approx(0.5)  # round one's slowest, then the final
    assert decision.sequential_seconds == pytest.approx(0.6)  # a server that answers one request at a time
    assert decision_of().sequential_seconds == 0.0


def test_a_model_without_choice_is_rejected(fake_model: Any) -> None:
    with pytest.raises(ValueError, match="choice"):
        ChoiceDecider(fake_model({}, kinds=frozenset({"binary"})))


def test_finalists_per_chunk_must_be_positive(fake_model: Any) -> None:
    with pytest.raises(ValueError, match="finalists_per_chunk"):
        ChoiceDecider(fake_model({}), finalists_per_chunk=0)
