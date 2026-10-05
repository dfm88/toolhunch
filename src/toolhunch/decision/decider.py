"""Stage 2 of a search: re-rank the retrieval candidates with a decision model, and abstain when none fits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import anyio

from toolhunch.cards import DetailLevel
from toolhunch.decision.base import ChoiceAnswer, DecisionError, DecisionRequest, DecisionUsage
from toolhunch.decision.planner import (
    NONE_KEY,
    PROMPT_VERSION,
    CandidatesDoNotFit,
    fit_state,
    plan_question,
    plan_rounds,
)
from toolhunch.retrieval.base import ScoredCard
from toolhunch.tokens import HeuristicTokenizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from toolhunch.cards import ToolCard
    from toolhunch.decision.base import DecisionModel, DecisionResponse, QuestionKind
    from toolhunch.decision.planner import PlannedQuestion, RoundPlan
    from toolhunch.tokens import Tokenizer

__all__ = ["Abstention", "ChoiceDecider", "Decider", "Decision", "Exchange", "ThresholdKey"]

_QUESTION_ID = "tool"  # every request holds one question, under this id
_OPTION_KEYS = "names-v1"  # the scheme `option_keys` names options by: change it only together with that function


@dataclass(frozen=True, slots=True, kw_only=True)
class Abstention:
    """When the decider says "none of these" instead of naming a tool.

    A decision abstains if the reserved option is no less likely than the best tool (a tie abstains), or if the
    best tool falls short of `threshold`. Probabilities are comparable only within one question, so a threshold is
    good for one `ThresholdKey`: change the model, the prompt version or the payload shape and every probability
    moves.

    Attributes:
        threshold: The probability the best tool needs to count as an answer. At 0 it adds no condition.
        reserved_option: Whether the last question includes the reserved "none of these" option. Without it, only
            `threshold` can make the decider abstain.
    """

    threshold: float = 0.0
    reserved_option: bool = True


@dataclass(frozen=True, slots=True)
class ThresholdKey:
    """What a threshold holds for: one model, one prompt version and one payload shape.

    `str(key)` joins the four fields with `"|"`. Record it wherever a threshold is applied or measured.

    Attributes:
        model: The model's `model_id`.
        prompt_version: The planner's prompt version, and the model's own after a `"+"` when it has one.
        payload_shape: A 16-character hex digest (SHA-256, canonical JSON with sorted keys) of `Decision.shape`.
        question_kind: The kind of question the probabilities answer.
    """

    model: str
    prompt_version: str
    payload_shape: str
    question_kind: QuestionKind

    def __str__(self) -> str:
        return "|".join((self.model, self.prompt_version, self.payload_shape, self.question_kind))


@dataclass(frozen=True, slots=True, kw_only=True)
class Exchange:
    """One call to the decision model and what came back.

    Attributes:
        round: 1 for the only question and for the first of two rounds, 2 for the final question.
        request: What was sent.
        response: What the model returned, raw provenance included.
        detail: The level of detail the cards were shown at. It depends on how long the cards are, so it can differ
            between searches and is not part of `Decision.key`. A card over the model's per-text cap was shown lower.
        estimated_input_tokens: What the planner counted for the question against the model's token budget.
        option_card_ids: The asked option keys mapped to card ids, excluding the reserved option.
    """

    round: int
    request: DecisionRequest
    response: DecisionResponse
    detail: DetailLevel
    estimated_input_tokens: int
    option_card_ids: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True, kw_only=True)
class Decision:
    """What a decider concluded about a set of candidates.

    Attributes:
        ranked: Every candidate, best first, each scored by its probability. The options of the final question
            lead, most probable first and, among equals, in retrieval order. The candidates dropped after round one
            trail in retrieval order at 0.0: the final question never asked about them.
        probabilities: The final question's distribution over card ids, most probable first. It leaves out the
            reserved option, and round one's answers, which are kept in `exchanges`.
        none_probability: The reserved option's probability in the final question; `None` when there was no
            reserved option, or no question.
        abstained: Whether the decider says that no candidate fits. `ranked` is kept either way.
        key: What any threshold applied to this decision holds for.
        shape: What `key.payload_shape` digests: the number of candidates, the option count of every question asked
            in each round, whether there was a reserved option, and the settings that shape the questions. A
            setting that leaves the questions as they were before it existed is left out, so a key from before
            it still holds: `"min_detail"` only above `NAME`, `"budgets"` (the limits the questions were planned
            under) only when the model declares `max_question_tokens` or `max_option_tokens`, and `"tokenizer"`
            (its class name, not its settings) only when it is not a `HeuristicTokenizer`.
        state_cut: Whether the state was cut to fit the model.
        exchanges: Every call made: round one in chunk order, then the final question. Empty when nothing was asked.
    """

    ranked: tuple[ScoredCard, ...]
    probabilities: Mapping[str, float]
    none_probability: float | None
    abstained: bool
    key: ThresholdKey
    shape: Mapping[str, Any]
    state_cut: bool
    exchanges: tuple[Exchange, ...]

    @property
    def usage(self) -> DecisionUsage:
        """What every call consumed together."""
        return sum((exchange.response.usage for exchange in self.exchanges), DecisionUsage())

    @property
    def seconds(self) -> float:
        """How long the calls took, counted by round: within a round only the slowest counts, then the rounds add up.

        The calls of round one run at the same time, and the final question waits for them. The figure is taken from
        the `seconds` of the responses, not measured around the calls.
        """
        return _critical_path((exchange.round, exchange.response.seconds) for exchange in self.exchanges)

    @property
    def sequential_seconds(self) -> float:
        """How long the calls take against a server that answers one request at a time: every call's `seconds` added.

        0.0 when nothing was asked. Compare `seconds`, which counts the calls of round one as running at the same time.
        """
        return sum((exchange.response.seconds for exchange in self.exchanges), 0.0)

    @property
    def server_seconds(self) -> float | None:
        """`seconds` by the servers' own clocks.

        `None` unless every response reports one; 0.0 when nothing was asked.
        """
        timings: list[tuple[int, float]] = []
        for exchange in self.exchanges:
            if exchange.response.server_seconds is None:
                return None
            timings.append((exchange.round, exchange.response.server_seconds))
        return _critical_path(timings)


class Decider(Protocol):
    """Stage 2: re-rank the candidates a retriever found, and say when none of them fits."""

    async def decide(self, state: str, candidates: Sequence[ScoredCard], /) -> Decision:
        """Rank `candidates` for `state`, the text `build_state` writes.

        `candidates` come in retrieval order, best first; a decider that ties two of them keeps that order.
        """
        ...


class ChoiceDecider:
    """A `Decider` that asks a decision model to choose among the candidates.

    The candidates become the options of one multiple-choice question, and their probabilities re-rank them.
    When they do not fit the model's declared limits (too many options, too many tokens), the decider asks in two
    rounds instead: the candidates are dealt into chunks, ranked by the model at the same time, and the best
    `finalists_per_chunk` of each chunk, fewer when the option cap leaves no room for that many, compete in a
    final question. Only that final distribution is reported as `Decision.probabilities`; the answers of round
    one stay in `Decision.exchanges`. The questions themselves are planned by `plan_rounds`.

    With an `abstention`, the last question also carries a reserved "none of these" option, unless
    `Abstention.reserved_option` is off, and the decision says when the decider abstains. The ranking is reported
    either way. Whatever a threshold is applied to, `Decision.key` says which model, prompt version and payload
    shape it holds for.

    A state over the model's budget is cut in the middle by `fit_state`, and `Decision.state_cut` says so. With no
    candidate, or one and no reserved option, there is nothing to choose between and nothing is asked. Cards are
    described to the model by `max_detail` at most and `min_detail` at least, lower when a limit demands it, and
    counted by `tokenizer` (`HeuristicTokenizer` by default). A question that fits only below `min_detail` is split
    into groups instead. These settings, the model's declared budgets and the tokenizer enter `Decision.shape` only
    when they differ from the defaults, so a key from before they existed still holds. The tokenizer enters by its
    class name alone: a `HeuristicTokenizer` with another `bytes_per_token` plans differently under the default key.

    A model failure is raised as it is, `DecisionError` for a call that failed or answered something that does not
    match the question, with no fallback to retrieval order.
    """

    def __init__(
        self,
        model: DecisionModel,
        *,
        abstention: Abstention | None = None,
        max_detail: DetailLevel = DetailLevel.FULL,
        min_detail: DetailLevel = DetailLevel.NAME,
        finalists_per_chunk: int = 2,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Configure the decider.

        Raises:
            ValueError: `model` does not answer choice questions, `finalists_per_chunk` is below 1, or `min_detail`
                is above `max_detail`.
        """
        if "choice" not in model.question_kinds:
            raise ValueError(
                f"{model.model_id} answers {sorted(model.question_kinds)} questions, "
                "and the decider asks choice questions"
            )
        if finalists_per_chunk < 1:
            raise ValueError(f"finalists_per_chunk must be at least 1, got {finalists_per_chunk}")
        if min_detail > max_detail:
            raise ValueError(f"min_detail={min_detail.name} is above max_detail={max_detail.name}")
        self._model = model
        self._abstention = abstention
        self._reserved = abstention is not None and abstention.reserved_option
        self._max_detail = max_detail
        self._min_detail = min_detail
        self._finalists_per_chunk = finalists_per_chunk
        self._tokenizer: Tokenizer = HeuristicTokenizer() if tokenizer is None else tokenizer

    async def decide(self, state: str, candidates: Sequence[ScoredCard], /) -> Decision:
        """Rank `candidates` for `state`, the text `build_state` writes, and return the `Decision`.

        `candidates` come in retrieval order, best first, and that order breaks every tie.

        Raises:
            ValueError: The candidates do not have distinct card ids, or the state's search queries alone take
                its state budget.
            CandidatesDoNotFit: The candidates do not fit the model's limits, however they are asked. A
                `ValueError`, raised before any call. `card_id` is the id of the card no question can show: its
                name is over `max_text_tokens`, or its option key and framing are over `max_option_tokens`. It is
                `None` when the candidates as a whole do not fit: more than two rounds hold, two of them do not fit
                the token budget together, or not even one finalist from each group fits the final question.
            DecisionError: A call to the model failed or its answer does not match the question. Round one asks
                its chunks at the same time; when several fail, the first in chunk order is raised, and the calls
                of the other chunks are not cancelled.
        """
        cards = [match.card for match in candidates]
        seen: set[str] = set()
        for card in cards:
            if card.id in seen:
                raise ValueError(f"candidates must have distinct card ids, {card.id!r} appears more than once")
            seen.add(card.id)
        if not cards or (len(cards) == 1 and not self._reserved):
            return self._without_a_call(cards)

        limits = self._model.limits
        state, cut = fit_state(state, limits=limits, tokenizer=self._tokenizer)
        plan = plan_rounds(
            state,
            cards,
            reserved=self._reserved,
            limits=limits,
            tokenizer=self._tokenizer,
            max_detail=self._max_detail,
            min_detail=self._min_detail,
            finalists_per_chunk=self._finalists_per_chunk,
        )
        rounds = await self._ask_rounds(state, cards, plan)

        final = rounds[-1][0]
        by_card = final.by_card()
        best_first = _by_probability([card for card in cards if card.id in by_card], by_card)
        eliminated = [card for card in cards if card.id not in by_card]
        none = final.answer.probabilities[NONE_KEY] if self._reserved else None
        return self._decision(
            cards=cards,
            ranked=(
                *(ScoredCard(card, by_card[card.id]) for card in best_first),
                *(ScoredCard(card, 0.0) for card in eliminated),
            ),
            probabilities={card.id: by_card[card.id] for card in best_first},
            none_probability=none,
            abstained=self._abstains(best=max(by_card.values()), none=none),
            state_cut=cut,
            rounds=rounds,
        )

    def _without_a_call(self, cards: Sequence[ToolCard]) -> Decision:
        """Decide when there is nothing to choose between: no candidate, or one and no reserved option."""
        # With one candidate its probability is 1, so only a threshold above that makes it abstain.
        abstained = self._abstains(best=1.0, none=None) if cards else self._abstention is not None
        return self._decision(
            cards=cards,
            ranked=tuple(ScoredCard(card, 1.0) for card in cards),
            probabilities={card.id: 1.0 for card in cards},
            none_probability=None,
            abstained=abstained,
            state_cut=False,
            rounds=[],
        )

    def _abstains(self, *, best: float, none: float | None) -> bool:
        if self._abstention is None:
            return False
        return (none is not None and none >= best) or best < self._abstention.threshold

    async def _ask_rounds(self, state: str, cards: Sequence[ToolCard], plan: RoundPlan) -> list[list[_Asked]]:
        """Ask what `plan` says, and return the questions asked in each round, the final one last."""
        if plan.single is not None:
            return [[await self._ask(state, plan.single, round_number=1)]]
        first = await self._ask_chunks(state, plan)
        kept: set[str] = set()
        for chunk, asked in zip(plan.chunks, first, strict=True):
            ordered = chunk.cards if asked is None else _by_probability(chunk.cards, asked.by_card())
            kept.update(card.id for card in ordered[: plan.finalists_per_chunk])
        finalists = [card for card in cards if card.id in kept]
        final = plan_question(
            state,
            finalists,
            reserved=self._reserved,
            limits=self._model.limits,
            tokenizer=self._tokenizer,
            max_detail=self._max_detail,
            min_detail=self._min_detail,
        )
        if final is None:
            # `plan_rounds` checked a worst-case final before the first call, so this should not happen: it is a
            # guard in case the final built from the finalists costs more than that worst case.
            raise CandidatesDoNotFit(
                f"the final question over the {len(finalists)} finalists of the first round does not fit the limits "
                f"of {self._model.model_id}, though the plan expected it to"
            )
        return [[asked for asked in first if asked is not None], [await self._ask(state, final, round_number=2)]]

    async def _ask_chunks(self, state: str, plan: RoundPlan) -> list[_Asked | None]:
        """Ask round one's questions at the same time: one entry per chunk, `None` for a chunk of one card."""
        answered: dict[int, _Asked] = {}
        failed: dict[int, Exception] = {}

        async def ask(index: int, planned: PlannedQuestion) -> None:
            try:
                answered[index] = await self._ask(state, planned, round_number=1)
            except Exception as error:
                # A task group wraps even a lone failure in an ExceptionGroup. Keeping each outcome lets the
                # first failure in chunk order be raised as itself, whatever the timing.
                failed[index] = error

        async with anyio.create_task_group() as group:
            for index, chunk in enumerate(plan.chunks):
                if chunk.question is not None:
                    group.start_soon(ask, index, chunk.question)
        if failed:
            raise failed[min(failed)]
        return [answered.get(index) for index in range(len(plan.chunks))]

    async def _ask(self, state: str, planned: PlannedQuestion, *, round_number: int) -> _Asked:
        request = DecisionRequest(state=state, questions={_QUESTION_ID: planned.question})
        response = await self._model.ask(request)
        # The adapters check their answers, but a model is any object with `ask`: check that this one answered.
        answer = response.answers.get(_QUESTION_ID)
        if not isinstance(answer, ChoiceAnswer):
            found = "no answer" if answer is None else f"a {type(answer).__name__}"
            raise DecisionError(
                f"{self._model.model_id}: expected a choice answer to question {_QUESTION_ID!r}, "
                f"the response has {found}"
            )
        if set(answer.probabilities) != set(planned.question.options):
            raise DecisionError(
                f"{self._model.model_id}: expected probabilities for {list(planned.question.options)}, "
                f"got {list(answer.probabilities)}"
            )
        exchange = Exchange(
            round=round_number,
            request=request,
            response=response,
            detail=planned.detail,
            estimated_input_tokens=planned.estimated_input_tokens,
            option_card_ids=dict(planned.card_ids),
        )
        return _Asked(planned=planned, exchange=exchange, answer=answer)

    def _decision(
        self,
        *,
        cards: Sequence[ToolCard],
        ranked: tuple[ScoredCard, ...],
        probabilities: Mapping[str, float],
        none_probability: float | None,
        abstained: bool,
        state_cut: bool,
        rounds: list[list[_Asked]],
    ) -> Decision:
        shape: dict[str, Any] = {
            "candidates": len(cards),
            "questions": [
                [len(asked.planned.question.options) for asked in asked_in_round] for asked_in_round in rounds
            ],
            "reserved_option": self._reserved,
            "max_detail": self._max_detail.name,
            "finalists_per_chunk": self._finalists_per_chunk,
            "questions_per_request": 1,
            "option_keys": _OPTION_KEYS,
        }
        # Each field below enters only when it differs from what the decider did before it existed, so an existing
        # key does not move.
        if self._min_detail > DetailLevel.NAME:
            shape["min_detail"] = self._min_detail.name
        limits = self._model.limits
        if limits.max_question_tokens is not None or limits.max_option_tokens is not None:
            shape["budgets"] = {
                "max_options_per_choice": limits.max_options_per_choice,
                "max_request_tokens": limits.max_request_tokens,
                "max_state_plus_question_tokens": limits.max_state_plus_question_tokens,
                "max_question_tokens": limits.max_question_tokens,
                "max_option_tokens": limits.max_option_tokens,
                "max_text_tokens": limits.max_text_tokens,
            }
        if type(self._tokenizer) is not HeuristicTokenizer:
            shape["tokenizer"] = type(self._tokenizer).__qualname__
        canonical = json.dumps(shape, sort_keys=True, separators=(",", ":"))
        own_version = self._model.prompt_version
        return Decision(
            ranked=ranked,
            probabilities=probabilities,
            none_probability=none_probability,
            abstained=abstained,
            key=ThresholdKey(
                model=self._model.model_id,
                prompt_version=PROMPT_VERSION if own_version is None else f"{PROMPT_VERSION}+{own_version}",
                payload_shape=hashlib.sha256(canonical.encode()).hexdigest()[:16],
                question_kind="choice",
            ),
            shape=shape,
            state_cut=state_cut,
            exchanges=tuple(asked.exchange for asked_in_round in rounds for asked in asked_in_round),
        )


@dataclass(frozen=True, slots=True)
class _Asked:
    """A planned question, the exchange it produced and the answer it got."""

    planned: PlannedQuestion
    exchange: Exchange
    answer: ChoiceAnswer

    def by_card(self) -> dict[str, float]:
        """The answer's probability for each card id; the reserved option is left out."""
        card_ids = self.planned.card_ids
        return {card_ids[key]: probability for key, probability in self.answer.probabilities.items() if key in card_ids}


def _by_probability(cards: Sequence[ToolCard], probabilities: Mapping[str, float]) -> list[ToolCard]:
    """Best first. The sort is stable, so cards of equal probability keep the order they were given in."""
    return sorted(cards, key=lambda card: probabilities[card.id], reverse=True)


def _critical_path(timings: Iterable[tuple[int, float]]) -> float:
    """The slowest call of each round, summed: the calls of one round run at the same time."""
    slowest: dict[int, float] = {}
    for round_number, seconds in timings:
        slowest[round_number] = max(seconds, slowest.get(round_number, 0.0))
    return sum(slowest.values())
