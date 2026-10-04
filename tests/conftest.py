import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import pytest
from dotenv import load_dotenv

from toolhunch import DetailLevel, ScoredCard
from toolhunch.decision import (
    Answer,
    BinaryQuestion,
    ChoiceQuestion,
    Decision,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    Exchange,
    ModelLimits,
    QuestionKind,
    ThresholdKey,
    check_request,
    choice_answer,
)
from toolhunch.decision.planner import NONE_KEY, NONE_TEXT

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

_CHOICE_ONLY: frozenset[QuestionKind] = frozenset({"choice"})


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeModel:
    """A decision model for tests: it answers choice questions in proportion to a weight per tool.

    A tool's weight is looked up under its option key (`"search #2"` for a repeated name), then under its name,
    which is the option text up to the first `":"` (every detail level starts with it), and is 1 when neither is
    given. The reserved option weighs `none_weight`. Like the real adapters it checks each request against its
    `limits` before answering, and it reports 0.1 s of wall time and 0.01 s of server time per call.

    Two hooks let a test misbehave on purpose: `before` is awaited with each request before it is answered, so it
    can wait or raise; `answers` replaces the weighted answers of every response.

    Attributes:
        requests: The requests it was asked, in order, each after passing the limits check.
    """

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        none_weight: float = 1.0,
        limits: ModelLimits | None = None,
        kinds: frozenset[QuestionKind] = _CHOICE_ONLY,
        prompt_version: str | None = None,
        before: Callable[[DecisionRequest], Awaitable[None]] | None = None,
        answers: Mapping[str, Answer] | None = None,
    ) -> None:
        self.model_id = "fake@test"
        self.limits = ModelLimits(source="test", checked=date(2026, 9, 28)) if limits is None else limits
        self.question_kinds = kinds
        self.prompt_version = prompt_version
        self.requests: list[DecisionRequest] = []
        self._weights = dict(weights or {})
        self._none_weight = none_weight
        self._before = before
        self._answers = answers

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        self.requests.append(request)
        if self._before is not None:
            await self._before(request)
        return DecisionResponse(
            answers=self._weighted_answers(request) if self._answers is None else self._answers,
            usage=DecisionUsage(1, 10, 1),
            seconds=0.1,
            server_seconds=0.01,
            raw={},
        )

    def _weighted_answers(self, request: DecisionRequest) -> dict[str, Answer]:
        answers: dict[str, Answer] = {}
        for key, question in request.questions.items():
            if not isinstance(question, ChoiceQuestion):
                raise NotImplementedError(f"FakeModel answers choice questions only, not {question.kind}")
            weights = {option: self._weight(option, text) for option, text in question.options.items()}
            answers[key] = choice_answer(weights, keys=list(question.options), model_id=self.model_id)
        return answers

    def _weight(self, key: str, text: str) -> float:
        if (key, text) == (NONE_KEY, NONE_TEXT):
            return self._none_weight
        return self._weights.get(key, self._weights.get(text.split(":", 1)[0], 1.0))


@pytest.fixture
def fake_model() -> type[FakeModel]:
    """The `FakeModel` class; under importlib import mode a test module cannot import it from here."""
    return FakeModel


class RecordingDecider:
    """A `Decider` for tests: it records what it is asked and answers from a script.

    It returns the candidates in the order of `order`; ids the script does not name follow, in the order it got them.
    A candidate's score falls with its rank. `abstain` makes it abstain, and `error` is raised instead of answering.
    The decision holds one scripted exchange that carries `usage`, because `Decision.usage` is the sum over the
    exchanges. `delay` is how long it takes to answer, in seconds.

    Attributes:
        states: The states it was asked about, in order.
        candidates: The card ids it was given with each state, in the order it got them.
    """

    def __init__(
        self,
        *,
        order: Sequence[str] = (),
        usage: DecisionUsage | None = None,
        abstain: bool = False,
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.states: list[str] = []
        self.candidates: list[list[str]] = []
        self._rank = {card_id: rank for rank, card_id in enumerate(order)}
        self._usage = DecisionUsage() if usage is None else usage
        self._abstain = abstain
        self._error = error
        self._delay = delay

    async def decide(self, state: str, candidates: Sequence[ScoredCard], /) -> Decision:
        self.states.append(state)
        self.candidates.append([match.card.id for match in candidates])
        if self._delay:
            await anyio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        cards = sorted((match.card for match in candidates), key=lambda card: self._rank.get(card.id, len(self._rank)))
        weights = range(len(cards), 0, -1)
        total = sum(weights)
        probabilities = {card.id: weight / total for card, weight in zip(cards, weights, strict=True)}
        exchange = Exchange(
            round=1,
            request=DecisionRequest(state=state, questions={"tool": BinaryQuestion(instructions="scripted")}),
            response=DecisionResponse(answers={}, usage=self._usage, seconds=0.0, raw={}),
            detail=DetailLevel.FULL,
            estimated_input_tokens=0,
        )
        return Decision(
            ranked=tuple(ScoredCard(card, probabilities[card.id]) for card in cards),
            probabilities=probabilities,
            none_probability=None,
            abstained=self._abstain,
            key=ThresholdKey("recording@test", "scripted", "0" * 16, "choice"),
            shape={},
            state_cut=False,
            exchanges=(exchange,),
        )


@pytest.fixture
def recording_decider() -> type[RecordingDecider]:
    """The `RecordingDecider` class; under importlib import mode a test module cannot import it from here."""
    return RecordingDecider


def _env(name: str) -> str | None:
    """A variable from the environment or the repository's .env."""
    load_dotenv(Path(__file__).parents[1] / ".env", override=False)
    return os.environ.get(name)


@pytest.fixture
def openai_api_key() -> str:
    """The OpenAI key from the environment or the repository's .env; skips the test when absent."""
    if not (key := _env("OPENAI_API_KEY")):
        pytest.skip("OPENAI_API_KEY is not set")
    return key


@pytest.fixture
def typesafe_api_key() -> str:
    """The TypeSafe key from the environment or the repository's .env; skips the test when absent."""
    if not (key := _env("TYPESAFE_API_KEY")):
        pytest.skip("TYPESAFE_API_KEY is not set")
    return key


@pytest.fixture
def cloudflare_account() -> str:
    """Cloudflare's account ID and API token, from the environment or .env; skips the test unless both are set."""
    account, key = _env("CLOUDFLARE_ACCOUNT_ID"), _env("CLOUDFLARE_API_KEY")
    if not account or not key:
        pytest.skip("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_KEY are not both set")
    return account


@pytest.fixture
def strands_base_url() -> str:
    """The root of a local Strands Decider server (`STRANDS_BASE_URL`); skips the test when it is not set."""
    if not (base_url := _env("STRANDS_BASE_URL")):
        pytest.skip("STRANDS_BASE_URL is not set")
    return base_url


@pytest.fixture
def clm_base_url() -> str:
    """The CLM server root from the environment or the repository's .env; skips the test unless the key is set too."""
    base_url, key = _env("CLM_BASE_URL"), _env("CLM_API_KEY")
    if not base_url or not key:
        pytest.skip("CLM_BASE_URL and CLM_API_KEY are not both set")
    return base_url
