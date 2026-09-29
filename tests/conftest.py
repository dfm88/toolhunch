import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv

from toolhunch.decision import (
    Answer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    QuestionKind,
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
def clm_base_url() -> str:
    """The CLM server root from the environment or the repository's .env; skips the test unless the key is set too."""
    base_url, key = _env("CLM_BASE_URL"), _env("CLM_API_KEY")
    if not base_url or not key:
        pytest.skip("CLM_BASE_URL and CLM_API_KEY are not both set")
    return base_url
