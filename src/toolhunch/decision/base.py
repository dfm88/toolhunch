"""The canonical types every decision model speaks: questions, answers, limits and the protocol."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date  # noqa: TC003  # pydantic evaluates the annotations of ModelLimits at runtime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "Answer",
    "BinaryAnswer",
    "BinaryQuestion",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "DecisionError",
    "DecisionModel",
    "DecisionRequest",
    "DecisionResponse",
    "DecisionUsage",
    "ModelLimits",
    "Question",
    "QuestionKind",
    "ScoreAnswer",
    "ScoreQuestion",
    "check_request",
    "choice_answer",
]

type QuestionKind = Literal["choice", "binary", "score"]
"""What a question asks for: one of several options, a yes/no, or a level on an ordinal scale."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceQuestion:
    """Pick one of several named options.

    Attributes:
        instructions: What is being asked.
        options: Option key to option text, in presentation order. At least two.
    """

    kind: ClassVar[QuestionKind] = "choice"

    instructions: str
    options: Mapping[str, str]

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(f"a choice question needs at least two options, got {len(self.options)}")


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryQuestion:
    """Ask whether something holds; the answer is a probability of true (Jev's `noul`).

    Attributes:
        instructions: The statement to judge.
    """

    kind: ClassVar[QuestionKind] = "binary"

    instructions: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreQuestion:
    """Place something on an ordinal scale.

    Attributes:
        instructions: What is being scored.
        levels: Level descriptions, lowest first. At least two.
    """

    kind: ClassVar[QuestionKind] = "score"

    instructions: str
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"a score question needs at least two levels, got {len(self.levels)}")


type Question = ChoiceQuestion | BinaryQuestion | ScoreQuestion
"""Any of the three question types; each carries its `kind`."""


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionRequest:
    """One call to a decision model: a state and the named questions asked about it.

    Attributes:
        state: The context every question is asked about.
        questions: Question id to question. At least one.
    """

    state: str
    questions: Mapping[str, Question]

    def __post_init__(self) -> None:
        if not self.questions:
            raise ValueError("a decision request needs at least one question")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceAnswer:
    """The answer to a `ChoiceQuestion`.

    Build it with `choice_answer`, which checks the keys and renormalises.

    Attributes:
        probabilities: Option key to probability, one per option in option order, summing to 1.
    """

    probabilities: Mapping[str, float]

    @property
    def choice(self) -> str:
        """The most probable option; a tie goes to the earlier option."""
        return max(self.probabilities, key=lambda key: self.probabilities[key])


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryAnswer:
    """The answer to a `BinaryQuestion`.

    Attributes:
        probability: The probability that the statement holds.
    """

    probability: float


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreAnswer:
    """The answer to a `ScoreQuestion`.

    Attributes:
        expected: The expected level index; 0 is the first (lowest) level.
        probabilities: One probability per level, when the server returns them.
    """

    expected: float
    probabilities: tuple[float, ...] | None = None


type Answer = ChoiceAnswer | BinaryAnswer | ScoreAnswer
"""Any of the three answer types."""


@dataclass(frozen=True, slots=True)
class DecisionUsage:
    """Paid resources one or more decision-model calls consumed.

    Positional, like `RetrievalUsage`.

    Attributes:
        requests: Model calls made.
        input_tokens: Input tokens, as the server reports them.
        output_tokens: Output tokens, as the server reports them.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: DecisionUsage) -> DecisionUsage:
        return DecisionUsage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecisionResponse:
    """What a decision model returned for one request.

    Attributes:
        answers: Question id to answer, one per question, each of its question's kind.
        usage: What the call consumed.
        seconds: Client wall time, retries included.
        raw: The decoded response body, kept for provenance. Fields the canonical types do not model
            (a server's `confidence`, its `model`, a `legend`, `billing_units`) are found only here.
        server_seconds: The server's own latency, when it reports one.
    """

    answers: Mapping[str, Answer]
    usage: DecisionUsage
    seconds: float
    raw: Mapping[str, Any]
    server_seconds: float | None = None


class DecisionError(Exception):
    """A decision model failed or returned something unusable.

    The pipeline propagates it: there is no silent fallback to retrieval order. Every message starts
    with the model id.

    Attributes:
        status: The HTTP status of the reply that ended the call, when a server refused it after any
            retries; `None` for every other failure, an unusable answer or no reply at all.
    """

    def __init__(self, *args: object, status: int | None = None) -> None:
        super().__init__(*args)
        self.status = status


class ModelLimits(BaseModel):
    """What a decision model accepts and costs, as declared data with its source and date.

    Vendors change limits, so each set carries where the numbers come from and when they were
    checked. Every limit is optional; `None` means not declared, not unlimited. Override a declared
    set for one model with `model_copy(update=...)`.

    Attributes:
        max_options_per_choice: Options in one choice question, the reserved option included.
        max_request_tokens: Tokens in the whole request.
        max_state_plus_question_tokens: Tokens in the state plus the longest question.
        max_text_tokens: Tokens in each text the model encodes on its own: every option, and the
            state together with the question's instructions.
        max_questions_per_request: Questions in one request.
        score_levels: The fewest and the most levels a score question may have.
        price_input_per_mtok: USD per million input tokens.
        price_output_per_mtok: USD per million output tokens.
        source: Where the numbers come from.
        checked: The date they were last checked.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_options_per_choice: int | None = Field(default=None, ge=2)
    max_request_tokens: int | None = Field(default=None, ge=1)
    max_state_plus_question_tokens: int | None = Field(default=None, ge=1)
    max_text_tokens: int | None = Field(default=None, ge=1)
    max_questions_per_request: int | None = Field(default=None, ge=1)
    score_levels: tuple[int, int] | None = None
    price_input_per_mtok: float | None = Field(default=None, ge=0)
    price_output_per_mtok: float | None = Field(default=None, ge=0)
    source: str
    checked: date

    def estimate_usd(self, usage: DecisionUsage) -> float | None:
        """Price `usage` in USD; `None` when no input price is declared.

        Output tokens are priced at zero when only the input price is declared.
        """
        if self.price_input_per_mtok is None:
            return None
        output_price = self.price_output_per_mtok or 0.0
        return (usage.input_tokens * self.price_input_per_mtok + usage.output_tokens * output_price) / 1_000_000


class DecisionModel(Protocol):
    """A model that answers canonical questions: a Jev-style endpoint, an LLM's logprobs, a local encoder."""

    @property
    def model_id(self) -> str:
        """`"<model>@<host>"`, taken from configuration and never from a response."""
        ...

    @property
    def limits(self) -> ModelLimits:
        """What the model accepts and costs."""
        ...

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The kinds of question the model answers."""
        ...

    @property
    def prompt_version(self) -> str | None:
        """The version of the adapter's own prompt template, if it has one."""
        ...

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Answer every question in `request`.

        `**options` (and an instance-level `extra_body`) are merged into the payload's top level: the
        merge is shallow and the last write wins.

        Raises:
            DecisionError: The call failed or returned something unusable.
        """
        ...


def check_request(request: DecisionRequest, *, limits: ModelLimits, kinds: frozenset[QuestionKind]) -> None:
    """Check `request` against a model's declared `kinds` and `limits` before anything is sent.

    Checks the kinds, the option cap, the questions per request and the score levels. Token caps are
    the planner's job, since only an estimate exists before sending.

    Raises:
        ValueError: The request does not fit the model.
    """
    if limits.max_questions_per_request is not None and len(request.questions) > limits.max_questions_per_request:
        raise ValueError(
            f"the request has {len(request.questions)} questions, the model takes at most "
            f"{limits.max_questions_per_request}"
        )
    for key, question in request.questions.items():
        if question.kind not in kinds:
            raise ValueError(f"question {key!r} is a {question.kind} question, the model answers {sorted(kinds)}")
        if isinstance(question, ChoiceQuestion):
            if limits.max_options_per_choice is not None and len(question.options) > limits.max_options_per_choice:
                raise ValueError(
                    f"question {key!r} has {len(question.options)} options, the model takes at most "
                    f"{limits.max_options_per_choice}"
                )
        elif isinstance(question, ScoreQuestion) and limits.score_levels is not None:
            fewest, most = limits.score_levels
            if not fewest <= len(question.levels) <= most:
                raise ValueError(
                    f"question {key!r} has {len(question.levels)} levels, the model takes {fewest} to {most}"
                )


def choice_answer(probabilities: Mapping[str, object], *, keys: Sequence[str], model_id: str) -> ChoiceAnswer:
    """Turn a model's raw distribution into a `ChoiceAnswer` over `keys`.

    The result follows the order of `keys` and is renormalised to sum to 1: models round their
    probabilities, so a distribution may sum to 0.99. `int` and `float` values are accepted, `bool` is not.

    Raises:
        DecisionError: `probabilities` does not cover exactly `keys`, a value is not a finite,
            non-negative number, or the values do not sum to a positive finite number.
    """
    if set(probabilities) != set(keys):
        raise DecisionError(f"{model_id}: expected probabilities for {list(keys)}, got {list(probabilities)}")
    values: dict[str, float] = {}
    for key in keys:
        value = probabilities[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise DecisionError(f"{model_id}: the probability of {key!r} is not a number: {value!r}")
        # One exact comparison rejects NaN, infinities, negatives and ints too large for a float.
        if not 0 <= value <= sys.float_info.max:
            raise DecisionError(f"{model_id}: the probability of {key!r} is not finite and non-negative: {value!r}")
        values[key] = float(value)
    total = sum(values.values())
    if not 0 < total <= sys.float_info.max:
        raise DecisionError(f"{model_id}: the probabilities do not sum to a positive finite number: {total!r}")
    return ChoiceAnswer(probabilities={key: values[key] / total for key in keys})
