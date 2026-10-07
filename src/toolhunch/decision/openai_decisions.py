"""A decision model behind OpenAI's Decisions API, `POST /v1/decisions`: typed questions, full distributions."""

from __future__ import annotations

import contextlib
import math
import sys
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, ValidationError

from toolhunch.decision._http import Endpoint, JsonPoster
from toolhunch.decision.base import (
    BinaryAnswer,
    BinaryQuestion,
    ChoiceQuestion,
    DecisionError,
    DecisionRefused,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    ScoreAnswer,
    ScoreQuestion,
    check_request,
    choice_answer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx2

    from toolhunch.decision._http import JsonReply
    from toolhunch.decision.base import Answer, DecisionRequest, Question, QuestionKind

__all__ = ["OPENAI_DECISIONS_LIMITS", "PROMPT_VERSION", "OpenAIDecisionModel"]

OPENAI_DECISIONS_LIMITS = ModelLimits(
    max_options_per_choice=255,
    max_questions_per_request=64,
    price_input_per_mtok=0.10,
    price_output_per_mtok=0.0,
    source=(
        "developers.openai.com/api/docs/guides/decisions: $0.10 per M input tokens, no output charge, no caps "
        "stated; openai-python v3.26.0 types; probe 2026-10-07: a 255-option choice and 64 questions in one "
        "request accepted, more not tried; a choice needs 2 options"
    ),
    checked=date(2026, 10, 7),
)
"""The Decisions API's price and the caps the probe found; it declares no token window, since OpenAI states none."""

PROMPT_VERSION = "decisions-v1"
"""The version of the mapping from canonical questions to the Decisions payload.

The mapping shapes what the model sees (a key-only option has no `description`, a score level is its `label`), so
a threshold holds for one version of it; it also keeps this adapter's keys apart from other adapters that ask the
same model.
"""

_KINDS: frozenset[QuestionKind] = frozenset({"choice", "binary", "score"})
_WIRE_TYPES = {"choice": "choice", "binary": "predicate", "score": "score"}
_LATENCY_HEADER = "openai-processing-ms"


class _Usage(BaseModel):
    input_tokens: int = Field(default=0, strict=True, ge=0)
    output_tokens: int = Field(default=0, strict=True, ge=0)


class _Reply(BaseModel):
    answers: list[dict[str, Any]]
    usage: _Usage = Field(default_factory=_Usage)


class OpenAIDecisionModel:
    """A decision model behind OpenAI's Decisions API, `POST {base_url}/decisions`.

    It answers choice, binary and score questions, several in one request. The canonical request maps onto the
    API's own shape (version `PROMPT_VERSION`): the state is the `input` string; each question keeps its id as
    `name`; a choice question's options become `choices` of `{value: key, description: text}` in option order, an
    empty text leaving `description` out; a binary question becomes a `predicate`; a score question's levels become
    `levels` of `{label: text}`, lowest first.

    Decoding checks the reply against the request: one answer per question, in question order, with its question's
    name when the reply gives one and its question's type. A choice answer's `probabilities` must cover exactly the
    option keys, as strings, each once; it is renormalised and follows the order of the options. The API rounds
    its probabilities to two decimals, so two options can tie, and a tie goes to the earlier option
    (`ChoiceAnswer.choice`), whatever the reply's own `choice` says. A predicate gives `BinaryAnswer.probability`;
    a score gives `ScoreAnswer`, its per-level probabilities ordered by level. A question the API refused raises
    `DecisionRefused`, which names every refused question and carries the reply's usage, since a refused reply may
    still be billed. The reply's `model`, `confidence` and the rest stay in `DecisionResponse.raw`; the
    `openai-processing-ms` header becomes `server_seconds`.

    The identity `model_id` is `"<model>@<host>"`, taken from `model` and `base_url` (`ValueError` unless it is an
    absolute http(s) URL without credentials). `limits` is declared data: a request beyond it raises `ValueError`
    before anything is sent. `extra_body` and the `**options` of `ask` are merged into the top level of the payload,
    shallowly and last write wins, so either can set `safety_identifier`; the identity does not follow.

    The key is `api_key` or, read at call time, the `api_key_env` variable. It never appears in `repr` or in an error
    message. A transport error, a 429 or a transient server error is retried up to `max_retries` times with backoff;
    any other failure raises `DecisionError`. `http_client` is used as is and never closed by the model; `timeout`
    applies only to a client the model creates on its first call, which `aclose()` closes.
    """

    def __init__(
        self,
        model: str = "gpt-6-luna",
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        limits: ModelLimits = OPENAI_DECISIONS_LIMITS,
        extra_body: Mapping[str, Any] | None = None,
        http_client: httpx2.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Configure the endpoint; `base_url` is the API root, such as `https://api.openai.com/v1`."""
        endpoint = Endpoint.parse(base_url, model=model)
        self._model = model
        self._base_url = endpoint.base_url
        self._model_id = endpoint.model_id
        self._limits = limits
        self._extra_body = dict(extra_body or {})
        self._poster = JsonPoster(
            f"{self._base_url}/decisions",
            model_id=self._model_id,
            api_key=api_key,
            api_key_env=api_key_env,
            http_client=http_client,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def model_id(self) -> str:
        """`"<model>@<host>"`, from the configuration and never from a response."""
        return self._model_id

    @property
    def limits(self) -> ModelLimits:
        """What the model accepts and costs."""
        return self._limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """All three kinds: choice, binary and score."""
        return _KINDS

    @property
    def prompt_version(self) -> str:
        """`PROMPT_VERSION`: the version of the payload mapping."""
        return PROMPT_VERSION

    def __repr__(self) -> str:
        return f"OpenAIDecisionModel(model_id={self._model_id!r})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask every question of `request` in one call.

        `**options` are merged into the payload after the instance's `extra_body`, shallowly, so the last write wins.

        Raises:
            ValueError: `request` does not fit the model's kinds or declared limits. Nothing is sent.
            DecisionRefused: The API refused one or more questions.
            DecisionError: The call failed, or the reply does not answer `request`.
        """
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        payload: dict[str, Any] = {
            "model": self._model,
            "input": request.state,
            "questions": [self._question_body(key, question) for key, question in request.questions.items()],
        }
        payload |= self._extra_body
        payload |= options
        return self._decode(request, await self._poster.post(payload))

    async def aclose(self) -> None:
        """Close the HTTP client if the model created it; an injected client stays open."""
        await self._poster.aclose()

    @staticmethod
    def _question_body(key: str, question: Question) -> dict[str, Any]:
        body: dict[str, Any] = {"type": _WIRE_TYPES[question.kind], "name": key, "instructions": question.instructions}
        if isinstance(question, ChoiceQuestion):
            body["choices"] = [
                {"value": option, "description": text} if text else {"value": option}
                for option, text in question.options.items()
            ]
        elif isinstance(question, ScoreQuestion):
            body["levels"] = [{"label": level} for level in question.levels]
        return body

    def _decode(self, request: DecisionRequest, reply: JsonReply) -> DecisionResponse:
        try:
            parsed = _Reply.model_validate(reply.body)
        except ValidationError as error:
            problem = error.errors(include_url=False, include_context=False, include_input=False)[0]
            where = ".".join(str(part) for part in problem["loc"])
            raise self._error(f"unexpected response at {where}: {problem['msg']}") from None
        usage = DecisionUsage(
            requests=1, input_tokens=parsed.usage.input_tokens, output_tokens=parsed.usage.output_tokens
        )
        if len(parsed.answers) != len(request.questions):
            raise self._error(f"{len(request.questions)} questions asked, {len(parsed.answers)} answers in the reply")
        answers: dict[str, Answer] = {}
        refused: list[str] = []
        for (key, question), wire in zip(request.questions.items(), parsed.answers, strict=True):
            if (name := wire.get("name")) is not None and name != key:
                raise self._error(f"the answer in the place of question {key!r} is named {name!r}")
            if wire.get("type") == "refusal":
                refused.append(key)
                continue
            answers[key] = self._answer(key, question, wire)
        if refused:
            raise DecisionRefused(
                f"{self._model_id}: the API refused {len(refused)} question(s): {refused}", names=refused, usage=usage
            )
        return DecisionResponse(
            answers=answers,
            usage=usage,
            seconds=reply.seconds,
            server_seconds=self._server_seconds(reply.headers),
            raw=reply.body,
        )

    def _answer(self, key: str, question: Question, wire: Mapping[str, Any]) -> Answer:
        expected = _WIRE_TYPES[question.kind]
        if wire.get("type") != expected:
            raise self._error(
                f"question {key!r} needs an answer of type {expected!r}, the response has type {wire.get('type')!r}"
            )
        if isinstance(question, ChoiceQuestion):
            return choice_answer(
                self._distribution(key, wire, kind=str), keys=list(question.options), model_id=self._model_id
            )
        if isinstance(question, BinaryQuestion):
            if (probability := self._number(wire.get("probability"))) is None or not 0 <= probability <= 1:
                raise self._error(f"question {key!r}: `probability` is not a number in [0, 1]")
            return BinaryAnswer(probability=probability)
        if (score := self._number(wire.get("score"))) is None:
            raise self._error(f"question {key!r}: `score` is not a finite number")
        levels = choice_answer(
            self._distribution(key, wire, kind=int),
            keys=[str(level) for level in range(len(question.levels))],
            model_id=self._model_id,
        )
        return ScoreAnswer(expected=score, probabilities=tuple(levels.probabilities.values()))

    def _distribution(self, key: str, wire: Mapping[str, Any], *, kind: type[str] | type[int]) -> dict[str, object]:
        """`probabilities: [{value, probability}]` as a mapping; each value of type `kind`, each value once."""
        entries = wire.get("probabilities")
        if not isinstance(entries, list):
            raise self._error(f"question {key!r}: `probabilities` is not a list")
        distribution: dict[str, object] = {}
        for item in cast("list[object]", entries):
            entry = cast("dict[str, object]", item) if isinstance(item, dict) else {}
            value = entry.get("value")
            # A choice value is typed: a boolean `true` is not the option "true". A bool is an int to Python, too.
            if isinstance(value, bool) or not isinstance(value, kind):
                raise self._error(f"question {key!r}: a probability's value is not a {kind.__name__}: {value!r}")
            if str(value) in distribution:
                raise self._error(f"question {key!r}: the value {value!r} is given twice")
            distribution[str(value)] = entry.get("probability")
        return distribution

    @staticmethod
    def _server_seconds(headers: Mapping[str, str]) -> float | None:
        """The server's own latency in seconds; `None` when the header is absent or unusable."""
        with contextlib.suppress(ValueError):  # absent ("") or not a number
            if 0 <= (milliseconds := float(headers.get(_LATENCY_HEADER, ""))) < math.inf:
                return milliseconds / 1000
        return None

    def _error(self, message: str) -> DecisionError:
        return DecisionError(f"{self._model_id}: {message}")

    @staticmethod
    def _number(value: object) -> float | None:
        """`value` as a float when it is a finite JSON number; `None` for anything else, a `bool` included."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value) if -sys.float_info.max <= value <= sys.float_info.max else None
