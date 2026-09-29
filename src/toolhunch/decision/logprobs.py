"""A decision model that asks an LLM for one option letter and reads the logprobs: the baseline decider."""

from __future__ import annotations

import json
import math
import string
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field, ValidationError

from toolhunch.decision._http import Endpoint, JsonPoster
from toolhunch.decision.base import (
    ChoiceQuestion,
    DecisionError,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    check_request,
    choice_answer,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx2

    from toolhunch.decision._http import JsonReply
    from toolhunch.decision.base import DecisionRequest, QuestionKind

__all__ = ["LOGPROB_LIMITS", "PROMPT_VERSION", "SYSTEM_PROMPT", "OpenAILogprobModel"]

LOGPROB_LIMITS = ModelLimits(
    max_options_per_choice=20,
    max_questions_per_request=1,
    source=(
        "OpenAI API reference: top_logprobs 0-20; probe 2026-09-28: gpt-4.1-mini returns 20, "
        "gpt-5.4-mini at most 5, gpt-5-mini none"
    ),
    checked=date(2026, 9, 28),
)
"""One question per call, of at most 20 options: the letters must fit among the top logprobs the API returns."""

PROMPT_VERSION = "letters-v1"
"""The version of `SYSTEM_PROMPT` and the user message together.

A threshold holds for one model, one prompt version and one payload shape, so change either prompt only
together with this constant.
"""

SYSTEM_PROMPT = (
    "You answer multiple-choice questions about a state. Reply with the letter of the best option and nothing else."
)
"""The fixed system message."""

_USER_PROMPT = (
    "State (a JSON string):\n{state}\n\n"
    "Question: {instructions}\n\n"
    "Options (each a JSON string):\n{options}\n\n"
    "Answer with one letter."
)
_KINDS: frozenset[QuestionKind] = frozenset({"choice"})
_LETTERS = string.ascii_uppercase
_TOP_LOGPROBS = 20  # the most the API allows: variants of one letter ("A", " A") each use up a slot
_SHOWN_CHARACTERS = 100  # of what the model said instead of a letter, in an error message


class _TopLogprob(BaseModel):
    token: str
    logprob: float = Field(strict=True, allow_inf_nan=False)  # NaN and infinities are not JSON


class _TokenLogprobs(BaseModel):
    top_logprobs: list[_TopLogprob] = Field(min_length=1)


class _Logprobs(BaseModel):
    content: list[_TokenLogprobs] = Field(min_length=1)


class _Choice(BaseModel):
    logprobs: _Logprobs


class _Usage(BaseModel):
    prompt_tokens: int = Field(default=0, strict=True, ge=0)
    completion_tokens: int = Field(default=0, strict=True, ge=0)


class _Completion(BaseModel):
    choices: list[_Choice] = Field(min_length=1)
    usage: _Usage = Field(default_factory=_Usage)


def _encode(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


class OpenAILogprobModel:
    """A decision model that asks an LLM for one option letter and reads the logprobs: the baseline decider.

    A choice question becomes a multiple-choice prompt whose options are lettered A, B, … in option
    order. The model posts it to `{base_url}/chat/completions` and asks for one token (`max_tokens=1`,
    `temperature=0`, `logprobs=true`, `top_logprobs=20`); the probabilities of the letters among that
    token's top logprobs are the answer. It answers choice questions only, one question per request.

    The prompt is fixed copy, version `PROMPT_VERSION`. The state, the instructions and every option
    text are JSON-encoded, so text taken from a tool card stays inside one JSON string: it cannot open a
    new line or fake another option.

    Decoding reads the first generated token's `top_logprobs`. A token counts for a letter when its
    text, stripped of whitespace, equals that letter, so `"B"` and `" B"` both count for B. The variants
    of one letter add up, a letter missing from the list gets 0, and the distribution is renormalised
    over the letters. A response without top logprobs, or with no option letter among them, raises
    `DecisionError`. Anything else the endpoint returns stays in `DecisionResponse.raw`.

    Any endpoint that speaks Chat Completions and returns `top_logprobs` works through `base_url`; which
    models do is up to the provider, and `LOGPROB_LIMITS` records what was probed. For a model that
    returns fewer than 20, lower `max_options_per_choice` in `limits`: a letter that is not in the list
    counts as 0. For OpenRouter, set `base_url="https://openrouter.ai/api/v1"`,
    `api_key_env="OPENROUTER_API_KEY"` and `extra_body={"provider": {"require_parameters": True}}`, which
    routes only to providers that support every parameter of the request, `logprobs` included. That
    route has not been tried against the live service.

    The identity `model_id` is `"<model>@<host>"`, taken from `model` and `base_url` (`ValueError` unless
    it is an absolute http(s) URL without credentials: pass the key as `api_key`). The `model` field of a
    response, which names a dated snapshot, is never used. `limits` is declared data: a request beyond it
    raises `ValueError` before anything is sent. `extra_body` and the `**options` of `ask` are merged into
    the top level of the payload, shallowly and last write wins, so either can override any field,
    `model` included; the identity does not follow.

    The key is `api_key` or, read at call time, the `api_key_env` variable; with neither, no
    `Authorization` header is sent, for a server without auth. It never appears in `repr` or in an error
    message. A transport error, a 429 or a transient server error is retried up to `max_retries` times
    with backoff; any other failure raises `DecisionError`. `http_client` is used as is, and never closed
    by the model; `timeout` applies only to a client the model creates on its first call, which
    `aclose()` closes.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str | None = "OPENAI_API_KEY",
        limits: ModelLimits = LOGPROB_LIMITS,
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
            f"{self._base_url}/chat/completions",
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
        """Only `choice`."""
        return _KINDS

    @property
    def prompt_version(self) -> str:
        """`PROMPT_VERSION`: the version of the prompt this adapter builds."""
        return PROMPT_VERSION

    def __repr__(self) -> str:
        return f"OpenAILogprobModel(model={self._model!r}, base_url={self._base_url!r})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask the one choice question of `request`.

        `**options` are merged into the payload after the instance's `extra_body`, shallowly, so the
        last write wins. The answer follows the order of the options and sums to 1.

        Raises:
            ValueError: `request` does not fit the model: not one choice question, or more options than
                `limits` declares or than letters (26). Nothing is sent.
            DecisionError: The call failed, or the response has no option letter among its top logprobs.
        """
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        # Limits are configuration and a caller may lift a cap; what the prompt can express stays.
        if len(request.questions) != 1:
            raise ValueError(f"the model asks one question per request, the request has {len(request.questions)}")
        ((key, asked),) = request.questions.items()
        question = cast("ChoiceQuestion", asked)  # check_request has refused every other kind
        if len(question.options) > len(_LETTERS):
            raise ValueError(
                f"question {key!r} has {len(question.options)} options, the letters can label at most {len(_LETTERS)}"
            )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_message(request.state, question)},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
        }
        payload |= self._extra_body
        payload |= options
        return self._decode(key, question, await self._poster.post(payload))

    async def aclose(self) -> None:
        """Close the HTTP client if the model created it; an injected client stays open."""
        await self._poster.aclose()

    @staticmethod
    def _user_message(state: str, question: ChoiceQuestion) -> str:
        options = "\n".join(
            f"{_LETTERS[index]}. {_encode(text)}" for index, text in enumerate(question.options.values())
        )
        return _USER_PROMPT.format(state=_encode(state), instructions=_encode(question.instructions), options=options)

    def _decode(self, key: str, question: ChoiceQuestion, reply: JsonReply) -> DecisionResponse:
        try:
            completion = _Completion.model_validate(reply.body)
        except ValidationError as error:
            problem = error.errors(include_url=False, include_context=False, include_input=False)[0]
            where = ".".join(str(part) for part in problem["loc"])
            raise self._error(f"unexpected response at {where}: {problem['msg']}") from None
        top = completion.choices[0].logprobs.content[0].top_logprobs
        letters = _LETTERS[: len(question.options)]
        # Membership in a dict, not in the string: "AB" and "" are `in` "ABC".
        weights = dict.fromkeys(letters, 0.0)
        found = [(text, entry.logprob) for entry in top if (text := entry.token.strip()) in weights]
        if not found:
            said = ", ".join(repr(entry.token) for entry in top[:5])[:_SHOWN_CHARACTERS]
            raise self._error(
                f"no option letter {letters[0]}-{letters[-1]} among the top logprobs, which start with {said}"
            )
        best = max(logprob for _, logprob in found)
        # Relative to the best letter, so exp() neither overflows nor underflows to zero for every letter.
        for letter, logprob in found:
            weights[letter] += math.exp(logprob - best)
        answer = choice_answer(
            {option: weights[letter] for letter, option in zip(letters, question.options, strict=True)},
            keys=list(question.options),
            model_id=self._model_id,
        )
        return DecisionResponse(
            answers={key: answer},
            usage=DecisionUsage(
                requests=1,
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
            ),
            seconds=reply.seconds,
            raw=reply.body,
        )

    def _error(self, message: str) -> DecisionError:
        return DecisionError(f"{self._model_id}: {message}")
