"""Decision models that speak TypeSafe's `/v1/systemone` protocol: Jev, and servers that copy it.

CLM and Strands Decider serve it at `/v1/systemone`; Cloudflare's Clef takes the same body at its own Workers AI
endpoint and wraps the answer in a `result` envelope.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from datetime import date
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, Field, ValidationError

from toolhunch.decision._http import Endpoint, JsonPoster
from toolhunch.decision.base import (
    BinaryAnswer,
    ChoiceQuestion,
    DecisionError,
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

__all__ = [
    "CLEF_FLASH_LIMITS",
    "CLEF_LIMITS",
    "CLM_LIMITS",
    "JEV_LIMITS",
    "STRANDS_LIMITS",
    "JevWireModel",
    "clef",
    "clm",
    "jev",
    "strands_decider",
]

# TypeSafe's docs say "64k" and "32k": read as 64,000 and 32,000, the lower of the two readings.
JEV_LIMITS = ModelLimits(
    max_options_per_choice=255,
    max_request_tokens=64_000,
    max_state_plus_question_tokens=32_000,
    score_levels=(2, 10),
    price_input_per_mtok=0.042,
    price_output_per_mtok=0.0,
    source="https://docs.typesafe.ai (limits and pricing)",
    checked=date(2026, 9, 27),
)
"""Jev's limits and price, as TypeSafe documents them."""

CLM_LIMITS = ModelLimits(
    max_text_tokens=2048,
    source=(
        "Contrastive-LM/CLM @ bb42c6c: clm-serve cuts each text at 2,048 tokens by default; "
        "bench/deploy/clm_modal.py --max-model-len 2048; a 7,700-token option billed as 2,048 tokens "
        "(probe 2026-09-28)"
    ),
    checked=date(2026, 9, 28),
)
"""CLM's limits: its server cuts every text at 2,048 tokens, without saying so."""

STRANDS_LIMITS = ModelLimits(
    max_options_per_choice=255,
    max_request_tokens=4096,
    score_levels=(2, 10),
    price_input_per_mtok=0.0,
    price_output_per_mtok=0.0,
    source=(
        "strands-labs/strands-decider @ 75c9fd3: schema.py MAX_CHOICE_OPTIONS = 255, score levels 2-10; "
        "StrandsAgents/strands-decider-2B-hobson-v19 @ bb282d7: hobson_config.json max_length 4096; runs locally"
    ),
    checked=date(2026, 10, 4),
)
"""Strands Decider 2B's limits: 255 options and a 4,096-token window, served locally at no charge.

Over the window its server cuts the state unless it runs with `--strict-window`, which refuses with HTTP 422.
"""

_CLEF_SOURCE = (
    "developers.cloudflare.com/workers-ai/models/{model}: 65,536-token context, 1-64 questions, "
    "${price} per M input tokens, no output charge; option cap: a 255-option choice accepted on 2026-10-04 "
    "(api-version 2026-10-01.epoch), more not tried"
)
CLEF_LIMITS = ModelLimits(
    max_options_per_choice=255,
    max_request_tokens=65_536,
    max_questions_per_request=64,
    price_input_per_mtok=0.24,
    price_output_per_mtok=0.0,
    source=_CLEF_SOURCE.format(model="clef", price="0.24"),
    checked=date(2026, 10, 4),
)
"""Cloudflare Clef's limits and price on Workers AI."""

CLEF_FLASH_LIMITS = CLEF_LIMITS.model_copy(
    update={"price_input_per_mtok": 0.09, "source": _CLEF_SOURCE.format(model="clef-flash", price="0.09")}
)
"""Cloudflare Clef-flash's limits and price on Workers AI."""

_CLEF_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run"

_JEV_BASE_URL = "https://api.typesafe.ai/v1"
_KINDS: frozenset[QuestionKind] = frozenset({"choice", "binary", "score"})
_WIRE_TYPES = {"choice": "choice", "binary": "noul", "score": "score"}


class _Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class _Envelope(BaseModel):
    answers: dict[str, dict[str, Any]]
    usage: _Usage = Field(default_factory=_Usage)


class JevWireModel:
    """A decision model behind `POST {base_url}/systemone`, the wire protocol of TypeSafe's Jev.

    One class serves Jev itself (see `jev`) and any server that implements the same protocol, such as
    CLM (see `clm`). It answers choice, binary and score questions.

    `base_url` must be an absolute http(s) URL without credentials: `ValueError` otherwise, so pass the
    key as `api_key`. Requests go to `base_url + path`. The identity `model_id` is `"<model>@<host>"`, taken from
    `model` and `base_url`, and is all `repr` shows, so an account ID in the URL never reaches it. The `model` and
    `confidence` fields of a response are never used; they stay in `DecisionResponse.raw`.

    `response_root` names the member of the reply that holds the answer, for a server that wraps it in an envelope
    (`"result"` on Workers AI); `None` reads the reply itself.
    `limits` is declared data: a request beyond it raises `ValueError` before anything is sent.

    `extra_body` and the `**options` of `ask` are merged into the top level of the payload, shallowly and
    last write wins, so either can override any field, `model` included; the identity does not follow.

    The key is `api_key` or, read at call time, the `api_key_env` variable; with neither, no
    `Authorization` header is sent, for a server without auth. It never appears in `repr` or in an error
    message. A transport error, a 429 or a transient server error is retried up to `max_retries` times
    with backoff; any other failure raises `DecisionError`.

    `http_client` is used as is and never closed by the model. Without one, the model creates a client on
    its first call and `aclose()` closes it. `latency_header` names the response header that carries the
    server's own latency in milliseconds; it becomes `DecisionResponse.server_seconds`.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key_env: str | None,
        limits: ModelLimits,
        api_key: str | None = None,
        extra_body: Mapping[str, Any] | None = None,
        latency_header: str | None = None,
        path: str = "/systemone",
        response_root: str | None = None,
        http_client: httpx2.AsyncClient | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """Configure the endpoint; `base_url` is the API root, such as `https://api.typesafe.ai/v1`."""
        endpoint = Endpoint.parse(base_url, model=model)
        self._model = model
        self._base_url = endpoint.base_url
        self._model_id = endpoint.model_id
        self._limits = limits
        self._extra_body = dict(extra_body or {})
        self._latency_header = latency_header
        self._response_root = response_root
        self._poster = JsonPoster(
            f"{self._base_url}{path}",
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
    def prompt_version(self) -> None:
        """Always `None`: the server owns the prompt, so the adapter has no template of its own."""
        return None

    def __repr__(self) -> str:
        return f"JevWireModel(model_id={self._model_id!r})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask every question of `request` in one call.

        `**options` are merged into the payload after the instance's `extra_body`, shallowly, so the
        last write wins. The response is checked against the request: every question must come back
        answered with its own kind, and a choice distribution must cover exactly the option keys. That
        distribution is renormalised to sum to 1 and follows the order of the options, whatever order
        the server used.

        Raises:
            ValueError: `request` does not fit the model's kinds or declared limits. Nothing is sent.
            DecisionError: The call failed, or the response does not answer `request`.
        """
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        payload: dict[str, Any] = {
            "state": request.state,
            "model": self._model,
            "questions": {key: self._question_body(question) for key, question in request.questions.items()},
        }
        payload |= self._extra_body
        payload |= options
        return self._decode(request, await self._poster.post(payload))

    async def aclose(self) -> None:
        """Close the HTTP client if the model created it; an injected client stays open."""
        await self._poster.aclose()

    @staticmethod
    def _question_body(question: Question) -> dict[str, Any]:
        body: dict[str, Any] = {"type": _WIRE_TYPES[question.kind], "instructions": question.instructions}
        if isinstance(question, ChoiceQuestion):
            # An empty option text goes out as null, as in TypeSafe's SDK.
            body["criteria"] = {key: text or None for key, text in question.options.items()}
        elif isinstance(question, ScoreQuestion):
            body["criteria"] = list(question.levels)
        return body

    def _decode(self, request: DecisionRequest, reply: JsonReply) -> DecisionResponse:
        body = reply.body if self._response_root is None else self._unwrap(reply.body)
        try:
            envelope = _Envelope.model_validate(body)
        except ValidationError as error:
            problem = error.errors(include_url=False, include_context=False, include_input=False)[0]
            where = ".".join(str(part) for part in problem["loc"])
            raise self._error(f"unexpected response at {where}: {problem['msg']}") from None
        if unasked := envelope.answers.keys() - request.questions.keys():
            raise self._error(f"the response answers questions that were not asked: {sorted(unasked)}")
        return DecisionResponse(
            answers={
                key: self._answer(key, question, envelope.answers.get(key))
                for key, question in request.questions.items()
            },
            usage=DecisionUsage(
                requests=1, input_tokens=envelope.usage.input_tokens, output_tokens=envelope.usage.output_tokens
            ),
            seconds=reply.seconds,
            server_seconds=self._server_seconds(reply.headers),
            raw=reply.body,
        )

    def _unwrap(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """The member `response_root` of `body`, which must be a JSON object; `DecisionError` otherwise."""
        root = self._response_root
        inner = body.get(root) if root is not None else body
        if not isinstance(inner, dict):
            errors = f"; errors: {json.dumps(body['errors'])[:500]}" if body.get("errors") else ""
            raise self._error(f"the reply has no {root!r} object{errors}")
        return cast("dict[str, Any]", inner)

    def _answer(self, key: str, question: Question, wire: Mapping[str, Any] | None) -> Answer:
        if wire is None:
            raise self._error(f"no answer for question {key!r}")
        expected = _WIRE_TYPES[question.kind]
        if wire.get("type") != expected:
            raise self._error(
                f"question {key!r} needs an answer of type {expected!r}, the response has type {wire.get('type')!r}"
            )
        if isinstance(question, ChoiceQuestion):
            return choice_answer(self._distribution(key, wire), keys=list(question.options), model_id=self._model_id)
        if isinstance(question, ScoreQuestion):
            if (score := self._number(wire.get("score"))) is None:
                raise self._error(f"question {key!r}: `score` is not a finite number: {wire.get('score')!r}")
            if wire.get("probabilities") is None:  # some servers answer with the expectation only
                return ScoreAnswer(expected=score)
            levels = choice_answer(
                self._distribution(key, wire),
                keys=[str(level) for level in range(len(question.levels))],
                model_id=self._model_id,
            )
            return ScoreAnswer(expected=score, probabilities=tuple(levels.probabilities.values()))
        # What is left is a binary question.
        if (noul := self._number(wire.get("noul"))) is None or not 0 <= noul <= 1:
            raise self._error(f"question {key!r}: `noul` is not a number in [0, 1]: {wire.get('noul')!r}")
        return BinaryAnswer(probability=noul)

    def _distribution(self, key: str, wire: Mapping[str, Any]) -> dict[str, object]:
        probabilities = wire.get("probabilities")
        if not isinstance(probabilities, dict):
            raise self._error(f"question {key!r}: `probabilities` is not an object: {probabilities!r}")
        return cast("dict[str, object]", probabilities)

    def _server_seconds(self, headers: Mapping[str, str]) -> float | None:
        """The server's own latency in seconds; `None` when no header is configured, or it is absent or unusable."""
        if self._latency_header is not None:
            with contextlib.suppress(ValueError):  # absent ("") or not a number
                if 0 <= (milliseconds := float(headers.get(self._latency_header, ""))) < math.inf:
                    return milliseconds / 1000
        return None

    def _error(self, message: str) -> DecisionError:
        return DecisionError(f"{self._model_id}: {message}")

    @staticmethod
    def _number(value: object) -> float | None:
        """`value` as a float when it is a finite JSON number; `None` for anything else, a `bool` included."""
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        # One exact comparison rejects NaN, infinities and ints too large for a float.
        return float(value) if -sys.float_info.max <= value <= sys.float_info.max else None


def jev(
    model: str = "jev-latest",
    *,
    api_key: str | None = None,
    api_key_env: str | None = "TYPESAFE_API_KEY",
    limits: ModelLimits = JEV_LIMITS,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """TypeSafe's Jev, at `https://api.typesafe.ai/v1`.

    The key is `api_key` or, at call time, the `TYPESAFE_API_KEY` variable. `server_seconds` comes from
    the `x-envoy-upstream-service-time` header. `model` is `jev-latest` or a pinned version such as
    `jev-1.13.0`; pin one when results must be reproducible.
    """
    return JevWireModel(
        model,
        base_url=_JEV_BASE_URL,
        api_key=api_key,
        api_key_env=api_key_env,
        limits=limits,
        extra_body=extra_body,
        latency_header="x-envoy-upstream-service-time",
        http_client=http_client,
        timeout=timeout,
        max_retries=max_retries,
    )


def clm(
    base_url: str,
    *,
    model: str = "clm-latest",
    api_key: str | None = None,
    api_key_env: str | None = "CLM_API_KEY",
    limits: ModelLimits = CLM_LIMITS,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """A CLM server: `clm-serve` from the Contrastive-LM/CLM project, which speaks Jev's wire protocol.

    `base_url` is the server root; `/v1` is appended. The key is `api_key` or, at call time, the
    `CLM_API_KEY` variable; pass `api_key_env=None` for a server that runs without auth.
    `server_seconds` comes from the `x-clm-latency-ms` header.
    """
    return JevWireModel(
        model,
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key=api_key,
        api_key_env=api_key_env,
        limits=limits,
        extra_body=extra_body,
        latency_header="x-clm-latency-ms",
        http_client=http_client,
        timeout=timeout,
        max_retries=max_retries,
    )


def strands_decider(
    base_url: str = "http://127.0.0.1:8000",
    *,
    model: str = "strands-decider-2B-hobson-v19",
    limits: ModelLimits = STRANDS_LIMITS,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """A Strands Decider server: `strands-decider serve` from strands-labs/strands-decider, without auth.

    `base_url` is the server root; `/v1` is appended. Run the server with `--strict-window`, so a prompt over its
    4,096-token window fails instead of losing part of its state; the planner keeps requests within
    `STRANDS_LIMITS` and splits larger choices into two rounds.
    """
    return JevWireModel(
        model,
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key_env=None,
        limits=limits,
        extra_body=extra_body,
        http_client=http_client,
        timeout=timeout,
        max_retries=max_retries,
    )


def clef(
    model: Literal["clef", "clef-flash"] = "clef-flash",
    *,
    account_id: str | None = None,
    account_id_env: str = "CLOUDFLARE_ACCOUNT_ID",
    api_key: str | None = None,
    api_key_env: str | None = "CLOUDFLARE_API_KEY",
    limits: ModelLimits | None = None,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """Cloudflare's Clef or Clef-flash on Workers AI, which takes Jev's body and wraps its answer in `result`.

    The account ID is `account_id` or, read now, the `account_id_env` variable; it goes into the URL only, never
    into `model_id` or `repr`. The key is `api_key` or, at call time, the `api_key_env` variable: a Cloudflare
    API token with Workers AI permission (the REST API takes `Authorization: Bearer`, which a Global API Key does
    not use). `limits` defaults to `CLEF_LIMITS` or `CLEF_FLASH_LIMITS`.

    Raises:
        ValueError: `model` is neither `"clef"` nor `"clef-flash"`, or no account ID is given or set.
    """
    if model not in ("clef", "clef-flash"):
        raise ValueError(f'model must be "clef" or "clef-flash", got {model!r}')
    account = account_id or os.environ.get(account_id_env, "")
    if not account:
        raise ValueError(f"clef needs a Cloudflare account ID: set {account_id_env} or pass account_id")
    return JevWireModel(
        model,
        base_url=_CLEF_BASE_URL.format(account=account),
        api_key=api_key,
        api_key_env=api_key_env,
        limits=limits or (CLEF_LIMITS if model == "clef" else CLEF_FLASH_LIMITS),
        extra_body=extra_body,
        path=f"/@cf/cloudflare/{model}",
        response_root="result",
        http_client=http_client,
        timeout=timeout,
        max_retries=max_retries,
    )
