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
    "LAYA_LIMITS",
    "RIZZO_FLOW_LIMITS",
    "STRANDS_LIMITS",
    "JevWireModel",
    "clef",
    "clm",
    "jev",
    "laya",
    "rizzo_flow",
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

LAYA_LIMITS = ModelLimits(
    max_options_per_choice=100,
    max_state_plus_question_tokens=512,
    max_question_tokens=192,
    max_option_tokens=48,
    max_questions_per_request=64,
    price_input_per_mtok=0.0,
    price_output_per_mtok=0.0,
    source=(
        "laya 0.3.27 serve.py (100 options, 64 questions) and common.py build_head/render_options (48-token "
        "option cap); convaiinnovations/laya @ 55cf4c4 rl_agent_config.json (max_len 512, head_max_len 192)"
    ),
    checked=date(2026, 10, 5),
)
"""Laya's limits as its English checkpoint ships them, served locally at no charge.

The question budget is the head (the question and every option), and the state budget is the head plus the state.
Laya cuts what exceeds them instead of refusing.
"""

# What the English checkpoint reads when a request does not set the budgets: `head_max_len` and `max_len` in its
# rl_agent_config.json. `LAYA_LIMITS` keeps its own margin below them.
_LAYA_HEAD_MAX_LEN = 192
_LAYA_MAX_LEN = 512
# The checkpoint names Laya 0.3.27 reports in `routing.model` (`router.DEFAULT_MODELS`). Its server also accepts
# aliases such as `laya` and `en`, and Hub repo ids, and answers with the canonical name.
_LAYA_CHECKPOINTS = ("english", "multilingual", "typed-decisions")

# The server's `--ctx` default, and what its prompt template takes beyond the state and the question: a minimal
# request costs 149 tokens, rounded up to a multiple of 64.
_RIZZO_FLOW_CONTEXT = 8192
_RIZZO_FLOW_TEMPLATE_MARGIN = 192
RIZZO_FLOW_LIMITS = ModelLimits(
    max_options_per_choice=26,
    max_state_plus_question_tokens=_RIZZO_FLOW_CONTEXT - _RIZZO_FLOW_TEMPLATE_MARGIN,
    max_questions_per_request=64,
    price_input_per_mtok=0.0,
    price_output_per_mtok=0.0,
    source=(
        "Rizzo-AI-Academy/rizzo-flow @ b9ba007: schema.py MAX_SLOTS = 26; --ctx 8192 bounds each question with its "
        "state (prompts.py compile_request); template margin 192 from a probe on 2026-10-05: a minimal request "
        "costs 149 tokens"
    ),
    checked=date(2026, 10, 5),
)
"""rizzo-flow's limits: 26 options, and an 8,192-token context for each question with its state, served locally."""

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
    key as `api_key`. Requests go to `base_url + path`, `path` given a leading slash when it lacks one. The identity
    `model_id` is `"<model>@<host>"`, taken from `model` and `base_url`, and is all `repr` shows, so an account ID in
    the URL never reaches it. The `model` and `confidence` fields of a response are never used; they stay in
    `DecisionResponse.raw`.

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
            f"{self._base_url}/{path.lstrip('/')}",
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


class _StrictLayaModel(JevWireModel):
    """`JevWireModel` for Laya that refuses a reply in which Laya cut or swapped something and said so."""

    def _decode(self, request: DecisionRequest, reply: JsonReply) -> DecisionResponse:
        self._refuse_lossy(reply.body)
        return super()._decode(request, reply)

    def _refuse_lossy(self, body: Mapping[str, Any]) -> None:
        usage = body.get("usage")
        usage = cast("dict[str, Any]", usage) if isinstance(usage, dict) else {}
        if usage.get("truncated"):
            dropped, total = usage.get("state_tokens_dropped", "?"), usage.get("state_tokens", "?")
            named = (
                f" (truncated_questions: {json.dumps(questions)[:200]})"
                if (questions := usage.get("truncated_questions"))
                else ""
            )
            raise self._error(f"Laya cut the state: {dropped} of {total} state tokens dropped{named}")
        if collapsed := usage.get("options"):
            raise self._error(f"Laya collapsed options that it renders identically: {json.dumps(collapsed)[:500]}")
        routing = body.get("routing")
        if not isinstance(routing, dict):
            raise self._error("no routing in the reply: is LAYA_JEV_STRICT set? Unset it, or pass strict=False")
        # An unknown model id is not refused: Laya routes the request to a checkpoint of its own choosing.
        if (answered := cast("dict[str, Any]", routing).get("model")) != self._model:
            raise self._error(f"Laya answered with checkpoint {answered!r}, not {self._model!r}")


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
    4,096-token window fails instead of losing part of its state. The planner keeps requests within
    `STRANDS_LIMITS`: it renders the cards at the most detailed level that fits, and splits a choice into two
    rounds only when it does not fit even by name.
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


def laya(
    base_url: str = "http://127.0.0.1:8010",
    *,
    model: str = "english",
    head_max_len: int | None = None,
    max_len: int | None = None,
    strict: bool = True,
    limits: ModelLimits | None = None,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """A Laya server: `laya-serve` from the laya package (convaiinnovations/laya), without auth.

    `base_url` is the server root; `/v1` is appended. `model` is the checkpoint: `"english"` (the default),
    `"multilingual"` or `"typed-decisions"`. `server_seconds` comes from the `X-Inference-Time-Ms` header.

    Laya reads the question (its instructions and options) within `head_max_len` tokens and the question with the
    state within `max_len`, and cuts what exceeds them. The server's defaults, 192 and 512, are the English
    checkpoint's, and so are the default `limits`: `LAYA_LIMITS`. Other checkpoints read more, so those limits are
    safe for them but small. Whatever `head_max_len` and `max_len` the body ends up carrying, from these arguments
    or from `extra_body` (which is merged after them and wins), is what the default `limits` follow, by the same
    margin that `LAYA_LIMITS` keeps below the defaults, so the planner shows the model more per question. Pass
    `limits` to set everything yourself.

    With `strict` (the default) a reply raises `DecisionError` when Laya reports that it cut the state, reports
    options it collapsed into one, answers with another checkpoint than `model` (an unknown model id is not
    refused, Laya routes it to a checkpoint of its own choosing), or carries no `routing` (a server run with
    `LAYA_JEV_STRICT` leaves out the fields `strict` reads). `strict` cannot see what Laya does not report: an
    option cut at 48 tokens, or shortened while still distinct from the others. `LAYA_LIMITS` declares that cap,
    and the planner keeps within it. With `strict=False` such a reply is decoded as any other.

    Raises:
        ValueError: `strict` is on and `model` is not one of the three checkpoint names (the server accepts aliases
            such as `laya` and Hub repo ids, but reports the canonical name, so a strict model cannot tell them from
            another checkpoint: pass `strict=False` to use one). Or the budgets the body carries are not integers
            of at least 1, or `head_max_len` is not smaller than `max_len`, where a budget that is not sent counts
            as the server's default.
    """
    if strict and model not in _LAYA_CHECKPOINTS:
        raise ValueError(
            f"with strict=True, model must be one of {', '.join(_LAYA_CHECKPOINTS)}, got {model!r}: "
            "Laya reports the canonical name for an alias, so pass strict=False to use an alias"
        )
    body: dict[str, Any] = {}
    if head_max_len is not None:
        body["head_max_len"] = head_max_len
    if max_len is not None:
        body["max_len"] = max_len
    body |= dict(extra_body or {})
    head, length = _laya_budgets(body)
    return (_StrictLayaModel if strict else JevWireModel)(
        model,
        base_url=f"{base_url.rstrip('/')}/v1",
        api_key_env=None,
        limits=_laya_limits(head, length) if limits is None else limits,
        extra_body=body,
        latency_header="X-Inference-Time-Ms",
        http_client=http_client,
        timeout=timeout,
        max_retries=max_retries,
    )


def _laya_budgets(body: Mapping[str, Any]) -> tuple[int | None, int | None]:
    """The `head_max_len` and `max_len` that `body` sends, checked: `None` for one it leaves out."""
    budgets: list[int | None] = []
    for name in ("head_max_len", "max_len"):
        value = body.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f"{name} must be an integer of at least 1, got {value!r}")
        budgets.append(value)
    head, length = budgets
    effective_head = _LAYA_HEAD_MAX_LEN if head is None else head
    effective_length = _LAYA_MAX_LEN if length is None else length
    if effective_head >= effective_length:
        raise ValueError(
            f"head_max_len ({effective_head}) must be smaller than max_len ({effective_length}); "
            f"the server's defaults are {_LAYA_HEAD_MAX_LEN} and {_LAYA_MAX_LEN} for a budget that is not sent"
        )
    return head, length


def _laya_limits(head_max_len: int | None, max_len: int | None) -> ModelLimits:
    """`LAYA_LIMITS` for the budgets a request sends: the margin below the server's defaults stays what it is."""
    update: dict[str, Any] = {}
    sent: list[str] = []
    if head_max_len is not None:
        margin = _LAYA_HEAD_MAX_LEN - cast("int", LAYA_LIMITS.max_question_tokens)
        update["max_question_tokens"] = head_max_len - margin
        sent.append(f"head_max_len {head_max_len}")
    if max_len is not None:
        margin = _LAYA_MAX_LEN - cast("int", LAYA_LIMITS.max_state_plus_question_tokens)
        update["max_state_plus_question_tokens"] = max_len - margin
        sent.append(f"max_len {max_len}")
    if sent:
        update["source"] = f"{LAYA_LIMITS.source}; sent with every request: {', '.join(sent)}"
    # Validated, unlike `model_copy(update=...)`: a margin that leaves nothing is an error, not a limit of 0.
    return ModelLimits.model_validate({**LAYA_LIMITS.model_dump(), **update})


def rizzo_flow(
    base_url: str = "http://127.0.0.1:8017",
    *,
    model: str = "rizzo-flow-4b-q8_0",
    limits: ModelLimits = RIZZO_FLOW_LIMITS,
    extra_body: Mapping[str, Any] | None = None,
    http_client: httpx2.AsyncClient | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> JevWireModel:
    """A rizzo-flow server: `rizzo serve` from Rizzo-AI-Academy/rizzo-flow, without auth.

    `base_url` is the server root; `/v1` is appended. The server sends no latency header, so `server_seconds` is
    `None`; its own timing is in the reply's `x_rizzo.timing`, kept in `DecisionResponse.raw`.

    The default `model` names the weights, a 4B model at q8_0 quantization. A server loaded with other weights
    answers HTTP 400, so a run cannot silently use different weights under the same name. `rizzo-latest` is
    accepted by every server, whatever it loaded, which is why it is not the default; the server also answers to
    any `jev-*` id.

    The server takes at most 26 options per choice, and refuses a question whose state does not fit its context
    (HTTP 422, no cutting); `limits` declares both, so such a request fails before anything is sent.
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
