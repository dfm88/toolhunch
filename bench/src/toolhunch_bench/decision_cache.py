"""A persistent cache in front of a decision model: each request is paid once, and a re-run replays the response."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from toolhunch.decision import (
    BinaryAnswer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    DecisionUsage,
    ScoreAnswer,
    ScoreQuestion,
)
from toolhunch_bench import BENCH_DIR

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from toolhunch.decision import Answer, DecisionModel, DecisionRequest, ModelLimits, Question, QuestionKind

__all__ = ["DECISION_CACHE_PATH", "CachedDecisionModel", "request_key", "response_from_json", "response_to_json"]

DECISION_CACHE_PATH = BENCH_DIR / "runs" / "cache" / "decisions.sqlite"


def request_key(
    model_id: str, request: DecisionRequest, options: Mapping[str, Any], *, prompt_version: str | None
) -> str:
    """The cache key of one ask: the SHA-256 hex digest of a canonical JSON of what the model is asked.

    The JSON is compact ASCII with the members `model`, `prompt_version`, `state`, `questions` and
    `options`, in that order. `prompt_version` is the version of the model's own prompt template
    (`DecisionModel.prompt_version`), `null` for a model without one: a changed template never replays the
    answers of the old one. A question is `[id, kind, instructions, payload]`; questions keep the order of
    the request, and the payload is `[[key, text], ...]` for a choice (options in presentation order), the
    levels for a score and `null` for a binary question. Only `options`, the keyword options of the call,
    is sorted by key, at every depth.

    The key is a persistent format: building it differently orphans every response stored under the old one.
    """
    canonical = json.dumps(
        {
            "model": model_id,
            "prompt_version": prompt_version,
            "state": request.state,
            "questions": [
                [question_id, question.kind, question.instructions, _payload(question)]
                for question_id, question in request.questions.items()
            ],
            # A round trip through `sort_keys` orders nested mappings too; the outer dump then keeps that order.
            "options": json.loads(json.dumps(dict(options), sort_keys=True)),
        },
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _payload(question: Question) -> list[list[str]] | list[str] | None:
    if isinstance(question, ChoiceQuestion):
        return [[key, text] for key, text in question.options.items()]
    if isinstance(question, ScoreQuestion):
        return list(question.levels)
    return None


def response_to_json(response: DecisionResponse) -> dict[str, Any]:
    """`response` as a JSON-compatible dict, the form the cache stores; `response_from_json` reverses it.

    Each answer carries its `kind` (`"choice"`, `"binary"` or `"score"`). `raw` must hold only JSON types,
    as the decoded body of a response does.
    """
    return {
        "answers": {question_id: _answer_to_json(answer) for question_id, answer in response.answers.items()},
        "usage": asdict(response.usage),
        "seconds": response.seconds,
        "server_seconds": response.server_seconds,
        "raw": dict(response.raw),
    }


def response_from_json(data: Mapping[str, Any]) -> DecisionResponse:
    """The `DecisionResponse` that `response_to_json` turned into `data`, option order included.

    Raises:
        ValueError: An answer has a kind that is not `"choice"`, `"binary"` or `"score"`.
    """
    usage = data["usage"]
    return DecisionResponse(
        answers={question_id: _answer_from_json(answer) for question_id, answer in data["answers"].items()},
        usage=DecisionUsage(
            requests=usage["requests"], input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]
        ),
        seconds=data["seconds"],
        raw=data["raw"],
        server_seconds=data["server_seconds"],
    )


def _answer_to_json(answer: Answer) -> dict[str, Any]:
    if isinstance(answer, ChoiceAnswer):
        return {"kind": "choice", "probabilities": dict(answer.probabilities)}
    if isinstance(answer, BinaryAnswer):
        return {"kind": "binary", "probability": answer.probability}
    return {
        "kind": "score",
        "expected": answer.expected,
        "probabilities": None if answer.probabilities is None else list(answer.probabilities),
    }


def _answer_from_json(data: Mapping[str, Any]) -> Answer:
    match data["kind"]:
        case "choice":
            return ChoiceAnswer(probabilities=data["probabilities"])
        case "binary":
            return BinaryAnswer(probability=data["probability"])
        case "score":
            levels = data["probabilities"]
            return ScoreAnswer(expected=data["expected"], probabilities=None if levels is None else tuple(levels))
        case kind:
            raise ValueError(f"unknown answer kind {kind!r}")


class CachedDecisionModel:
    """A [`DecisionModel`][toolhunch.decision.DecisionModel] that keeps every response in SQLite.

    A request is put to the wrapped model once. Its response is stored under `request_key` as soon as it
    arrives, so a run that fails keeps what it paid for, and asking the same again, in this run or a later
    one, replays the stored response: the original `seconds`, `server_seconds`, usage and `raw`, at no cost.
    The wrapped model's `model_id`, `limits`, `question_kinds` and `prompt_version` are passed through.

    The key covers the model id, the model's prompt version, the request and the options given to `ask`. What
    else changes a model's answers, such as an `extra_body` set on the instance, is not in it: give each such
    configuration its own `path`. Concurrent asks for the same request may all reach the wrapped model, and
    the last response stored replaces the others.

    With `bypass` the database is never opened: nothing is read or written and every ask goes to the wrapped
    model, for checking that its answers hold still.

    Attributes:
        billed: What the responses that came from the wrapped model consumed; a replay adds nothing.
        hits: Asks answered from the cache.
        misses: Asks answered by the wrapped model, which are the ones `billed` adds up.
        failures: Asks the wrapped model raised on, with or without `bypass`. They are neither stored nor in `billed`,
            though a provider may have billed some of them: a response it sent can still be unusable.
    """

    @property
    def bypasses_cache(self) -> bool:
        """Whether every ask goes directly to the wrapped model without opening a database."""
        return self._db is None

    def __init__(self, inner: DecisionModel, *, path: Path = DECISION_CACHE_PATH, bypass: bool = False) -> None:
        """Open (or create) the cache at `path` in front of `inner`; with `bypass`, leave `path` alone."""
        self._inner = inner
        self._db: sqlite3.Connection | None = None
        if not bypass:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, model_id TEXT, response TEXT)"
            )
        self.billed = DecisionUsage()
        self.hits = 0
        self.misses = 0
        self.failures = 0

    @property
    def model_id(self) -> str:
        """The wrapped model's id."""
        return self._inner.model_id

    @property
    def limits(self) -> ModelLimits:
        """The wrapped model's declared limits."""
        return self._inner.limits

    @property
    def question_kinds(self) -> frozenset[QuestionKind]:
        """The kinds of question the wrapped model answers."""
        return self._inner.question_kinds

    @property
    def prompt_version(self) -> str | None:
        """The wrapped model's own prompt version, if it has one."""
        return self._inner.prompt_version

    def __repr__(self) -> str:
        return f"CachedDecisionModel({self._inner!r}, bypass={self._db is None})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Answer `request` from the cache, or by asking the wrapped model and storing what it returns.

        A stored response comes back as it was stored and adds nothing to `billed`.

        Raises:
            DecisionError: The wrapped model failed. This, like any other error of the wrapped model, is raised
                as it is, nothing is stored, and the ask is counted in `failures`.
        """
        if self._db is None:
            return await self._forward(request, options)
        key = request_key(self.model_id, request, options, prompt_version=self.prompt_version)
        row = self._db.execute("SELECT response FROM decisions WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return response_from_json(json.loads(row[0]))
        response = await self._forward(request, options)
        self._db.execute(
            "INSERT OR REPLACE INTO decisions (key, model_id, response) VALUES (?, ?, ?)",
            (key, self.model_id, json.dumps(response_to_json(response))),
        )
        self._db.commit()
        return response

    def close(self) -> None:
        """Close the database. The wrapped model is left open: its `aclose`, if it has one, is its owner's to call."""
        if self._db is not None:
            self._db.close()

    async def _forward(self, request: DecisionRequest, options: Mapping[str, Any]) -> DecisionResponse:
        try:
            response = await self._inner.ask(request, **options)
        except BaseException:  # a cancelled ask counts too: it may have reached the provider
            self.failures += 1
            raise
        self.misses += 1
        self.billed += response.usage
        return response
