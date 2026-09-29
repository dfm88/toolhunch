import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import pytest

from toolhunch.decision import (
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionModel,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    QuestionKind,
    ScoreAnswer,
    ScoreQuestion,
)
from toolhunch_bench.decision_cache import CachedDecisionModel, request_key, response_from_json, response_to_json

pytestmark = pytest.mark.anyio


class FakeInner:
    """A decision model that answers every call differently, so a replayed response differs from a fresh one.

    The first `failures` calls raise `DecisionError`. Every call is recorded, failed ones included.
    """

    def __init__(self, *, tag: str = "inner", failures: int = 0) -> None:
        self.model_id = "fake@test"
        self.limits = ModelLimits(source="test", checked=date(2026, 9, 29))
        self.question_kinds: frozenset[QuestionKind] = frozenset({"choice"})
        self.prompt_version: str | None = "p1"
        self.tag = tag
        self.calls: list[tuple[DecisionRequest, dict[str, Any]]] = []
        self._failures = failures

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        await anyio.sleep(0)  # a checkpoint, so that concurrent asks interleave
        self.calls.append((request, options))
        call = len(self.calls)
        if call <= self._failures:
            raise DecisionError(f"{self.model_id}: no usable answer")
        return DecisionResponse(
            answers={
                question_id: ChoiceAnswer(probabilities={key: 1 / len(question.options) for key in question.options})
                for question_id, question in request.questions.items()
                if isinstance(question, ChoiceQuestion)
            },
            usage=DecisionUsage(1, 100 + call, 2),
            seconds=0.25 * call,
            server_seconds=0.125 * call,
            raw={"tag": self.tag, "call": call, "text": "café"},
        )


def choice(
    *,
    state: str = "weather in Rome",
    instructions: str = "Which tool answers the request?",
    options: Mapping[str, str] | None = None,
) -> DecisionRequest:
    shown = {"a": "get_weather: Weather forecast.", "b": "send_email: Send an email."} if options is None else options
    return DecisionRequest(state=state, questions={"tool": ChoiceQuestion(instructions=instructions, options=shown)})


def test_the_key_is_the_sha256_of_a_canonical_json() -> None:
    request = DecisionRequest(
        state="café",
        questions={
            "tool": ChoiceQuestion(instructions="pick", options={"b": "two", "a": "one"}),
            "fits": BinaryQuestion(instructions="does it fit"),
            "level": ScoreQuestion(instructions="how well", levels=("low", "high")),
        },
    )
    canonical = (
        '{"model":"m@h","prompt_version":"letters-v1","state":"caf\\u00e9","questions":['
        '["tool","choice","pick",[["b","two"],["a","one"]]],'
        '["fits","binary","does it fit",null],'
        '["level","score","how well",["low","high"]]],'
        '"options":{"seed":1,"temperature":0}}'
    )

    key = request_key("m@h", request, {"temperature": 0, "seed": 1}, prompt_version="letters-v1")

    assert key == hashlib.sha256(canonical.encode()).hexdigest()
    # A model whose prompt template changed is asked afresh: its key differs from the old template's.
    assert request_key("m@h", request, {"temperature": 0, "seed": 1}, prompt_version="letters-v2") != key
    # One with no template of its own (Jev, CLM) has a null version.
    unversioned = hashlib.sha256(canonical.replace('"letters-v1"', "null").encode()).hexdigest()
    assert request_key("m@h", request, {"temperature": 0, "seed": 1}, prompt_version=None) == unversioned


def test_option_order_and_options_are_part_of_the_key() -> None:
    def key(request: DecisionRequest, options: Mapping[str, Any]) -> str:
        return request_key("m@h", request, options, prompt_version=None)

    base = key(choice(), {"temperature": 0})

    swapped = choice(options={"b": "send_email: Send an email.", "a": "get_weather: Weather forecast."})
    assert key(swapped, {"temperature": 0}) != base
    assert key(choice(), {"temperature": 1}) != base
    assert key(choice(), {}) != base
    # The options are keyed by name, at every depth: the order they were passed in does not matter.
    nested = key(choice(), {"seed": 1, "extra": {"x": 1, "y": 2}})
    assert nested == key(choice(), {"extra": {"y": 2, "x": 1}, "seed": 1})


async def test_a_hit_replays_the_stored_response_and_bills_nothing(tmp_path: Path) -> None:
    inner = FakeInner()
    cache = CachedDecisionModel(inner, path=tmp_path / "decisions.sqlite")

    first = await cache.ask(choice(), temperature=0)
    second = await cache.ask(choice(), temperature=0)
    cache.close()

    assert len(inner.calls) == 1
    assert second == first  # `seconds`, `server_seconds`, usage and `raw` included
    assert (cache.hits, cache.misses, cache.billed) == (1, 1, first.usage)


async def test_a_reopened_cache_replays_what_an_earlier_one_stored(tmp_path: Path) -> None:
    path = tmp_path / "decisions.sqlite"
    earlier = CachedDecisionModel(FakeInner(), path=path)
    stored = await earlier.ask(choice(), temperature=0)
    earlier.close()

    inner = FakeInner()
    reopened = CachedDecisionModel(inner, path=path)
    replayed = await reopened.ask(choice(), temperature=0)
    reopened.close()

    assert replayed == stored
    assert (inner.calls, reopened.hits, reopened.misses, reopened.billed) == ([], 1, 0, DecisionUsage())

    # The same model under a new prompt template is asked again: the answers of the old template are not replayed.
    revised_model = FakeInner()
    revised_model.prompt_version = "p2"
    revised = CachedDecisionModel(revised_model, path=path)
    await revised.ask(choice(), temperature=0)
    revised.close()
    assert (len(revised_model.calls), revised.hits, revised.misses) == (1, 0, 1)


async def test_bypass_always_asks(tmp_path: Path) -> None:
    path = tmp_path / "decisions.sqlite"
    warm = CachedDecisionModel(FakeInner(tag="stored"), path=path)
    stored = await warm.ask(choice())
    warm.close()

    inner = FakeInner(tag="fresh")
    bypassing = CachedDecisionModel(inner, path=path, bypass=True)
    answered = [await bypassing.ask(request) for request in (choice(), choice(), choice(state="another request"))]
    bypassing.close()

    assert len(inner.calls) == 3  # even the request that is stored, and the same one twice
    assert answered[0].raw["tag"] == "fresh"
    assert (bypassing.hits, bypassing.misses) == (0, 3)
    assert bypassing.billed == sum((response.usage for response in answered), DecisionUsage())

    after = CachedDecisionModel(FakeInner(), path=path)
    assert await after.ask(choice()) == stored  # nothing was overwritten
    await after.ask(choice(state="another request"))  # and nothing was written
    after.close()
    assert (after.hits, after.misses) == (1, 1)

    untouched = tmp_path / "untouched.sqlite"
    CachedDecisionModel(FakeInner(), path=untouched, bypass=True).close()
    assert not untouched.exists()


async def test_a_failed_ask_raises_and_is_neither_stored_nor_billed(tmp_path: Path) -> None:
    inner = FakeInner(failures=1)
    cache = CachedDecisionModel(inner, path=tmp_path / "decisions.sqlite")

    with pytest.raises(DecisionError, match="fake@test"):
        await cache.ask(choice())
    assert (cache.hits, cache.misses, cache.billed) == (0, 0, DecisionUsage())

    retried = await cache.ask(choice())  # the error was not replayed: the model is asked again
    cache.close()

    assert len(inner.calls) == 2
    assert (cache.hits, cache.misses, cache.billed) == (0, 1, retried.usage)


async def test_concurrent_asks_are_each_stored(tmp_path: Path) -> None:
    path = tmp_path / "decisions.sqlite"
    inner = FakeInner()
    cache = CachedDecisionModel(inner, path=path)

    async with anyio.create_task_group() as group:
        for request in (choice(), choice(state="another request"), choice()):
            group.start_soon(cache.ask, request)
    cache.close()

    # The third ask repeats one still in flight: both reach the model, and the later write replaces the earlier.
    assert (len(inner.calls), cache.hits, cache.misses) == (3, 0, 3)
    reopened = CachedDecisionModel(FakeInner(), path=path)
    await reopened.ask(choice())
    await reopened.ask(choice(state="another request"))
    reopened.close()
    assert (reopened.hits, reopened.misses) == (2, 0)


def test_every_answer_kind_round_trips_through_json() -> None:
    full = DecisionResponse(
        answers={
            "pick": ChoiceAnswer(probabilities={"b": 0.75, "a": 0.25}),
            "fits": BinaryAnswer(probability=0.125),
            "level": ScoreAnswer(expected=1.4, probabilities=(0.1, 0.4, 0.5)),
            "level-only": ScoreAnswer(expected=0.5),
        },
        usage=DecisionUsage(1, 120, 3),
        seconds=0.375,
        server_seconds=0.25,
        raw={"id": "cmpl-1", "confidence": None, "note": "café", "top": [{"token": "B", "logprob": -0.0001}]},
    )
    bare = DecisionResponse(answers={"fits": BinaryAnswer(probability=1.0)}, usage=DecisionUsage(), seconds=0.0, raw={})

    for response in (full, bare):
        stored = json.dumps(response_to_json(response))  # the text the cache keeps
        assert response_from_json(json.loads(stored)) == response

    replayed = response_from_json(json.loads(json.dumps(response_to_json(full))))
    pick = replayed.answers["pick"]
    assert isinstance(pick, ChoiceAnswer)
    assert list(pick.probabilities) == ["b", "a"]  # option order survives, which comparing dicts does not check
    assert response_to_json(bare) == {
        "answers": {"fits": {"kind": "binary", "probability": 1.0}},
        "usage": {"requests": 0, "input_tokens": 0, "output_tokens": 0},
        "seconds": 0.0,
        "server_seconds": None,
        "raw": {},
    }
    assert response_to_json(full)["answers"] == {
        "pick": {"kind": "choice", "probabilities": {"b": 0.75, "a": 0.25}},
        "fits": {"kind": "binary", "probability": 0.125},
        "level": {"kind": "score", "expected": 1.4, "probabilities": [0.1, 0.4, 0.5]},
        "level-only": {"kind": "score", "expected": 0.5, "probabilities": None},
    }


def test_it_takes_the_identity_of_the_model_it_wraps(tmp_path: Path) -> None:
    inner = FakeInner()
    cache = CachedDecisionModel(inner, path=tmp_path / "decisions.sqlite")
    model: DecisionModel = cache  # that it satisfies the protocol is pyright's to check

    assert (model.model_id, model.limits, model.question_kinds, model.prompt_version) == (
        "fake@test",
        inner.limits,
        frozenset({"choice"}),
        "p1",
    )
    cache.close()
