import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx2
import pytest

from toolhunch.decision import (
    OPENAI_DECISIONS_LIMITS,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionModel,
    DecisionRefused,
    DecisionRequest,
    DecisionUsage,
    OpenAIDecisionModel,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

pytestmark = pytest.mark.anyio

SECRET = "sk-test-not-a-real-key"
FIXTURES = Path(__file__).parent / "fixtures"


def recorded(name: str) -> dict[str, Any]:
    """A reply the P4 probe recorded from the live API (`openai_decisions_<name>.json`), with what was sent."""
    return json.loads((FIXTURES / f"openai_decisions_{name}.json").read_text(encoding="utf-8"))


def canonical(payload: Mapping[str, Any]) -> DecisionRequest:
    """The canonical request a Decisions payload encodes: the inverse of the adapter's mapping."""
    questions: dict[str, Question] = {}
    for wire in payload["questions"]:
        if wire["type"] == "choice":
            options = {choice["value"]: choice.get("description", "") for choice in wire["choices"]}
            questions[wire["name"]] = ChoiceQuestion(instructions=wire["instructions"], options=options)
        elif wire["type"] == "predicate":
            questions[wire["name"]] = BinaryQuestion(instructions=wire["instructions"])
        else:
            levels = tuple(level["label"] for level in wire["levels"])
            questions[wire["name"]] = ScoreQuestion(instructions=wire["instructions"], levels=levels)
    return DecisionRequest(state=payload["input"], questions=questions)


def serving(
    status: int, body: Mapping[str, Any], headers: Mapping[str, str] | None = None, **configuration: Any
) -> tuple[OpenAIDecisionModel, list[httpx2.Request]]:
    """A model whose every request is answered with `body`, and the list of the requests it made."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, json=dict(body), headers=dict(headers or {}))

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAIDecisionModel(api_key=SECRET, http_client=client, max_retries=0, **configuration), requests


def answer_with(answers: list[dict[str, Any]], *, input_tokens: int = 50) -> dict[str, Any]:
    """A reply in the API's shape (openai-python v3.26.0 `Decision`) carrying `answers`."""
    usage = {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": 0,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens,
    }
    return {"model": "gpt-6-luna", "answers": answers, "usage": usage}


THREE = DecisionRequest(
    state="Request: book a table.",
    questions={"tool": ChoiceQuestion(instructions="Which tool?", options={"a": "Alpha.", "b": "", "none": "None."})},
)


def choice(probabilities: list[tuple[object, float]], *, name: object = "tool") -> dict[str, Any]:
    return {
        "type": "choice",
        "name": name,
        "choice": probabilities[0][0],
        "probabilities": [{"value": value, "probability": p} for value, p in probabilities],
        "confidence": 0.5,
    }


def test_it_is_a_decision_model_with_its_own_prompt_version() -> None:
    model: DecisionModel = OpenAIDecisionModel(api_key=SECRET)
    assert model.model_id == "gpt-6-luna@api.openai.com"
    assert model.prompt_version == "decisions-v1"
    assert model.question_kinds == {"choice", "binary", "score"}
    assert model.limits is OPENAI_DECISIONS_LIMITS
    assert SECRET not in repr(model)


@pytest.mark.parametrize("name", ["tool_choice", "mixed", "score"])
async def test_the_recorded_requests_are_what_the_adapter_sends(name: str) -> None:
    fixture = recorded(name)
    model, requests = serving(fixture["status"], fixture["response"], fixture["headers"])
    await model.ask(canonical(fixture["request"]))
    assert str(requests[0].url) == "https://api.openai.com/v1/decisions"
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert json.loads(requests[0].content) == fixture["request"]


async def test_the_recorded_replies_decode_to_the_three_answer_kinds() -> None:
    tool, mixed, score = recorded("tool_choice"), recorded("mixed"), recorded("score")

    model, _ = serving(200, tool["response"], tool["headers"])
    response = await model.ask(canonical(tool["request"]))
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == [choice["value"] for choice in tool["request"]["questions"][0]["choices"]]
    assert answer.choice == "book_restaurant"
    assert sum(answer.probabilities.values()) == pytest.approx(1)
    assert (response.usage.requests, response.usage.input_tokens) == (1, tool["response"]["usage"]["input_tokens"])
    assert response.server_seconds == int(tool["headers"]["openai-processing-ms"]) / 1000
    assert response.raw["model"] == "gpt-6-luna"

    model, _ = serving(200, mixed["response"])
    response = await model.ask(canonical(mixed["request"]))
    assert response.answers["dinner"] == BinaryAnswer(probability=1.0)
    assert response.server_seconds is None  # no header in this reply

    model, _ = serving(200, score["response"])
    response = await model.ask(canonical(score["request"]))
    assert response.answers["urgency"] == ScoreAnswer(expected=1.0, probabilities=(0.0, 1.0, 0.0))


async def test_a_key_only_option_has_no_description_and_extra_body_reaches_the_top_level() -> None:
    model, requests = serving(
        200, answer_with([choice([("a", 0.5), ("b", 0.3), ("none", 0.2)])]), extra_body={"safety_identifier": "user-1"}
    )
    await model.ask(THREE, model="gpt-6-luna-next")
    payload = json.loads(requests[0].content)
    assert payload["questions"][0]["choices"] == [
        {"value": "a", "description": "Alpha."},
        {"value": "b"},
        {"value": "none", "description": "None."},
    ]
    assert (payload["safety_identifier"], payload["model"]) == ("user-1", "gpt-6-luna-next")


async def test_a_refused_question_raises_with_its_name_and_the_reply_usage() -> None:
    request = DecisionRequest(
        state="Request: a private message.",
        questions={
            "tool": ChoiceQuestion(instructions="Which?", options={"a": "A", "b": "B"}),
            "safe": BinaryQuestion(instructions="Is it safe?"),
        },
    )
    model, _ = serving(200, answer_with([{"type": "refusal", "name": "tool"}, {"type": "refusal", "name": None}]))
    with pytest.raises(DecisionRefused) as caught:
        await model.ask(request)
    assert caught.value.names == ("tool", "safe")
    assert caught.value.usage == DecisionUsage(1, 50, 0)
    assert str(caught.value).startswith("gpt-6-luna@api.openai.com: ")
    assert "private" not in str(caught.value)  # the state is never quoted


REPLIES_THAT_DO_NOT_ANSWER: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "an answer short": lambda: [],
    "an answer more": lambda: [choice([("a", 0.5), ("b", 0.3), ("none", 0.2)])] * 2,
    "another question's name": lambda: [choice([("a", 0.5), ("b", 0.3), ("none", 0.2)], name="other")],
    "another kind": lambda: [{"type": "predicate", "name": "tool", "probability": 0.5}],
    "a boolean value": lambda: [choice([("a", 0.5), (True, 0.3), ("none", 0.2)])],
    "a key missing": lambda: [choice([("a", 0.5), ("none", 0.5)])],
    "a key twice": lambda: [choice([("a", 0.5), ("a", 0.3), ("b", 0.1), ("none", 0.1)])],
    "a value not asked": lambda: [choice([("a", 0.5), ("b", 0.3), ("c", 0.2)])],
}


@pytest.mark.parametrize("case", REPLIES_THAT_DO_NOT_ANSWER)
async def test_a_reply_that_does_not_answer_the_request_is_an_error(case: str) -> None:
    model, _ = serving(200, answer_with(REPLIES_THAT_DO_NOT_ANSWER[case]()))
    with pytest.raises(DecisionError) as caught:
        await model.ask(THREE)
    assert not isinstance(caught.value, DecisionRefused)
    assert str(caught.value).startswith("gpt-6-luna@api.openai.com: ")


async def test_score_probabilities_must_cover_every_level_index() -> None:
    request = DecisionRequest(
        state="s", questions={"u": ScoreQuestion(instructions="How?", levels=("lo", "mid", "hi"))}
    )
    reply = answer_with(
        [{"type": "score", "name": "u", "score": 1.0, "confidence": 1.0, "probabilities": [
            {"value": 0, "label": "lo", "probability": 0.5}, {"value": 2, "label": "hi", "probability": 0.5}
        ]}]
    )  # fmt: skip
    model, _ = serving(200, reply)
    with pytest.raises(DecisionError, match="expected probabilities"):
        await model.ask(request)


async def test_the_recorded_error_reply_is_final_and_keeps_its_status() -> None:
    fixture = recorded("error")
    model, requests = serving(fixture["status"], fixture["response"], fixture["headers"])
    with pytest.raises(DecisionError, match="array_below_min_length") as caught:
        await model.ask(THREE)
    assert (caught.value.status, len(requests)) == (400, 1)
