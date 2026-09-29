import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

from toolhunch.decision import (
    CLM_LIMITS,
    JEV_LIMITS,
    BinaryAnswer,
    BinaryQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionModel,
    DecisionRequest,
    DecisionUsage,
    JevWireModel,
    ScoreAnswer,
    ScoreQuestion,
    clm,
    jev,
)

pytestmark = pytest.mark.anyio

SECRET = "sk-test-not-a-real-key"
FIXTURES = Path(__file__).parent / "fixtures"
JEV_ID = "jev-latest@api.typesafe.ai"

DOCS_REQUEST = DecisionRequest(
    state=(
        "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. "
        "I'm losing sales. Please help ASAP."
    ),
    questions={
        "department": ChoiceQuestion(
            instructions="Which team should handle this",
            options={
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        ),
        "frustration": ScoreQuestion(
            instructions="How frustrated the customer appears",
            levels=("Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"),
        ),
        "is_urgent": BinaryQuestion(instructions="The message conveys urgency or time-sensitivity"),
    },
)

STATE = "Request: Book me a table for two at an Italian place tonight at 8."

TOOL_REQUEST = DecisionRequest(
    state=STATE,
    questions={
        "tool": ChoiceQuestion(
            instructions="Which tool should the assistant call next to fulfil the request?",
            options={
                "get_weather": "get_weather: Get the current weather and the forecast for a city.",
                "book_restaurant": (
                    "book_restaurant: Reserve a table at a restaurant for a given date, time and party size."
                ),
                "order_food_delivery": "order_food_delivery: Order food for home delivery from a nearby restaurant.",
                "send_email": "send_email: Send an email to a recipient with a subject and a body.",
                "none": "None of these tools can fulfil the request.",
            },
        )
    },
)

TWO_REQUEST = DecisionRequest(
    state=STATE,
    questions={
        "q1": BinaryQuestion(instructions="Is this about food?"),
        "q2": ScoreQuestion(instructions="How urgent is it?", levels=("not urgent", "urgent", "very urgent")),
    },
)


def load(name: str) -> dict[str, Any]:
    """A recorded exchange: `{source, request, status, headers?, response}`."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def replay(fixture: dict[str, Any], requests: list[httpx2.Request]) -> httpx2.AsyncClient:
    """A client that records every request and answers it with the fixture's status, headers and response."""
    # A recorded header can be null, for a server that did not set it: leave it out.
    headers = {"content-type": "application/json"}
    headers |= {name: value for name, value in fixture.get("headers", {}).items() if value is not None}
    # json.dumps rather than json=, because httpx2 refuses to encode the NaN some cases below need.
    content = json.dumps(fixture["response"])

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(fixture["status"], content=content, headers=headers)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def preset(kind: str, fixture: dict[str, Any], requests: list[httpx2.Request], **settings: Any) -> JevWireModel:
    """The `jev` or `clm` preset asking for the model the fixture recorded, over a client that replays it."""
    settings = {"api_key": SECRET, "http_client": replay(fixture, requests)} | settings
    model = fixture["request"]["model"]
    return clm("https://clm.example", model=model, **settings) if kind == "clm" else jev(model, **settings)


async def test_encoding_reproduces_the_typesafe_docs_request() -> None:
    fixture = load("typesafe_docs_quickstart")
    requests: list[httpx2.Request] = []
    model = jev(api_key=SECRET, http_client=replay(fixture, requests))
    response = await model.ask(DOCS_REQUEST)
    (request,) = requests
    body = json.loads(request.content)
    assert body == fixture["request"]
    assert list(body["questions"]["department"]["criteria"]) == ["billing", "technical", "sales"]  # option order
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {SECRET}"
    assert response.answers == {
        "department": ChoiceAnswer(probabilities={"billing": 0.159, "technical": 0.84, "sales": 0.001}),
        "frustration": ScoreAnswer(expected=1.035),
        "is_urgent": BinaryAnswer(probability=0.999),
    }
    assert response.usage == DecisionUsage(requests=1, input_tokens=312, output_tokens=48)
    assert response.raw == fixture["response"]
    assert response.server_seconds is None  # this recording has no latency header


@pytest.mark.parametrize(
    ("kind", "name", "header"),
    [("jev", "jev_tool_choice", "x-envoy-upstream-service-time"), ("clm", "clm_tool_choice", "x-clm-latency-ms")],
)
async def test_recorded_tool_choices_decode_in_option_order(kind: str, name: str, header: str) -> None:
    fixture = load(name)
    requests: list[httpx2.Request] = []
    response = await preset(kind, fixture, requests).ask(TOOL_REQUEST)
    assert json.loads(requests[0].content) == fixture["request"]
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == list(fixture["request"]["questions"]["tool"]["criteria"])
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.choice == "book_restaurant"
    assert response.server_seconds == pytest.approx(float(fixture["headers"][header]) / 1000)
    usage = fixture["response"]["usage"]
    assert response.usage == DecisionUsage(
        requests=1, input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"]
    )


async def test_a_rounded_shuffled_distribution_is_decoded_in_option_order_and_renormalised() -> None:
    fixture = load("jev_tool_choice")
    # How Jev rounds: two decimals, 0.99 in total, exact zeros, the keys in no particular order.
    fixture["response"]["answers"]["tool"]["probabilities"] = {
        "send_email": 0.0,
        "order_food_delivery": 0.05,
        "none": 0.0,
        "book_restaurant": 0.93,
        "get_weather": 0.01,
    }
    response = await preset("jev", fixture, []).ask(TOOL_REQUEST)
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == ["get_weather", "book_restaurant", "order_food_delivery", "send_email", "none"]
    assert list(answer.probabilities.values()) == pytest.approx([0.01 / 0.99, 0.93 / 0.99, 0.05 / 0.99, 0.0, 0.0])
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.choice == "book_restaurant"


async def test_recorded_score_and_noul_answers() -> None:
    fixture = load("jev_two_questions")
    response = await preset("jev", fixture, []).ask(TWO_REQUEST)
    assert response.answers == {
        "q1": BinaryAnswer(probability=0.97),
        "q2": ScoreAnswer(expected=1.16, probabilities=(0.06, 0.72, 0.22)),
    }
    assert response.usage == DecisionUsage(requests=1, input_tokens=326, output_tokens=35)
    assert response.server_seconds is None  # Jev's latency header is not in this recording

    fixture = load("clm_two_questions")
    response = await preset("clm", fixture, []).ask(TWO_REQUEST)
    served = fixture["response"]["answers"]
    q1, q2 = response.answers["q1"], response.answers["q2"]
    assert isinstance(q1, BinaryAnswer)
    assert q1.probability == served["q1"]["noul"]
    assert isinstance(q2, ScoreAnswer)
    assert q2.expected == served["q2"]["score"]
    assert q2.probabilities == pytest.approx(tuple(served["q2"]["probabilities"][str(level)] for level in range(3)))
    assert q2.probabilities is not None
    assert sum(q2.probabilities) == pytest.approx(1.0)
    assert response.usage == DecisionUsage(requests=1, input_tokens=75, output_tokens=0)  # billing_units is ignored
    assert response.server_seconds == pytest.approx(0.287)


async def test_score_probabilities_follow_level_order_whatever_the_key_order() -> None:
    fixture = load("clm_two_questions")
    fixture["response"]["answers"]["q2"]["probabilities"] = {"2": 0.5, "0": 0.3, "1": 0.2}
    response = await preset("clm", fixture, []).ask(TWO_REQUEST)
    q2 = response.answers["q2"]
    assert isinstance(q2, ScoreAnswer)
    assert q2.probabilities == pytest.approx((0.3, 0.2, 0.5))


@pytest.mark.parametrize(
    ("usage", "expected"),
    [({"input_tokens": 7}, DecisionUsage(1, 7, 0)), ({}, DecisionUsage(1, 0, 0)), (None, DecisionUsage(1, 0, 0))],
    ids=["one count", "empty", "absent"],
)
async def test_usage_counts_that_are_absent_default_to_zero(
    usage: dict[str, int] | None, expected: DecisionUsage
) -> None:
    fixture = load("jev_tool_choice")
    if usage is None:
        del fixture["response"]["usage"]
    else:
        fixture["response"]["usage"] = usage
    assert (await preset("jev", fixture, []).ask(TOOL_REQUEST)).usage == expected


@pytest.mark.parametrize("value", ["soon", "", "-5", "nan", "inf"])
async def test_an_unusable_latency_header_only_leaves_server_seconds_unset(value: str) -> None:
    fixture = load("clm_tool_choice")
    fixture["headers"]["x-clm-latency-ms"] = value
    response = await preset("clm", fixture, []).ask(TOOL_REQUEST)
    assert response.server_seconds is None
    assert isinstance(response.answers["tool"], ChoiceAnswer)  # the answer itself is still delivered


type Edit = Callable[[dict[str, Any]], object]
"""Breaks a recorded response in place."""

CHOICE_BREAKS: dict[str, tuple[Edit, str]] = {  # each with what the error must say
    "missing answer": (lambda r: r["answers"].pop("tool"), "no answer for question 'tool'"),
    "unasked answer": (lambda r: r["answers"].update(ghost={"type": "noul", "noul": 0.5}), "not asked"),
    "wrong type": (lambda r: r["answers"]["tool"].update(type="noul"), "type 'noul'"),
    "no type": (lambda r: r["answers"]["tool"].pop("type"), "type None"),
    "extra key": (lambda r: r["answers"]["tool"]["probabilities"].update(teleport=0.0), "teleport"),
    "missing key": (lambda r: r["answers"]["tool"]["probabilities"].pop("none"), "expected probabilities"),
    "nan": (lambda r: r["answers"]["tool"]["probabilities"].update(get_weather=math.nan), "not finite"),
    "negative": (lambda r: r["answers"]["tool"]["probabilities"].update(get_weather=-0.5), "non-negative"),
    "text": (lambda r: r["answers"]["tool"]["probabilities"].update(get_weather="0.5"), "not a number"),
    "all zero": (
        lambda r: r["answers"]["tool"].update(probabilities=dict.fromkeys(r["answers"]["tool"]["probabilities"], 0.0)),
        "positive",
    ),
    "no distribution": (lambda r: r["answers"]["tool"].pop("probabilities"), "`probabilities`"),
    "distribution as a list": (lambda r: r["answers"]["tool"].update(probabilities=[0.1, 0.9]), "`probabilities`"),
    "answers as a list": (lambda r: r.update(answers=[]), "answers"),
    "no answers": (lambda r: r.pop("answers"), "answers"),
    "usage as text": (lambda r: r["usage"].update(input_tokens="many"), "usage.input_tokens"),
}

TWO_BREAKS: dict[str, tuple[Edit, str]] = {
    "noul above one": (lambda r: r["answers"]["q1"].update(noul=1.5), "`noul`"),
    "noul below zero": (lambda r: r["answers"]["q1"].update(noul=-0.1), "`noul`"),
    "noul as text": (lambda r: r["answers"]["q1"].update(noul="0.97"), "`noul`"),
    "noul as a bool": (lambda r: r["answers"]["q1"].update(noul=True), "`noul`"),
    "noul nan": (lambda r: r["answers"]["q1"].update(noul=math.nan), "`noul`"),
    "no noul": (lambda r: r["answers"]["q1"].pop("noul"), "`noul`"),
    "score as text": (lambda r: r["answers"]["q2"].update(score="1.16"), "`score`"),
    "score nan": (lambda r: r["answers"]["q2"].update(score=math.nan), "`score`"),
    "no score": (lambda r: r["answers"]["q2"].pop("score"), "`score`"),
    "levels missing": (
        lambda r: r["answers"]["q2"].update(probabilities={"0": 0.5, "1": 0.5}),
        "expected probabilities",
    ),
    "levels negative": (
        lambda r: r["answers"]["q2"].update(probabilities={"0": -0.1, "1": 0.8, "2": 0.3}),
        "non-negative",
    ),
    "levels as a list": (lambda r: r["answers"]["q2"].update(probabilities=[0.1, 0.8, 0.1]), "`probabilities`"),
    "score answered as choice": (lambda r: r["answers"]["q2"].update(type="choice"), "type 'choice'"),
}


BREAKS = [
    *(
        pytest.param("jev_tool_choice", TOOL_REQUEST, edit, said, id=label)
        for label, (edit, said) in CHOICE_BREAKS.items()
    ),
    *(
        pytest.param("jev_two_questions", TWO_REQUEST, edit, said, id=label)
        for label, (edit, said) in TWO_BREAKS.items()
    ),
]


@pytest.mark.parametrize(("name", "asked", "edit", "must_say"), BREAKS)
async def test_unusable_answers_raise_with_the_model_id(
    name: str, asked: DecisionRequest, edit: Edit, must_say: str
) -> None:
    fixture = load(name)
    edit(fixture["response"])
    with pytest.raises(DecisionError) as error:
        await jev(api_key=SECRET, http_client=replay(fixture, [])).ask(asked)
    assert str(error.value).startswith(f"{JEV_ID}: ")
    assert must_say in str(error.value)


async def test_identity_comes_from_configuration() -> None:
    fixture = load("jev_two_questions")  # asks for jev-latest, and the response says jev-1.13.0
    model: DecisionModel = jev(api_key=SECRET, http_client=replay(fixture, []))
    response = await model.ask(TWO_REQUEST)
    assert model.model_id == JEV_ID
    assert response.raw["model"] == "jev-1.13.0"
    assert response.raw["answers"]["q2"]["confidence"] == 0.59
    assert "confidence" not in repr((response.answers, response.usage))  # the server's own fields live in raw only
    assert (model.limits, model.prompt_version) == (JEV_LIMITS, None)
    assert model.question_kinds == {"choice", "binary", "score"}
    assert SECRET not in repr(model)

    other = clm("http://localhost:8000/", api_key=SECRET)
    assert (other.model_id, other.limits) == ("clm-latest@localhost:8000", CLM_LIMITS)


async def test_a_server_without_auth_is_reached_through_the_plain_constructor() -> None:
    requests: list[httpx2.Request] = []
    model = JevWireModel(
        "my-model",
        base_url="http://localhost:8000/v1/",
        api_key_env=None,
        limits=CLM_LIMITS,
        http_client=replay(load("clm_tool_choice"), requests),  # the recording carries a latency header
    )
    response = await model.ask(TOOL_REQUEST)
    assert model.model_id == "my-model@localhost:8000"
    assert str(requests[0].url) == "http://localhost:8000/v1/systemone"
    assert "authorization" not in requests[0].headers
    assert response.server_seconds is None  # no latency_header was configured


@pytest.mark.parametrize("base_url", ["localhost:8000/v1", "/v1", "https://", "ftp://h.example/v1", ""])
def test_a_base_url_that_is_not_an_absolute_http_url_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        JevWireModel("m", base_url=base_url, api_key_env=None, limits=CLM_LIMITS)


@pytest.mark.parametrize(
    ("base_url", "must_say"),
    [
        ("https://user:hunter2@api.example.com/v1", "pass the key as api_key"),
        ("https://hunter2@api.example.com/v1", "pass the key as api_key"),  # a token as the user name
        ("user:hunter2@localhost:8000/v1", "absolute http(s) URL"),  # no scheme: parsed as a path, and still not quoted
    ],
    ids=["user and password", "user only", "no scheme"],
)
def test_a_base_url_with_credentials_is_refused_and_never_quoted(base_url: str, must_say: str) -> None:
    # A secret in the URL would reach repr, the model id, threshold keys and manifests: it must not get that far.
    with pytest.raises(ValueError, match="base_url") as error:
        JevWireModel("m", base_url=base_url, api_key_env=None, limits=CLM_LIMITS)
    assert must_say in str(error.value)
    assert "hunter2" not in str(error.value)
    with pytest.raises(ValueError, match="base_url") as from_preset:  # the presets go through the same check
        clm(base_url)
    assert "hunter2" not in str(from_preset.value)


async def test_extra_body_and_options_merge_last_write_wins() -> None:
    requests: list[httpx2.Request] = []
    model = clm(
        "https://clm.example/",
        api_key=SECRET,
        extra_body={"temperature": 2.0, "model": "clm-raw"},
        http_client=replay(load("clm_tool_choice"), requests),
    )
    await model.ask(TOOL_REQUEST, temperature=0.5)
    await model.ask(TOOL_REQUEST)
    first, second = (json.loads(request.content) for request in requests)
    assert (first["temperature"], first["model"]) == (0.5, "clm-raw")
    assert {"state", "questions"} <= first.keys()
    assert str(requests[0].url) == "https://clm.example/v1/systemone"
    assert second["temperature"] == 2.0  # an option holds for one call; the next one falls back to extra_body
    assert model.model_id == "clm-latest@clm.example"  # the identity stays with the configuration


async def test_requests_beyond_declared_limits_are_never_sent() -> None:
    requests: list[httpx2.Request] = []
    client = replay(load("jev_tool_choice"), requests)
    options = {f"tool_{index}": "text" for index in range(256)}
    too_many = DecisionRequest(state="s", questions={"q": ChoiceQuestion(instructions="which", options=options)})
    with pytest.raises(ValueError, match="256 options"):
        await jev(api_key=SECRET, http_client=client).ask(too_many)
    one_at_a_time = JEV_LIMITS.model_copy(update={"max_questions_per_request": 1})  # limits are per instance
    with pytest.raises(ValueError, match="2 questions"):
        await jev(api_key=SECRET, limits=one_at_a_time, http_client=client).ask(TWO_REQUEST)
    assert requests == []


async def test_an_empty_option_text_is_sent_as_null() -> None:
    requests: list[httpx2.Request] = []
    answered = {
        "status": 200,
        "response": {"answers": {"q": {"type": "choice", "probabilities": {"a": 0.4, "b": 0.6}}}},
    }
    request = DecisionRequest(
        state="s", questions={"q": ChoiceQuestion(instructions="which", options={"a": "", "b": "text"})}
    )
    response = await jev(api_key=SECRET, http_client=replay(answered, requests)).ask(request)
    assert json.loads(requests[0].content)["questions"]["q"]["criteria"] == {"a": None, "b": "text"}
    assert response.answers["q"] == ChoiceAnswer(probabilities={"a": 0.4, "b": 0.6})


async def test_a_final_client_error_names_the_model_and_the_servers_reason() -> None:
    fixture = load("clm_unknown_model")  # a CLM server asked for a model it does not have: HTTP 422
    requests: list[httpx2.Request] = []
    with pytest.raises(DecisionError) as error:
        await preset("clm", fixture, requests).ask(TOOL_REQUEST)
    assert json.loads(requests[0].content) == fixture["request"]
    assert str(error.value).startswith("jev-latest@clm.example: HTTP 422: ")
    assert "unknown model 'jev-latest'" in str(error.value)
    assert len(requests) == 1  # a 4xx is final: no retry


@pytest.mark.parametrize(
    ("kind", "name", "variable"),
    [("jev", "jev_tool_choice", "TYPESAFE_API_KEY"), ("clm", "clm_tool_choice", "CLM_API_KEY")],
)
async def test_presets_read_their_key_from_the_environment_at_call_time(
    monkeypatch: pytest.MonkeyPatch, kind: str, name: str, variable: str
) -> None:
    requests: list[httpx2.Request] = []
    model = preset(kind, load(name), requests, api_key=None)
    monkeypatch.delenv(variable, raising=False)
    with pytest.raises(DecisionError, match=variable):
        await model.ask(TOOL_REQUEST)
    assert requests == []
    monkeypatch.setenv(variable, SECRET)  # set after construction: the key is read when a call is made
    await model.ask(TOOL_REQUEST)
    assert requests[0].headers["authorization"] == f"Bearer {SECRET}"
    assert SECRET not in repr(model)


async def test_timeout_and_retries_reach_the_client_and_aclose_closes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[httpx2.Request] = []
    created: list[httpx2.AsyncClient] = []
    real_client = httpx2.AsyncClient

    def busy(request: httpx2.Request) -> httpx2.Response:
        attempts.append(request)
        return httpx2.Response(503, headers={"Retry-After": "0"})

    def factory(**kwargs: Any) -> httpx2.AsyncClient:
        created.append(real_client(transport=httpx2.MockTransport(busy), **kwargs))
        return created[-1]

    monkeypatch.setattr(httpx2, "AsyncClient", factory)
    model = clm("https://clm.example", api_key=SECRET, timeout=7.0, max_retries=1)
    with pytest.raises(DecisionError, match="HTTP 503"):
        await model.ask(TOOL_REQUEST)
    assert len(attempts) == 2  # one try and one retry
    (client,) = created
    assert client.timeout == httpx2.Timeout(7.0)
    await model.aclose()
    assert client.is_closed
