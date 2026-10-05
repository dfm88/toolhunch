"""Laya and rizzo-flow: the factories over `JevWireModel`, against replies recorded from the real servers."""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from toolhunch import ScoredCard, ToolCard
from toolhunch.decision import (
    LAYA_LIMITS,
    RIZZO_FLOW_LIMITS,
    ChoiceAnswer,
    ChoiceDecider,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    check_request,
    laya,
    rizzo_flow,
)

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.anyio

FIXTURES = Path(__file__).parent / "fixtures"
OPTIONS = {
    "get_weather": "get_weather: Get the current weather and the forecast for a city.",
    "book_restaurant": "book_restaurant: Reserve a table at a restaurant for a given date, time and party size.",
    "order_food_delivery": "order_food_delivery: Order food for home delivery from a nearby restaurant.",
    "send_email": "send_email: Send an email to a recipient with a subject and a body.",
    "none": "None of these tools can fulfil the request.",
}
REQUEST = DecisionRequest(
    state="Request: Book me a table for two at an Italian place tonight at 8.",
    questions={
        "tool": ChoiceQuestion(
            instructions="Which tool should the assistant call next to fulfil the request?", options=OPTIONS
        )
    },
)


def fixture(name: str) -> dict[str, Any]:
    """The recorded exchange `name`: its `status`, `headers` and `body`."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def serve(replies: list[dict[str, Any]], requests: list[httpx2.Request]) -> httpx2.AsyncClient:
    """A client that records each request and answers with the next recorded reply."""
    queue = iter(replies)

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        reply = next(queue)
        return httpx2.Response(reply["status"], headers=reply["headers"], json=reply["body"])

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def request_with_options(count: int) -> DecisionRequest:
    options = {f"tool_{index}": f"tool_{index}: does thing {index}" for index in range(count)}
    return DecisionRequest(
        state=REQUEST.state, questions={"tool": ChoiceQuestion(instructions="Which tool?", options=options)}
    )


async def test_laya_sends_its_budgets_and_refuses_what_it_cut() -> None:
    no_routing = fixture("laya_tool_choice")
    del no_routing["body"]["routing"]
    requests: list[httpx2.Request] = []
    client = serve(
        [
            fixture("laya_wide_tool_choice"),
            fixture("laya_state_cut"),
            fixture("laya_collapsed_options"),
            fixture("laya_other_checkpoint"),
            no_routing,
        ],
        requests,
    )
    model = laya(head_max_len=512, max_len=1024, http_client=client)
    assert LAYA_LIMITS.max_question_tokens is not None
    assert LAYA_LIMITS.max_state_plus_question_tokens is not None
    question_margin = 192 - LAYA_LIMITS.max_question_tokens
    window_margin = 512 - LAYA_LIMITS.max_state_plus_question_tokens
    assert model.limits.max_question_tokens == 512 - question_margin
    assert model.limits.max_state_plus_question_tokens == 1024 - window_margin
    assert repr(model) == "JevWireModel(model_id='english@127.0.0.1:8010')"

    response = await model.ask(REQUEST)

    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert answer.choice in OPTIONS
    assert response.server_seconds == pytest.approx(0.02491)
    for match in ("cut the state: 189 of 576", "collapsed", "'multilingual'", "no routing"):
        with pytest.raises(DecisionError, match=match) as refused:
            await model.ask(REQUEST)
        assert str(refused.value).startswith("english@127.0.0.1:8010: ")
    for sent in requests:
        assert str(sent.url) == "http://127.0.0.1:8010/v1/systemone"
        assert "authorization" not in sent.headers
        body = json.loads(sent.content)
        assert (body["model"], body["head_max_len"], body["max_len"]) == ("english", 512, 1024)

    # What is sent is what the limits follow: the caller's `extra_body` beats the arguments, and without `strict`
    # the same cut reply decodes.
    lenient_requests: list[httpx2.Request] = []
    lenient = laya(
        strict=False,
        max_len=1024,
        extra_body={"max_len": 768, "head_max_len": 256},
        http_client=serve([fixture("laya_state_cut")], lenient_requests),
    )
    cut = (await lenient.ask(REQUEST)).answers["tool"]
    assert isinstance(cut, ChoiceAnswer)
    assert cut.choice == "book_restaurant"
    lenient_body = json.loads(lenient_requests[0].content)
    assert (lenient_body["head_max_len"], lenient_body["max_len"]) == (256, 768)
    assert lenient.limits.max_question_tokens == 256 - question_margin
    assert lenient.limits.max_state_plus_question_tokens == 768 - window_margin

    # The budgets are validated, and strict names only a checkpoint, not an alias that Laya answers under another name.
    invalid: list[tuple[str, Callable[[], object]]] = [
        ("head_max_len must be an integer of at least 1", lambda: laya(head_max_len=0)),
        ("max_len must be an integer of at least 1", lambda: laya(max_len=-5)),
        (r"head_max_len \(1024\) must be smaller than max_len \(512\)", lambda: laya(head_max_len=1024)),
        ("must be smaller than max_len", lambda: laya(head_max_len=512, max_len=512)),
        ("must be smaller than max_len", lambda: laya(max_len=100)),
        ("max_len must be an integer", lambda: laya(extra_body={"max_len": 1.5})),
    ]
    for message, build in invalid:
        with pytest.raises(ValueError, match=message):
            build()
    with pytest.raises(ValueError, match="pass strict=False to use an alias"):
        laya(model="laya")
    assert laya(model="laya", strict=False).model_id == "laya@127.0.0.1:8010"
    assert laya(model="multilingual").model_id == "multilingual@127.0.0.1:8010"


async def test_rizzo_flow_names_its_weights_and_holds_26_options() -> None:
    refusal = fixture("rizzo_flow_over_window")
    requests: list[httpx2.Request] = []
    model = rizzo_flow(http_client=serve([fixture("rizzo_flow_tool_choice"), refusal], requests), max_retries=0)
    assert model.limits is RIZZO_FLOW_LIMITS

    response = await model.ask(REQUEST)

    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == list(OPTIONS)
    assert response.server_seconds is None  # the server sends no latency header
    with pytest.raises(DecisionError) as raised:
        await model.ask(REQUEST)
    assert raised.value.status == refusal["status"]
    check_request(request_with_options(26), limits=model.limits, kinds=model.question_kinds)
    with pytest.raises(ValueError, match="27 options"):
        await model.ask(request_with_options(27))  # refused by the declared limits: nothing is sent
    assert len(requests) == 2
    for sent in requests:
        assert str(sent.url) == "http://127.0.0.1:8017/v1/systemone"
        assert "authorization" not in sent.headers
        assert json.loads(sent.content)["model"] == "rizzo-flow-4b-q8_0"


async def test_rizzo_flow_keeps_each_option_text_within_its_8000_characters() -> None:
    # A ToolRet card's full text ran past 8,000 characters and the server refused the question with HTTP 422.
    long = "Search property listings by city, state and postal code, with prices and photos. " * 120
    assert len(long) > 8000
    candidates = [
        ScoredCard(ToolCard(name=f"tool_{index}", description=long if index == 1 else f"Does thing {index}."), 0.5)
        for index in range(3)
    ]
    template = fixture("rizzo_flow_tool_choice")["body"]
    sent: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        answers = {
            key: template["answers"]["tool"]
            | {"choice": next(iter(question["criteria"])), "probabilities": dict.fromkeys(question["criteria"], 0.25)}
            for key, question in body["questions"].items()
        }
        return httpx2.Response(200, json=template | {"answers": answers})

    model = rizzo_flow(http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)), max_retries=0)
    await ChoiceDecider(model).decide("Request: find a house in Austin", candidates)

    texts = [question["criteria"] for body in sent for question in body["questions"].values()]
    assert all(text["tool_1"] for text in texts)  # shown at a lower detail, not dropped
    assert max(len(text) for criteria in texts for text in criteria.values()) <= 8000
