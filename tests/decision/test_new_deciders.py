"""Strands Decider and Cloudflare Clef: the factories over `JevWireModel`, against recorded replies."""

import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from toolhunch.cards import DetailLevel, ToolCard
from toolhunch.decision import (
    STRANDS_LIMITS,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    clef,
    strands_decider,
)
from toolhunch.decision.planner import RoundPlan, plan_rounds
from toolhunch.tokens import HeuristicTokenizer

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


def replay(name: str, requests: list[httpx2.Request]) -> tuple[dict[str, Any], httpx2.AsyncClient]:
    """The recorded exchange `name`, and a client that records each request and answers with its reply."""
    fixture = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(fixture["status"], json=fixture["response"])

    return fixture, httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_strands_decider_answers_within_its_window() -> None:
    requests: list[httpx2.Request] = []
    fixture, client = replay("strands_tool_choice", requests)
    model = strands_decider(http_client=client)

    response = await model.ask(REQUEST)

    [sent] = requests
    assert str(sent.url) == "http://127.0.0.1:8000/v1/systemone"
    assert "authorization" not in sent.headers
    assert json.loads(sent.content)["model"] == "strands-decider-2B-hobson-v19"
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == list(OPTIONS)
    assert answer.choice == fixture["response"]["answers"]["tool"]["choice"]
    assert response.usage.input_tokens == fixture["response"]["usage"]["input_tokens"]
    assert model.model_id == "strands-decider-2B-hobson-v19@127.0.0.1:8000"

    # The declared 4,096-token window bounds every planned question: the planner lowers card detail first, then
    # splits into two rounds when even the lowest detail does not fit.
    def catalog(count: int, *, name_length: int) -> list[ToolCard]:
        return [
            ToolCard(
                name=f"t{index:03d}_" + "x" * name_length, id=f"t{index:03d}", description="Looks up an item. " * 8
            )
            for index in range(count)
        ]

    def plan(cards: list[ToolCard]) -> RoundPlan:
        return plan_rounds(
            REQUEST.state,
            cards,
            reserved=True,
            limits=STRANDS_LIMITS,
            tokenizer=HeuristicTokenizer(),
            max_detail=DetailLevel.FULL,
            finalists_per_chunk=3,
        )

    lowered = plan(catalog(101, name_length=8))
    assert lowered.single is not None
    assert lowered.single.detail < DetailLevel.FULL
    assert lowered.single.estimated_input_tokens <= 4096

    split = plan(catalog(200, name_length=60))
    assert split.single is None
    assert len(split.chunks) >= 2
    assert all(chunk.question is None or chunk.question.estimated_input_tokens <= 4096 for chunk in split.chunks)


async def test_clef_reads_the_workers_ai_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    requests: list[httpx2.Request] = []
    fixture, client = replay("clef_flash_tool_choice", requests)
    model = clef("clef-flash", account_id="acc123", api_key="k", http_client=client)

    response = await model.ask(REQUEST)

    [sent] = requests
    assert str(sent.url) == "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/cloudflare/clef-flash"
    assert sent.headers["authorization"] == "Bearer k"
    assert json.loads(sent.content)["model"] == "clef-flash"
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == list(OPTIONS)
    assert response.usage.input_tokens == fixture["response"]["result"]["usage"]["input_tokens"]
    assert model.model_id == "clef-flash@api.cloudflare.com"
    assert "acc123" not in model.model_id
    assert "acc123" not in repr(model)
    assert model.limits.max_options_per_choice == 255

    # A refused request: the recorded HTTP 400 with Workers AI's error envelope.
    _, refusing = replay("clef_error", [])
    with pytest.raises(DecisionError, match="HTTP 400"):
        await clef("clef-flash", account_id="acc123", api_key="k", http_client=refusing, max_retries=0).ask(REQUEST)

    # HTTP 200 without a `result` object: named, with the envelope's errors, never a KeyError.
    def no_result(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"success": False, "errors": [{"code": 7003}], "messages": []})

    empty = httpx2.AsyncClient(transport=httpx2.MockTransport(no_result))
    with pytest.raises(DecisionError, match=r"no 'result' object.*7003"):
        await clef("clef-flash", account_id="acc123", api_key="k", http_client=empty).ask(REQUEST)

    monkeypatch.delenv("TOOLHUNCH_TEST_UNSET", raising=False)
    with pytest.raises(ValueError, match="TOOLHUNCH_TEST_UNSET"):
        clef("clef-flash", account_id_env="TOOLHUNCH_TEST_UNSET")
