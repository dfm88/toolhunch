import pytest

from toolhunch.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionModel,
    DecisionRequest,
    DecisionResponse,
    OpenAILogprobModel,
    clm,
    jev,
)

pytestmark = [pytest.mark.anyio, pytest.mark.live]

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


async def check_a_tool_choice(model: DecisionModel) -> DecisionResponse:
    response = await model.ask(REQUEST)
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert list(answer.probabilities) == list(OPTIONS)
    assert response.usage.requests == 1
    return response


async def test_jev_answers_a_tool_choice(typesafe_api_key: str) -> None:
    model = jev("jev-1.13.0")  # the key is read from TYPESAFE_API_KEY, which the fixture has loaded
    try:
        assert (await check_a_tool_choice(model)).server_seconds is not None
    finally:
        await model.aclose()


async def test_clm_answers_a_tool_choice(clm_base_url: str) -> None:
    model = clm(clm_base_url)  # the key is read from CLM_API_KEY
    try:
        assert (await check_a_tool_choice(model)).server_seconds is not None
    finally:
        await model.aclose()


async def test_logprob_model_answers_a_tool_choice(openai_api_key: str) -> None:
    model = OpenAILogprobModel("gpt-4.1-mini-2025-04-14")  # the key is read from OPENAI_API_KEY
    try:
        response = await check_a_tool_choice(model)
        answer = response.answers["tool"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice == "book_restaurant"  # the letter B: the mapping back to the option key holds
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens == 1  # max_tokens is 1
        assert response.server_seconds is None  # this adapter has no server-side timing
    finally:
        await model.aclose()
