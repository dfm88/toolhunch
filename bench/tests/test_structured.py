import json
from typing import Any

import httpx2
import pytest

from toolhunch.decision import ChoiceAnswer, ChoiceQuestion, DecisionError, DecisionRequest
from toolhunch_bench.structured import StructuredChoiceModel

pytestmark = pytest.mark.anyio

REQUEST = DecisionRequest(
    state="Request: what's the weather in Milan?",
    questions={"tool": ChoiceQuestion(instructions="Which tool?", options={"mail": "send_email", "sky": "weather"})},
)


def replying(content: str, sent: list[dict[str, Any]]) -> httpx2.AsyncClient:
    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        body = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 6},
        }
        return httpx2.Response(200, json=body)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_the_letter_comes_back_as_a_one_hot_answer_with_reasoning_off() -> None:
    sent: list[dict[str, Any]] = []
    model = StructuredChoiceModel("gpt-6-luna", api_key="sk-test", http_client=replying('{"letter":"B"}', sent))

    response = await model.ask(REQUEST)

    [payload] = sent
    assert payload["reasoning_effort"] == "none"
    assert payload["max_completion_tokens"] == 16
    assert "max_tokens" not in payload
    assert "logprobs" not in payload
    assert payload["response_format"]["json_schema"]["schema"]["properties"]["letter"]["enum"] == ["A", "B"]
    answer = response.answers["tool"]
    assert isinstance(answer, ChoiceAnswer)
    assert dict(answer.probabilities) == {"mail": 0.0, "sky": 1.0}
    assert (response.usage.input_tokens, response.usage.output_tokens) == (120, 6)


@pytest.mark.parametrize("content", ['{"letter":"AB"}', '{"letter":"C"}', "B", '{"choice":"B"}'])
async def test_anything_but_one_of_the_letters_is_an_error(content: str) -> None:
    model = StructuredChoiceModel("gpt-6-luna", api_key="sk-test", http_client=replying(content, []))

    with pytest.raises(DecisionError, match=r"gpt-6-luna@api\.openai\.com"):
        await model.ask(REQUEST)
