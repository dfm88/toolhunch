"""The logprob prompt answered as structured output, for models whose logprobs cannot rank 20 options."""

from __future__ import annotations

import json
import string
from datetime import date
from typing import TYPE_CHECKING, Any, cast

from toolhunch.decision import (
    ChoiceQuestion,
    DecisionResponse,
    DecisionUsage,
    ModelLimits,
    OpenAILogprobModel,
    check_request,
    choice_answer,
)
from toolhunch.decision.logprobs import SYSTEM_PROMPT

if TYPE_CHECKING:
    from toolhunch.decision import DecisionRequest

STRUCTURED_LIMITS = ModelLimits(
    max_options_per_choice=26,
    max_questions_per_request=1,
    source="letters A-Z in a JSON-schema enum; probe 2026-09-30: gpt-6-luna returns logprobs only with "
    "reasoning_effort none, top_logprobs at most 5, and 1-2 alternatives in practice",
    checked=date(2026, 9, 30),
)
STRUCTURED_PROMPT_VERSION = "letters-v1-json-v1"
_MAX_COMPLETION_TOKENS = 16  # the provider's minimum; {"letter":"A"} takes about five


class StructuredChoiceModel(OpenAILogprobModel):
    """Ask the letter prompt of `OpenAILogprobModel` with reasoning off and read the letter from a JSON-schema answer.

    The chosen option gets probability 1 and every other option 0: the answer carries no confidence, so an
    abstention threshold has nothing to read and the options after the pick keep their presented order.
    """

    def __init__(self, model: str, **kwargs: Any) -> None:
        """Configure the endpoint as `OpenAILogprobModel` does, with `STRUCTURED_LIMITS`."""
        super().__init__(model, limits=STRUCTURED_LIMITS, **kwargs)

    @property
    def prompt_version(self) -> str:
        """`STRUCTURED_PROMPT_VERSION`: the letter prompt, answered as JSON."""
        return STRUCTURED_PROMPT_VERSION

    def __repr__(self) -> str:
        return f"StructuredChoiceModel(model={self._model!r}, base_url={self._base_url!r})"

    async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
        """Ask the one choice question of `request`; `DecisionError` when the answer is not one of its letters."""
        check_request(request, limits=self.limits, kinds=self.question_kinds)
        ((key, asked),) = request.questions.items()
        question = cast("ChoiceQuestion", asked)
        letters = tuple(string.ascii_uppercase[: len(question.options)])  # a tuple: "AB" is `in` "ABC"
        schema = {
            "type": "object",
            "properties": {"letter": {"type": "string", "enum": list(letters)}},
            "required": ["letter"],
            "additionalProperties": False,
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._user_message(request.state, question)},
            ],
            "max_completion_tokens": _MAX_COMPLETION_TOKENS,
            "temperature": 0,
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "option", "strict": True, "schema": schema},
            },
        }
        payload |= self._extra_body
        payload |= options
        reply = await self._poster.post(payload)
        body: Any = reply.body
        try:
            choice = body["choices"][0]
            content = choice["message"]["content"]
            usage: Any = body["usage"]
            input_tokens, output_tokens = int(usage["prompt_tokens"]), int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError):
            raise self._error("the response has no message content and token usage") from None
        if content is None:  # observed on gpt-6-luna: finish_reason "stop", an empty refusal, no content
            refusal = str(choice["message"].get("refusal"))[:40]
            raise self._error(f"no content: finish_reason {choice.get('finish_reason')!r}, refusal {refusal!r}")
        try:
            letter = json.loads(content)["letter"]
        except (KeyError, TypeError, ValueError):
            raise self._error("the content is not a JSON letter") from None
        if letter not in letters:
            raise self._error(f"the answer {str(letter)[:20]!r} is not one of the letters {letters[0]}-{letters[-1]}")
        answer = choice_answer(
            {option: float(own == letter) for own, option in zip(letters, question.options, strict=True)},
            keys=list(question.options),
            model_id=self.model_id,
        )
        return DecisionResponse(
            answers={key: answer},
            usage=DecisionUsage(requests=1, input_tokens=input_tokens, output_tokens=output_tokens),
            seconds=reply.seconds,
            raw=body,
        )
