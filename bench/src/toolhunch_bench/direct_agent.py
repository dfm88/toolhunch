"""Single function-tool requests with persistent replay and provider accounting."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, SystemPromptPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from toolhunch_bench.direct_cost import AGENT_MODEL, ProviderCall, ProviderFailure, SpendGuard, openai_usd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pydantic_ai.models import Model
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    from toolhunch import ToolCard

AGENT_PROMPT_VERSION = "direct-choice-v1"
AGENT_INSTRUCTION = "Call the one tool that serves the request. If no tool does, reply `none` without calling a tool."
AGENT_MAX_OUTPUT_TOKENS = 256
AGENT_SETTINGS: OpenAIChatModelSettings = {
    "parallel_tool_calls": False,
    "temperature": 0,
    "max_tokens": AGENT_MAX_OUTPUT_TOKENS,
    "timeout": 60,
}


@dataclass(frozen=True, slots=True)
class AgentAnswer:
    """Parsed selection and historical provider usage; replay status is recorded separately."""

    pick: str | None
    extra_calls: int
    input_tokens: int
    cache_read_tokens: int
    output_tokens: int
    seconds: float


@dataclass(frozen=True, slots=True)
class FunctionCards:
    """Provider-safe function definitions and their reversible names."""

    tools: tuple[ToolDefinition, ...]
    card_ids: Mapping[str, str]


def function_cards(cards: Sequence[ToolCard]) -> FunctionCards:
    """Encode complete card descriptions, resolving provider-name collisions without losing card ids."""
    tools: list[ToolDefinition] = []
    card_ids: dict[str, str] = {}
    for card in cards:
        base = re.sub(r"[^a-zA-Z0-9_-]", "_", card.name)[:64] or "tool"
        name = base
        counter = 1
        while name in card_ids:
            counter += 1
            suffix = f"_{counter}"
            name = base[: 64 - len(suffix)] + suffix
        card_ids[name] = card.id
        parameters: Mapping[str, Any] = {} if card.parameters is None else card.parameters.get("properties", {})
        tools.append(
            ToolDefinition(
                name=name,
                description=json.dumps({"name": card.name, "description": card.description}, ensure_ascii=False),
                parameters_json_schema={
                    "type": "object",
                    "properties": {parameter: {"type": "string"} for parameter in parameters},
                },
                strict=False,
            )
        )
    return FunctionCards(tools=tuple(tools), card_ids=card_ids)


def agent_payload(query: str, functions: FunctionCards, *, catalog_name: str) -> dict[str, Any]:
    """Canonical request content used by both replay keys and cost estimates."""
    return {
        "model": AGENT_MODEL,
        "api": "openai-chat-completions",
        "prompt_version": AGENT_PROMPT_VERSION,
        "messages": [{"system": AGENT_INSTRUCTION}, {"user": query}],
        "tools": [asdict(tool) for tool in functions.tools],
        "settings": dict(AGENT_SETTINGS) | {"openai_prompt_cache_key": catalog_name},
        "allow_text_output": True,
    }


class CachedAgent:
    """Replay parsed responses from SQLite, sending missing requests through Pydantic AI directly.

    SDK retries must be disabled on the supplied model. One explicit transient retry here is separately guarded.
    Keys cover snapshot, API, prompt, tools, messages and settings; response bodies and tool arguments are not stored.
    """

    def __init__(self, model: Model, *, path: Path, guard: SpendGuard) -> None:
        self._model, self._guard = model, guard
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS agent_responses (key TEXT PRIMARY KEY, response TEXT NOT NULL)")
        self.hits = self.misses = 0

    def close(self) -> None:
        """Close the replay database, leaving the supplied model owned by the caller."""
        self._db.close()

    async def ask(self, query: str, functions: FunctionCards, *, catalog_name: str) -> tuple[AgentAnswer, bool]:
        """Return a parsed answer and whether it came from local replay."""
        payload = agent_payload(query, functions, catalog_name=catalog_name)
        payload["model"] = self._model.model_name
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        key = hashlib.sha256(canonical.encode()).hexdigest()
        row = self._db.execute("SELECT response FROM agent_responses WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return AgentAnswer(**json.loads(row[0])), True
        # Every byte can become a BPE token. Serialized ToolDefinition includes more fields than the wire, and the
        # additional envelope allowance covers the model's message/function framing.
        upper = openai_usd(
            model=AGENT_MODEL,
            input_tokens=len(canonical.encode()) + 1024,
            output_tokens=AGENT_MAX_OUTPUT_TOKENS,
        )
        settings: OpenAIChatModelSettings = {**AGENT_SETTINGS, "openai_prompt_cache_key": catalog_name}
        messages = [ModelRequest(parts=[SystemPromptPart(content=AGENT_INSTRUCTION), UserPromptPart(content=query)])]
        for attempt in range(2):
            self._guard.before(upper)
            started = time.perf_counter()
            try:
                response = await model_request(
                    self._model,
                    messages,
                    model_settings=settings,
                    model_request_parameters=ModelRequestParameters(
                        function_tools=list(functions.tools),
                        allow_text_output=True,
                    ),
                    instrument=False,
                )
            except Exception as error:
                self._guard.record(
                    ProviderCall(
                        provider="openai",
                        model=AGENT_MODEL,
                        input_tokens=None,
                        output_tokens=None,
                        cache_read_tokens=None,
                        seconds=time.perf_counter() - started,
                        usd=None,
                        list_usd=None,
                        budget_charge_usd=upper,
                        error=type(error).__name__,
                    )
                )
                if attempt or (
                    isinstance(error, ModelHTTPError) and error.status_code not in {429, 500, 502, 503, 504}
                ):
                    raise ProviderFailure(type(error).__name__) from None
            else:
                seconds = time.perf_counter() - started
                usage = response.usage
                usd = openai_usd(
                    model=AGENT_MODEL,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                )
                self._guard.record(
                    ProviderCall(
                        provider="openai",
                        model=AGENT_MODEL,
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cache_read_tokens=usage.cache_read_tokens,
                        seconds=seconds,
                        usd=usd,
                        list_usd=openai_usd(
                            model=AGENT_MODEL,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                        ),
                        budget_charge_usd=usd,
                    )
                )
                calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
                if calls and calls[0].tool_name not in functions.card_ids:
                    raise ProviderFailure("unknown tool name in agent response")
                answer = AgentAnswer(
                    pick=functions.card_ids[calls[0].tool_name] if calls else None,
                    extra_calls=max(0, len(calls) - 1),
                    input_tokens=usage.input_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    output_tokens=usage.output_tokens,
                    seconds=seconds,
                )
                self.misses += 1
                self._db.execute("INSERT INTO agent_responses VALUES (?, ?)", (key, json.dumps(asdict(answer))))
                self._db.commit()
                return answer, False
        raise AssertionError("unreachable")
