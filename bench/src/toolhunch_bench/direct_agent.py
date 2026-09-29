"""Single function-tool requests with persistent replay and provider accounting."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, SystemPromptPart, TextPart, ToolCallPart, UserPromptPart
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
    text_is_none: bool = False


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
    """Canonical request content for replay keys and the conservative runtime guard reservation."""
    return {
        "model": AGENT_MODEL,
        "api": "openai-chat-completions",
        "prompt_version": AGENT_PROMPT_VERSION,
        "messages": [{"system": AGENT_INSTRUCTION}, {"user": query}],
        "tools": [asdict(tool) for tool in functions.tools],
        "settings": dict(AGENT_SETTINGS) | {"openai_prompt_cache_key": catalog_name},
        "allow_text_output": True,
    }


def wire_request(query: str, functions: FunctionCards) -> dict[str, Any]:
    """Prompt-bearing Chat Completions fields for definitions produced by `function_cards`."""
    return {
        "messages": [{"role": "system", "content": AGENT_INSTRUCTION}, {"role": "user", "content": query}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters_json_schema,
                },
            }
            for tool in functions.tools
        ],
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

    @staticmethod
    def _failure_message(error: Exception) -> str:
        if not isinstance(error, ModelHTTPError):
            return type(error).__name__
        # Provider fields are untrusted too. Only recognized diagnostic labels are ever retained; arbitrary strings
        # can contain secrets even when they are returned under `code` or `param` instead of `message`.
        raw_body = error.body
        body: Mapping[str, object] = cast("Mapping[str, object]", raw_body) if isinstance(raw_body, dict) else {}
        if isinstance(nested := body.get("error"), dict):
            body = cast("Mapping[str, object]", nested)
        codes = {
            "x",
            "invalid_request_error",
            "invalid_value",
            "array_above_max_length",
            "context_length_exceeded",
            "unsupported_parameter",
            "unsupported_value",
            "invalid_api_key",
            "rate_limit_exceeded",
            "insufficient_quota",
            "model_not_found",
            "server_error",
            "invalid_type",
            "missing_required_parameter",
        }
        params = {
            "tools",
            "messages",
            "model",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "parallel_tool_calls",
            "tool_choice",
            "prompt_cache_key",
            "response_format",
            "stream",
            "seed",
        }
        code, param = body.get("code"), body.get("param")
        safe_code = code if isinstance(code, str) and code in codes else "none" if code is None else "redacted"
        safe_param = param if isinstance(param, str) and param in params else "none" if param is None else "redacted"
        return f"ModelHTTPError {error.status_code} code={safe_code} param={safe_param}"

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
                diagnostic = self._failure_message(error)
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
                        error=diagnostic,
                    )
                )
                if attempt or (
                    isinstance(error, ModelHTTPError) and error.status_code not in {429, 500, 502, 503, 504}
                ):
                    raise ProviderFailure(diagnostic) from None
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
                    text_is_none=not calls
                    and "".join(part.content for part in response.parts if isinstance(part, TextPart)).strip()
                    == "none",
                )
                self.misses += 1
                self._db.execute("INSERT INTO agent_responses VALUES (?, ?)", (key, json.dumps(asdict(answer))))
                self._db.commit()
                return answer, False
        raise AssertionError("unreachable")
