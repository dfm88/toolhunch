"""P4 Stop 0: what OpenAI's Decisions API returns, before an adapter relies on it.

Raw HTTP on purpose, as P2's Clef probe: it checks the facts the library adapter is then built on (the reply's
shape, the `usage` names, the option and question caps, the error shape, the headers), which the guide does not
document. Fixtures are saved scrubbed: no key, no organization or project id, no request id.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import httpx2
from dotenv import load_dotenv

from toolhunch import DetailLevel, HeuristicTokenizer
from toolhunch_bench import BENCH_DIR
from toolhunch_bench.ledger import LEDGER_PATH, LedgerEntry, append_ledger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from toolhunch_bench.datasets.toolret import ToolRetData

__all__ = ["DECISIONS_PRICE", "UsageMissing", "decisions_probe"]

DECISIONS_PRICE = 0.10
"""USD per million input tokens (the Decisions guide, read 2026-10-07); output is not billed."""

_URL = "https://api.openai.com/v1/decisions"
_MODEL = "gpt-6-luna"
_OPTIONS = {
    "get_weather": "get_weather: Get the current weather and the forecast for a city.",
    "book_restaurant": "book_restaurant: Reserve a table at a restaurant for a given date, time and party size.",
    "order_food_delivery": "order_food_delivery: Order food for home delivery from a nearby restaurant.",
    "send_email": "send_email: Send an email to a recipient with a subject and a body.",
    "none": "None of these tools can fulfil the request.",
}
_STATE = "Request: Book me a table for two at an Italian place tonight at 8."
_INSTRUCTIONS = "Which tool should the assistant call next to fulfil the request?"
_MAX_BISECTION_REQUESTS = 8
_QUESTIONS_TRIED = 64
# Names only for everything else: the organization, the project and the request id never leave the run.
_KEPT_HEADER = re.compile(
    r"content-type|retry-after|openai-processing-ms|openai-version|x-ratelimit-.*|.*(version|model|latency|timing).*",
    re.IGNORECASE,
)
_SCRUBBED = re.compile(r"\b(org|proj)[-_][A-Za-z0-9]{6,}\b")
_UNKNOWN_TOKENS_PER_REQUEST = 40_000  # priced when a reply has no usable usage: above the largest request here


class UsageMissing(Exception):
    """A successful reply carried no `usage.input_tokens`: the cost guard could not price it (spec §5.1)."""


def _choice(name: str, options: Mapping[str, str]) -> dict[str, Any]:
    choices = [{"value": key, "description": text} if text else {"value": key} for key, text in options.items()]
    return {"type": "choice", "name": name, "instructions": _INSTRUCTIONS, "choices": choices}


def _body(*questions: dict[str, Any], state: str = _STATE) -> dict[str, Any]:
    return {"model": _MODEL, "input": state, "questions": list(questions)}


def _many(count: int) -> dict[str, str]:
    return {f"opt_{index:03d}": f"Option {index:03d}" for index in range(count)}


def _predicates(count: int) -> list[dict[str, Any]]:
    return [
        {"type": "predicate", "name": f"p{index:02d}", "instructions": f"Does the request mention the number {index}?"}
        for index in range(count)
    ]


def _object(value: object) -> dict[str, Any] | None:
    """`value` when it is a JSON object, else `None`."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _heuristic(body: Mapping[str, Any]) -> int:
    """The request's text as the bench's estimate counts it: the input, and each question's instructions and options."""
    count = HeuristicTokenizer().count
    tokens = count(body["input"])
    for question in body["questions"]:
        tokens += count(question["instructions"])
        for choice in question.get("choices", []):
            tokens += count(str(choice["value"])) + count(choice.get("description", ""))
        for level in question.get("levels", []):
            tokens += count(level["label"])
    return tokens


class _Prober:
    def __init__(self, client: httpx2.AsyncClient, *, key: str) -> None:
        self._client, self._key = client, key
        self.exchanges: list[dict[str, Any]] = []

    def scrub(self, text: str) -> str:
        return _SCRUBBED.sub(r"\1-SCRUBBED", text.replace(self._key, "***"))

    async def post(self, body: dict[str, Any], *, purpose: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._client.post(_URL, json=body, headers={"Authorization": f"Bearer {self._key}"})
        seconds = time.perf_counter() - started
        try:
            parsed: object = json.loads(self.scrub(response.text))
        except ValueError:
            parsed = self.scrub(response.text)[:2_000]
        top = _object(parsed)
        exchange: dict[str, Any] = {
            "purpose": purpose,
            "questions": len(body["questions"]),
            "options": [len(question.get("choices", [])) for question in body["questions"]],
            "heuristic_tokens": _heuristic(body),
            "status": response.status_code,
            "seconds": round(seconds, 4),
            "header_names": sorted(response.headers.keys()),
            "headers": {name: value for name, value in response.headers.items() if _KEPT_HEADER.fullmatch(name)},
            "reply_keys": sorted(top) if top is not None else None,
            "model_answered": top.get("model") if top is not None else None,
            "usage": _object(top.get("usage")) if top is not None else None,
            "body": parsed,
            "request": body,
        }
        self.exchanges.append(exchange)
        return exchange


def _input_tokens(exchange: Mapping[str, Any]) -> int | None:
    usage = _object(exchange["usage"])
    tokens: object = usage.get("input_tokens") if usage is not None else None
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


async def _cap(prober: _Prober, *, tried: int, build: Callable[[int], dict[str, Any]], label: str) -> dict[str, Any]:
    """`tried` if accepted; otherwise the largest accepted count, bisected in at most 8 requests."""
    first = await prober.post(build(tried), purpose=f"{label} {tried}")
    if first["status"] < 400:
        return {f"accepted_{tried}": True, "cap": tried, "requests": 1}
    accepted, refused, requests = 1, tried, 1
    while refused - accepted > 1 and requests < 1 + _MAX_BISECTION_REQUESTS:
        middle = (accepted + refused) // 2
        exchange = await prober.post(build(middle), purpose=f"{label} {middle}")
        requests += 1
        if exchange["status"] >= 500:
            return {f"accepted_{tried}": False, "cap": None, "requests": requests, "stopped_on": exchange["status"]}
        if exchange["status"] < 400:
            accepted = middle
        else:
            refused = middle
    return {
        f"accepted_{tried}": False,
        "cap": accepted,
        "exact": refused - accepted == 1,
        "requests": requests,
        "error": first["body"],
    }


def _distribution(exchange: Mapping[str, Any]) -> dict[str, float]:
    body = _object(exchange["body"]) or {}
    answers: object = body.get("answers")
    answer = _object(cast("list[object]", answers)[0]) if isinstance(answers, list) and answers else None
    entries: object = answer.get("probabilities") if answer is not None else None
    if not isinstance(entries, list):
        return {}
    distribution: dict[str, float] = {}
    for item in cast("list[object]", entries):
        if (entry := _object(item)) is not None and isinstance(probability := entry.get("probability"), int | float):
            distribution[str(entry.get("value"))] = float(probability)
    return distribution


def _overhead(small: Mapping[str, Any], large: Mapping[str, Any]) -> dict[str, Any] | None:
    """Billed tokens beyond the heuristic count, per request and per option, from two choices of different sizes."""
    billed_small, billed_large = _input_tokens(small), _input_tokens(large)
    if billed_small is None or billed_large is None:
        return None
    options_small, options_large = sum(small["options"]), sum(large["options"])
    extra_small = billed_small - small["heuristic_tokens"]
    extra_large = billed_large - large["heuristic_tokens"]
    per_option = (extra_large - extra_small) / (options_large - options_small)
    return {
        "billed": [billed_small, billed_large],
        "heuristic": [small["heuristic_tokens"], large["heuristic_tokens"]],
        "options": [options_small, options_large],
        "per_option": round(per_option, 2),
        "per_request": round(extra_small - per_option * options_small, 1),
    }


def _longest(data: ToolRetData) -> dict[str, str]:
    """ToolRet's longest name as an option key and its longest FULL card as that option's text, beside "none"."""
    cards = list(data.catalog)
    name = max((card.name for card in cards), key=len)
    text = max((card.render(DetailLevel.FULL) for card in cards), key=len)
    return {name: text, "none": _OPTIONS["none"]}


def _fixture(exchange: Mapping[str, Any], *, recorded: str) -> dict[str, Any]:
    return {
        "source": f"api.openai.com /v1/decisions, {_MODEL}, recorded {recorded}",
        "request": exchange["request"],
        "status": exchange["status"],
        "headers": exchange["headers"],
        "response": exchange["body"],
    }


async def decisions_probe(
    data: ToolRetData,
    *,
    runs_dir: Path,
    fixtures_dir: Path,
    ledger_path: Path = LEDGER_PATH,
    client: httpx2.AsyncClient | None = None,
) -> Path:
    """Probe the Decisions API, write `probe.json` and the scrubbed fixtures, and append the ledger line.

    Raises:
        ValueError: `OPENAI_API_KEY` is not set. Nothing is sent.
        UsageMissing: a successful reply has no `usage.input_tokens`, after `probe.json` and the ledger line are
            written.
    """
    load_dotenv(BENCH_DIR.parent / ".env", override=False)
    if not (key := os.environ.get("OPENAI_API_KEY", "")):
        raise ValueError("decisions-probe needs OPENAI_API_KEY in the environment or .env")
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{run_id}-decisions-probe"
    run_dir.mkdir(parents=True, exist_ok=False)
    owned = client is None
    http = client or httpx2.AsyncClient(timeout=120.0)
    prober = _Prober(http, key=key)
    findings: dict[str, Any] = {}
    recorded = now.date().isoformat()
    try:
        tool = await prober.post(_body(_choice("tool", _OPTIONS)), purpose="tool choice")
        findings["tool_choice"] = {
            name: tool[name] for name in ("status", "reply_keys", "model_answered", "usage", "headers", "header_names")
        }
        option_cap = await _cap(
            prober, tried=255, build=lambda count: _body(_choice("tool", _many(count))), label="options"
        )
        findings["option_cap"] = option_cap
        widest = _longest(data)
        longest = await prober.post(_body(_choice("tool", widest)), purpose="longest key and card")
        findings["longest"] = {
            "status": longest["status"],
            "key_characters": max(len(name) for name in widest),
            "text_characters": max(len(text) for text in widest.values()),
            "usage": longest["usage"],
            "error": longest["body"] if longest["status"] >= 400 else None,
        }
        mixed = await prober.post(
            _body(
                _choice("tool", _OPTIONS),
                {"type": "predicate", "name": "dinner", "instructions": "Is the request about dinner?"},
            ),
            purpose="choice and predicate",
        )
        score = await prober.post(
            _body(
                {
                    "type": "score",
                    "name": "urgency",
                    "instructions": "How urgent is the request?",
                    "levels": [{"label": "Not urgent"}, {"label": "Today"}, {"label": "Within the hour"}],
                }
            ),
            purpose="score",
        )
        findings["mixed"] = {"status": mixed["status"], "body": mixed["body"]}
        findings["score"] = {"status": score["status"], "body": score["body"]}
        findings["question_cap"] = await _cap(
            prober, tried=_QUESTIONS_TRIED, build=lambda count: _body(*_predicates(count)), label="questions"
        )
        error = await prober.post(_body(_choice("tool", {"only": "The only option."})), purpose="one option")
        findings["error"] = {"status": error["status"], "body": error["body"], "headers": error["headers"]}
        large = next((e for e in prober.exchanges if e["purpose"] == f"options {option_cap['cap']}"), None)
        findings["overhead"] = _overhead(tool, large) if large is not None and large["status"] < 400 else None
        repeats = [await prober.post(_body(_choice("tool", _OPTIONS)), purpose="repeat") for _ in range(3)]
        first = _distribution(tool)
        findings["repeats"] = {
            "max_abs_dp": max(
                (abs(_distribution(r).get(k, 0.0) - p) for r in repeats for k, p in first.items()), default=None
            ),
            "seconds": [tool["seconds"], *(r["seconds"] for r in repeats)],
            "median_seconds": statistics.median([tool["seconds"], *(r["seconds"] for r in repeats)]),
            "usage": [r["usage"] for r in repeats],
            "processing_ms": [r["headers"].get("openai-processing-ms") for r in [tool, *repeats]],
        }
        for exchange, name in ((tool, "tool_choice"), (mixed, "mixed"), (score, "score"), (error, "error")):
            (fixtures_dir / f"openai_decisions_{name}.json").write_text(
                json.dumps(_fixture(exchange, recorded=recorded), indent=2) + "\n"
            )
    finally:
        if owned:
            await http.aclose()
        tokens, unpriced = 0, 0
        for exchange in prober.exchanges:
            counted = _input_tokens(exchange)
            if counted is None:
                unpriced += 1
                counted = _UNKNOWN_TOKENS_PER_REQUEST
            tokens += counted
        summary = {
            "run_id": run_id,
            "checked": now.isoformat(),
            "requests": len(prober.exchanges),
            "findings": findings,
            "exchanges": [{k: v for k, v in e.items() if k != "request"} for e in prober.exchanges],
        }
        (run_dir / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
        if prober.exchanges:
            append_ledger(
                LedgerEntry(
                    timestamp=now,
                    run_id=run_id,
                    provider="openai",
                    model=f"{_MODEL} (decisions)",
                    input_tokens=tokens,
                    usd=tokens * DECISIONS_PRICE / 1_000_000,
                    purpose="P4: Decisions probe",
                    note=f"{len(prober.exchanges)} requests; {unpriced} without usable usage, priced at "
                    f"{_UNKNOWN_TOKENS_PER_REQUEST:,} tokens each (refused requests may not be billed)",
                ),
                path=ledger_path,
            )
    if any(_input_tokens(e) is None for e in prober.exchanges if e["status"] < 400):
        raise UsageMissing("STOP: the Decisions API's usage has no input_tokens (spec §5.1)")
    return run_dir
