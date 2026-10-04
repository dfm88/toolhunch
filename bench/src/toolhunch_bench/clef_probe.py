"""P2 Stop 0: what Cloudflare's Workers AI returns for Clef and Clef-flash, before an adapter relies on it.

Raw HTTP on purpose: the probe checks the facts the library adapter is then built on (the envelope, the `usage`
names, the option cap, the error shape, any version header). Each model gets a five-option tool choice, a
255-option choice (bisected down when refused) and a choice without options. Fixtures are saved scrubbed: the
account ID never leaves the run directory, which is git-ignored.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import httpx2
from dotenv import load_dotenv

from toolhunch_bench import BENCH_DIR
from toolhunch_bench.ledger import LEDGER_PATH, LedgerEntry, append_ledger

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["CLEF_PRICES", "UsageMissing", "clef_probe"]

CLEF_PRICES = {"clef": 0.24, "clef-flash": 0.09}
"""USD per million input tokens, from the Workers AI model pages (checked 2026-10-04); output is not billed."""

_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}"
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
_KEPT_HEADER = re.compile(r"content-type|.*(version|model|latency|timing|cf-ai).*", re.IGNORECASE)
_UNKNOWN_TOKENS_PER_REQUEST = 4_000  # priced when a reply has no usable usage: above the 255-option request's size


class UsageMissing(Exception):
    """A reply carried no `input_tokens` in its `usage`: the cost guard could not price it (spec §6.1)."""


def _body(model: str, options: dict[str, str] | None) -> dict[str, Any]:
    question: dict[str, Any] = {"type": "choice", "instructions": _INSTRUCTIONS}
    if options is not None:
        question["criteria"] = options
    return {"state": _STATE, "model": model, "questions": {"tool": question}}


def _many(count: int) -> dict[str, str]:
    return {f"opt_{index:03d}": f"Option {index:03d}" for index in range(count)}


def _object(value: object) -> dict[str, Any] | None:
    """`value` when it is a JSON object, else `None`."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _usage(body: object) -> dict[str, Any] | None:
    """The `usage` object, inside the `result` envelope or at the top level."""
    if (top := _object(body)) is not None:
        for holder in (_object(top.get("result")), top):
            if holder is not None and (usage := _object(holder.get("usage"))) is not None:
                return usage
    return None


class _Prober:
    def __init__(self, client: httpx2.AsyncClient, *, account: str, key: str) -> None:
        self._client, self._account, self._key = client, account, key
        self.exchanges: list[dict[str, Any]] = []

    def scrub(self, text: str) -> str:
        return text.replace(self._account, "ACCOUNT_ID").replace(self._key, "***")

    async def post(self, model: str, body: dict[str, Any], *, purpose: str) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self._client.post(
            _URL.format(account=self._account, model=model),
            json=body,
            headers={"Authorization": f"Bearer {self._key}"},
        )
        seconds = time.perf_counter() - started
        try:
            parsed: object = json.loads(self.scrub(response.text))
        except ValueError:
            parsed = self.scrub(response.text)[:2_000]
        top = _object(parsed)
        result = _object(top.get("result")) if top is not None else None
        criteria = _object(body["questions"]["tool"].get("criteria")) or {}
        exchange: dict[str, Any] = {
            "model": model,
            "purpose": purpose,
            "options": len(criteria),
            "status": response.status_code,
            "seconds": round(seconds, 4),
            "header_names": sorted(response.headers.keys()),
            "headers": {name: value for name, value in response.headers.items() if _KEPT_HEADER.fullmatch(name)},
            "envelope_keys": sorted(top) if top is not None else None,
            "result_keys": sorted(result) if result is not None else None,
            "model_answered": result.get("model") if result is not None else None,
            "usage": _usage(parsed),
            "body": parsed,
            "request": body,
        }
        self.exchanges.append(exchange)
        return exchange


def _input_tokens(exchange: dict[str, Any]) -> int | None:
    usage = _object(exchange["usage"])
    tokens: object = usage.get("input_tokens") if usage is not None else None
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else None


async def _option_cap(prober: _Prober, model: str) -> dict[str, Any]:
    """255 options if accepted; otherwise the largest accepted count, bisected in at most 8 requests."""
    first = await prober.post(model, _body(model, _many(255)), purpose="cap 255")
    if first["status"] < 400:
        return {"accepted_255": True, "cap": 255, "requests": 1}
    accepted, refused, requests = 2, 255, 1
    while refused - accepted > 1 and requests < 1 + _MAX_BISECTION_REQUESTS:
        middle = (accepted + refused) // 2
        exchange = await prober.post(model, _body(model, _many(middle)), purpose=f"cap {middle}")
        requests += 1
        if exchange["status"] >= 500:
            return {"accepted_255": False, "cap": None, "requests": requests, "stopped_on": exchange["status"]}
        if exchange["status"] < 400:
            accepted = middle
        else:
            refused = middle
    return {"accepted_255": False, "cap": accepted, "exact": refused - accepted == 1, "requests": requests}


def _fixture(exchange: dict[str, Any], *, recorded: str) -> dict[str, Any]:
    return {
        "source": f"api.cloudflare.com Workers AI, @cf/cloudflare/{exchange['model']}, recorded {recorded}",
        "request": exchange["request"],
        "status": exchange["status"],
        "headers": exchange["headers"],
        "response": exchange["body"],
    }


async def clef_probe(
    *,
    runs_dir: Path,
    fixtures_dir: Path,
    ledger_path: Path = LEDGER_PATH,
    client: httpx2.AsyncClient | None = None,
) -> Path:
    """Probe both Clef models, write `probe.json` and the scrubbed fixtures, and append the ledger line.

    Raises:
        ValueError: `CLOUDFLARE_ACCOUNT_ID` or `CLOUDFLARE_API_KEY` is not set. Nothing is sent.
        UsageMissing: a successful reply has no `usage.input_tokens`, after `probe.json` and the ledger line are
            written.
    """
    load_dotenv(BENCH_DIR.parent / ".env", override=False)
    account, key = os.environ.get("CLOUDFLARE_ACCOUNT_ID", ""), os.environ.get("CLOUDFLARE_API_KEY", "")
    if not account or not key:
        raise ValueError("clef-probe needs CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_KEY in the environment or .env")
    now = datetime.now(UTC)
    run_id = now.strftime("%Y%m%dT%H%M%SZ")
    run_dir = runs_dir / f"{run_id}-clef-probe"
    run_dir.mkdir(parents=True, exist_ok=False)
    owned = client is None
    http = client or httpx2.AsyncClient(timeout=60.0)
    prober = _Prober(http, account=account, key=key)
    findings: dict[str, Any] = {}
    try:
        for model in CLEF_PRICES:
            tool = await prober.post(model, _body(model, _OPTIONS), purpose="tool choice")
            cap = await _option_cap(prober, model)
            error = await prober.post(model, _body(model, None), purpose="no options")
            findings[model] = {
                "tool_choice": {key_: tool[key_] for key_ in ("status", "envelope_keys", "result_keys", "usage")},
                "response_model": tool["model_answered"],
                "headers": tool["headers"],
                "header_names": tool["header_names"],
                "option_cap": cap,
                "error": {"status": error["status"], "body": error["body"]},
            }
            if tool["status"] < 400:
                name = f"{model.replace('-', '_')}_tool_choice.json"
                (fixtures_dir / name).write_text(
                    json.dumps(_fixture(tool, recorded=now.date().isoformat()), indent=2) + "\n"
                )
            if model == "clef-flash":
                (fixtures_dir / "clef_error.json").write_text(
                    json.dumps(_fixture(error, recorded=now.date().isoformat()), indent=2) + "\n"
                )
    finally:
        if owned:
            await http.aclose()
        usd = 0.0
        tokens = 0
        unpriced = 0
        for exchange in prober.exchanges:
            counted = _input_tokens(exchange)
            if counted is None:
                unpriced += 1
                counted = _UNKNOWN_TOKENS_PER_REQUEST
            tokens += counted
            usd += counted * CLEF_PRICES[exchange["model"]] / 1_000_000
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
                    provider="cloudflare",
                    model="clef, clef-flash",
                    input_tokens=tokens,
                    usd=usd,
                    purpose="P2: Clef probe",
                    note=f"{len(prober.exchanges)} requests; {unpriced} without usable usage, priced at "
                    f"{_UNKNOWN_TOKENS_PER_REQUEST:,} tokens each (refused requests may not be billed)",
                ),
                path=ledger_path,
            )
    if any(_input_tokens(e) is None for e in prober.exchanges if e["status"] < 400):
        raise UsageMissing("STOP: Clef usage has no input_tokens (spec §6.1)")
    return run_dir
