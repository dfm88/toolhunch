"""The decider registry: every decider builds offline, and a decider billed per token is guarded and priced."""

import json
from collections.abc import Mapping

import httpx2
import pytest

from toolhunch.decision import (
    ChoiceQuestion,
    DecisionError,
    DecisionRefused,
    DecisionRequest,
    OpenAIDecisionModel,
    clef,
)
from toolhunch_bench.deciders import (
    DECIDERS,
    DeciderName,
    SerialDecisionModel,
    decision_model,
    local_provenance,
    missing_env,
)
from toolhunch_bench.direct_cost import GuardedDecisionModel, ProviderFailure, SpendGuard, SpendLimit

pytestmark = pytest.mark.anyio

ACCOUNT = "acct-0123456789abcdef"
OPTIONS = {"book_restaurant": "book_restaurant: Reserve a table.", "none": "None of these tools fits."}
REQUEST = DecisionRequest(
    state="Request: Book me a table for two tonight.",
    questions={"tool": ChoiceQuestion(instructions="Which tool should the assistant call next?", options=OPTIONS)},
)


def workers_ai(input_tokens: int, calls: list[httpx2.Request]) -> httpx2.AsyncClient:
    """A client that answers like Workers AI: the Jev-shaped answer inside a `result` envelope."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        answer = {"type": "choice", "choice": "book_restaurant", "probabilities": {"book_restaurant": 0.9, "none": 0.1}}
        result = {"model": "clef-flash", "answers": {"tool": answer}, "usage": {"input_tokens": input_tokens}}
        return httpx2.Response(200, json={"result": result, "success": True, "errors": [], "messages": []})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


@pytest.mark.usefixtures("laya_word_tokenizer")
async def test_registry_builds_and_guards_every_decider(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable, value in {
        "CLOUDFLARE_ACCOUNT_ID": ACCOUNT,
        "CLOUDFLARE_API_KEY": "cf-test-token",
        "CLM_BASE_URL": "https://clm.example",
    }.items():
        monkeypatch.setenv(variable, value)

    # Every name has an entry, and every entry builds without contacting anything; local servers are asked serially.
    assert set(DECIDERS) == set(DeciderName)
    for name in DeciderName:
        model = decision_model(name)
        assert model.model_id.startswith(DECIDERS[name].model)
        assert isinstance(model, SerialDecisionModel) == (DECIDERS[name].billing == "local")
        assert ACCOUNT not in model.model_id
        assert ACCOUNT not in repr(model)
        provenance = await local_provenance(name) if DECIDERS[name].local_url is None else DECIDERS[name].provenance
        assert ACCOUNT not in json.dumps(provenance)

    # A run checks the variables its deciders need, by name only.
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID")
    assert missing_env([DeciderName.CLEF, DeciderName.CLEF_FLASH, DeciderName.STRANDS]) == ["CLOUDFLARE_ACCOUNT_ID"]
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", ACCOUNT)

    # A Clef-flash reply is priced from its usage at the declared $0.09 per million input tokens.
    calls: list[httpx2.Request] = []
    guard = SpendGuard(prior_usd=0.0, cap_usd=8.0)
    model = GuardedDecisionModel(
        clef("clef-flash", http_client=workers_ai(1_000, calls)), guard=guard, provider="cloudflare"
    )
    await model.ask(REQUEST)
    [call] = guard.calls
    assert (call["provider"], call["input_tokens"]) == ("cloudflare", 1_000)
    assert call["usd"] == pytest.approx(1_000 * 0.09 / 1_000_000)

    # A priced reply without input tokens cannot be accounted for: a failure charged at its reservation, never $0,
    # and never retried, since the provider has billed the reply already.
    sent, unpriced_guard = len(calls), SpendGuard()
    unpriced = GuardedDecisionModel(
        clef("clef-flash", http_client=workers_ai(0, calls)), guard=unpriced_guard, provider="cloudflare", retries=1
    )
    with pytest.raises(ProviderFailure, match="without input tokens"):
        await unpriced.ask(REQUEST)
    assert len(calls) == sent + 1
    [failed] = unpriced_guard.calls
    assert failed["budget_charge_usd"] == pytest.approx(65_536 * 0.09 / 1_000_000)

    # An attempt that could pass the cap is refused before anything is sent.
    sent = len(calls)
    full = SpendGuard(prior_usd=8.0 - 1e-6, cap_usd=8.0)
    capped = GuardedDecisionModel(
        clef("clef-flash", http_client=workers_ai(1_000, calls)), guard=full, provider="cloudflare"
    )
    with pytest.raises(SpendLimit):
        await capped.ask(REQUEST)
    assert len(calls) == sent


def decisions_api(replies: list[httpx2.Response], calls: list[httpx2.Request]) -> httpx2.AsyncClient:
    """A client that answers each request with the next of `replies`, like OpenAI's Decisions API."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return replies.pop(0)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def decided(answer: Mapping[str, object], *, input_tokens: int = 2_000) -> httpx2.Response:
    usage = {"input_tokens": input_tokens, "output_tokens": 0}
    return httpx2.Response(200, json={"model": "gpt-6-luna", "answers": [answer], "usage": usage})


async def test_the_guard_retries_only_transient_failures_and_bills_a_refusal_at_its_usage() -> None:
    limits = decision_model(DeciderName.LUNA_DECISIONS).limits  # the registry's: the bench's 32,768-token cap
    reservation = 32_768 * 0.10 / 1_000_000
    pick = {"type": "choice", "name": "tool", "probabilities": [{"value": "book_restaurant", "probability": 0.9},
            {"value": "none", "probability": 0.1}], "choice": "book_restaurant", "confidence": 0.8}  # fmt: skip

    async def ask(
        *replies: httpx2.Response,
    ) -> tuple[SpendGuard, list[httpx2.Request], list[float], BaseException | None]:
        calls: list[httpx2.Request] = []
        waits: list[float] = []

        async def sleep(seconds: float) -> None:
            waits.append(seconds)

        guard = SpendGuard(prior_usd=0.0, cap_usd=3.0)
        client = decisions_api(list(replies), calls)
        inner = OpenAIDecisionModel(api_key="sk-test", limits=limits, http_client=client, max_retries=0)
        model = GuardedDecisionModel(inner, guard=guard, provider="openai", sleep=sleep)
        try:
            await model.ask(REQUEST)
        except DecisionError as error:
            return guard, calls, waits, error
        return guard, calls, waits, None

    # A 429 is retried after the wait it asked for; the answer is priced at the declared $0.10/M, and its model kept.
    guard, calls, waits, error = await ask(httpx2.Response(429, headers={"Retry-After": "2"}), decided(pick))
    assert (error, len(calls), waits) == (None, 2, [2.0])
    failed, answered = guard.calls
    assert (failed["status"], failed["budget_charge_usd"]) == (429, pytest.approx(reservation))
    assert (answered["usd"], answered["reply_model"]) == (pytest.approx(2_000 * 0.10 / 1_000_000), "gpt-6-luna")

    # A 400 would fail again: one attempt, charged at its reservation.
    guard, calls, waits, error = await ask(httpx2.Response(400, json={"error": {"message": "no"}}))
    assert (error is not None, len(calls), waits) == (True, 1, [])
    assert guard.calls[0]["budget_charge_usd"] == pytest.approx(reservation)

    # A refused question: the provider answered, so the attempt is charged at the usage it reported, not retried.
    guard, calls, waits, error = await ask(decided({"type": "refusal", "name": "tool"}, input_tokens=500))
    assert isinstance(error, DecisionRefused)
    assert (error.names, len(calls), waits) == (("tool",), 1, [])
    [refused] = guard.calls
    assert (refused["input_tokens"], refused["budget_charge_usd"]) == (500, pytest.approx(500 * 0.10 / 1_000_000))
