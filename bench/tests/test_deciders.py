"""The decider registry: every decider builds offline, and a decider billed per token is guarded and priced."""

import json

import httpx2
import pytest

from toolhunch.decision import ChoiceQuestion, DecisionRequest, clef
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
