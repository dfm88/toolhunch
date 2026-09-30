import importlib.util
import json
import os
import re
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlsplit

import httpx2
import pytest

from toolhunch import JevWireModel, ModelLimits
from toolhunch.decision import ChoiceQuestion, DecisionRequest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/jev_compatible_server.py"
SUMMARY = ROOT / "bench/results/2026-09-jev-compatible-smoke/summary.json"
PROFILES: list[dict[str, Any]] = json.loads(SUMMARY.read_text())["servers"]


@pytest.fixture
def compatible_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("jev_compatible_server_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example


@pytest.mark.anyio
@pytest.mark.parametrize("profile", PROFILES, ids=[profile["package"] for profile in PROFILES])
async def test_compatible_example_sends_keyless_wire_request_and_ranks_weather(
    compatible_example: ModuleType, profile: dict[str, Any]
) -> None:
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.method == "POST"
        assert str(request.url) == "http://127.0.0.1:18970/v1/systemone"
        assert "authorization" not in request.headers
        body = json.loads(request.content)
        assert body["model"] == profile["requested_model"]
        assert body["state"] == "What's the weather in Milan?"
        question = body["questions"]["tool"]
        assert question["type"] == "choice"
        assert list(question["criteria"]) == ["get_weather", "send_email", "none"]
        assert question["criteria"]["get_weather"] == "get_weather: Get the current weather in a city."
        return httpx2.Response(
            200,
            json={
                "answers": {
                    "tool": {
                        "type": "choice",
                        "probabilities": {"get_weather": 0.8, "send_email": 0.05, "none": 0.15},
                    }
                }
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond), trust_env=False) as client:
        model = JevWireModel(
            profile["requested_model"],
            base_url="http://127.0.0.1:18970/v1",
            api_key_env=None,
            limits=ModelLimits.model_validate(profile["limits"]),
            http_client=client,
        )
        decision = await compatible_example.rank_demo_tools(model)
    assert len(requests) == 1
    assert decision.ranked[0].card.name == "get_weather"
    assert decision.abstained is False
    assert decision.shape["reserved_option"] is True


@pytest.mark.anyio
@pytest.mark.parametrize("profile", PROFILES, ids=[profile["package"] for profile in PROFILES])
async def test_compatible_limits_reject_oversized_requests_before_http(
    compatible_example: ModuleType, profile: dict[str, Any]
) -> None:
    sent: list[httpx2.Request] = []

    def unexpected(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        raise AssertionError("A request beyond declared limits reached HTTP")

    limits = ModelLimits.model_validate(profile["limits"])
    assert limits.max_options_per_choice is not None
    assert limits.max_questions_per_request is not None
    ordinary = ChoiceQuestion(instructions="Pick", options={"a": "A", "b": "B"})
    oversized = ChoiceQuestion(
        instructions="Pick", options={f"o{i}": str(i) for i in range(limits.max_options_per_choice + 1)}
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(unexpected), trust_env=False) as client:
        model = JevWireModel(
            profile["requested_model"],
            base_url="http://127.0.0.1:18970/v1",
            api_key_env=None,
            limits=limits,
            http_client=client,
        )
        with pytest.raises(ValueError, match=r"options.*at most"):
            await model.ask(DecisionRequest(state="Milan", questions={"tool": oversized}))
        with pytest.raises(ValueError, match=r"questions.*at most"):
            await model.ask(
                DecisionRequest(
                    state="Milan", questions={f"q{i}": ordinary for i in range(limits.max_questions_per_request + 1)}
                )
            )
        token_limited = JevWireModel(
            profile["requested_model"],
            base_url="http://127.0.0.1:18970/v1",
            api_key_env=None,
            limits=limits.model_copy(update={"max_state_plus_question_tokens": 1}),
            http_client=client,
        )
        with pytest.raises(ValueError, match="budget"):
            await compatible_example.rank_demo_tools(token_limited)
    assert sent == []


def test_other_deciders_readme_block_matches_example_and_declared_smoke_limits(compatible_example: ModuleType) -> None:
    section = re.search(
        r"^## Other deciders\n(.*?)(?=^## |\Z)",
        (ROOT / "README.md").read_text(),
        flags=re.MULTILINE | re.DOTALL,
    )
    assert section is not None
    assert re.findall(r"^```python\n(.*?)^```", section.group(1), flags=re.MULTILINE | re.DOTALL) == [
        EXAMPLE.read_text()
    ]
    laya = next(profile for profile in PROFILES if profile["package"] == "laya")
    assert compatible_example.LAYA_LIMITS.model_dump(mode="json", exclude_none=True) == laya["limits"]


@pytest.mark.live
@pytest.mark.anyio
async def test_compatible_example_against_explicit_local_server(compatible_example: ModuleType) -> None:
    base_url = os.environ.get("TOOLHUNCH_LOCAL_BASE_URL")
    if not base_url:
        pytest.skip("Set TOOLHUNCH_LOCAL_BASE_URL to opt into the local server smoke")
    url = urlsplit(base_url)
    if url.scheme != "http" or url.hostname != "127.0.0.1" or url.username is not None or url.password is not None:
        pytest.fail("The live smoke accepts only an unauthenticated http://127.0.0.1 API root")
    package = os.environ.get("TOOLHUNCH_LOCAL_SERVER", "laya")
    profile = next((profile for profile in PROFILES if profile["package"] == package), None)
    if profile is None:
        pytest.fail("TOOLHUNCH_LOCAL_SERVER must name a server in the generated smoke summary")
    async with httpx2.AsyncClient(trust_env=False) as client:
        model = JevWireModel(
            profile["requested_model"],
            base_url=base_url,
            api_key_env=None,
            limits=ModelLimits.model_validate(profile["limits"]),
            http_client=client,
            timeout=40,
            max_retries=0,
        )
        decision = await compatible_example.rank_demo_tools(model)
        assert decision.ranked[0].card.name == "get_weather"
        assert decision.abstained is False
        assert decision.state_cut is False
