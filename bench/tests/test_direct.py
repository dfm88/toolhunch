import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import tiktoken
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ModelResponsePart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from typer.testing import CliRunner

from toolhunch import BM25Retriever, OpenAIEmbedder, ToolCard, ToolCatalog
from toolhunch.decision import JEV_LIMITS, STRANDS_LIMITS, ChoiceQuestion, DecisionRequest
from toolhunch_bench import cli
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
from toolhunch_bench.decision import CacheOnlyRetrieval, SharedRetrieval
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.direct import (
    CATALOGS,
    DIRECT_ARMS,
    CatalogOrderRetriever,
    DirectRunner,
    arm_decider,
    decider_arms,
    direct_catalogs,
    estimate_direct,
)
from toolhunch_bench.direct_agent import AGENT_SETTINGS, CachedAgent, agent_payload, function_cards, wire_request
from toolhunch_bench.direct_cost import (
    AGENT_MODEL,
    GuardedDecisionModel,
    GuardedEmbedder,
    ProviderFailure,
    SpendGuard,
    SpendLimit,
    openai_usd,
)
from toolhunch_bench.direct_report import build_direct_report
from toolhunch_bench.embedding_cache import CachedEmbedder
from toolhunch_bench.ledger import LedgerEntry, append_ledger, f2a_spend, read_ledger

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def offline_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    class WordEncoding:
        def encode(self, text: str, *, disallowed_special: Any = ()) -> list[int]:
            return list(range(len(text.split())))

    def encoding(name: str) -> Any:
        return WordEncoding()

    monkeypatch.setattr(tiktoken, "get_encoding", encoding)


def data(*, sources: dict[str, str] | None = None, count: int = 3) -> ToolRetData:
    selected = dict(CATALOGS) if sources is None else sources
    cards = [
        ToolCard(
            id=f"{source}_{i}",
            name=name,
            description=description,
            source=source,
            parameters={"type": "object", "properties": {"value": {}}},
        )
        for source in selected
        for i, (name, description) in enumerate(
            (
                ("weather tool", "Weather forecast. A complete second sentence."),
                ("weather/tool", "Another weather tool."),
                ("mail", "Send email."),
            )
        )
    ]
    tasks = tuple(
        ToolRetTask(
            f"{subtask}_{i}", subtask, "weather", "ignored instruction", frozenset({f"{source}_0", f"{source}_1"})
        )
        for source, subtask in selected.items()
        for i in range(count)
    )
    return ToolRetData(catalog=ToolCatalog(cards), raw_text={}, tasks=tasks, mapping_stats={})


def test_catalog_selection_negatives_and_name_mapping() -> None:
    original = data()
    first, second = direct_catalogs(original), direct_catalogs(original)
    assert [c.negatives for c in first] == [c.negatives for c in second]
    assert sum(len(c.negatives) for c in first) == sum(len(c.positives) for c in first) // 2
    assert [len(c.negatives) for c in first] == [2, 2, 1, 1, 1]
    for selected in first:
        for task, variant, catalog, _ in selected.requests():
            if variant == "negative":
                assert {card.id for card in catalog} == {card.id for card in selected.catalog} - task.relevant
    functions = function_cards(list(first[0].catalog))
    assert [tool.name for tool in functions.tools] == ["weather_tool", "weather_tool_2", "mail"]
    assert list(functions.card_ids.values()) == [card.id for card in first[0].catalog]
    assert json.loads(functions.tools[0].description or "{}")["description"].endswith("complete second sentence.")
    assert functions.tools[0].parameters_json_schema["properties"] == {"value": {"type": "string"}}
    bad = ToolRetTask("missing", "restgpt-spotify", "weather", "", frozenset({"outside"}))
    altered = ToolRetData(catalog=original.catalog, raw_text={}, tasks=(*original.tasks, bad), mapping_stats={})
    assert direct_catalogs(altered)[0].dropped == 1


async def test_estimate_wire_mapping_matches_pydantic_ai_without_internal_fields() -> None:
    functions = function_cards(list(direct_catalogs(data())[0].catalog))
    payload = wire_request("weather", functions)
    model = OpenAIChatModel(AGENT_MODEL, provider=OpenAIProvider(api_key="test"))
    try:
        mapped = model._map_tool_definition(functions.tools[0], {})  # pyright: ignore[reportPrivateUsage]
        assert payload["tools"][0] == mapped
        assert set(payload) == {"messages", "tools"}
        assert set(payload["tools"][0]["function"]) == {"name", "description", "parameters"}
        assert "strict" not in payload["tools"][0]["function"]
    finally:
        await model.client.close()


@pytest.mark.parametrize("kind", ["pick", "none", "formatted_none", "multiple", "other_text"])
async def test_agent_outcomes_and_replay_never_add_usage(tmp_path: Path, kind: str) -> None:
    calls = 0

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert len(info.function_tools) == 3
        assert info.model_settings is not None
        assert "parallel_tool_calls" in info.model_settings
        assert info.model_settings["parallel_tool_calls"] is False
        texts = {"none": "none", "formatted_none": "`None`.", "other_text": "Please clarify."}
        parts: list[ModelResponsePart] = (
            [TextPart(texts[kind])] if kind in texts else [ToolCallPart(info.function_tools[0].name, {})]
        )
        if kind == "multiple":
            parts.append(ToolCallPart(info.function_tools[1].name, {}))
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=100, cache_read_tokens=60, output_tokens=5))

    guard = SpendGuard()
    agent = CachedAgent(FunctionModel(answer), path=tmp_path / "agent.sqlite", guard=guard)
    functions = function_cards(list(direct_catalogs(data())[0].catalog))
    try:
        result, replayed = await agent.ask("weather", functions, catalog_name="test")
        assert not replayed
        assert result.pick == (
            None if kind in {"none", "formatted_none", "other_text"} else functions.card_ids[functions.tools[0].name]
        )
        assert result.extra_calls == (1 if kind == "multiple" else 0)
        assert result.text_is_none == (kind in {"none", "formatted_none"})
        cost, usage_count = guard.run_usd, len(guard.calls)
        cached, replayed = await agent.ask("weather", functions, catalog_name="test")
        assert replayed
        assert cached == result
        assert (guard.run_usd, len(guard.calls), calls) == (cost, usage_count, 1)
        assert guard.calls[0]["cache_read_tokens"] == 60
        assert guard.calls[0]["usd"] < guard.calls[0]["list_usd"]
    finally:
        agent.close()


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {"code": "invalid_value", "param": "tools", "message": "sk-SECRET"},
            "ModelHTTPError 400 code=invalid_value param=tools",
        ),
        (
            {"error": {"code": "array_above_max_length", "param": "tools", "message": "sk-SECRET"}},
            "ModelHTTPError 400 code=array_above_max_length param=tools",
        ),
        (
            {"code": "sk-SECRET", "param": "sk-SECRET", "message": "sk-SECRET"},
            "ModelHTTPError 400 code=redacted param=redacted",
        ),
    ],
)
async def test_http_diagnostics_keep_only_safe_fields(
    tmp_path: Path,
    body: dict[str, Any],
    expected: str,
) -> None:
    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ModelHTTPError(status_code=400, model_name=AGENT_MODEL, body=body)

    guard = SpendGuard()
    agent = CachedAgent(FunctionModel(answer), path=tmp_path / "agent.sqlite", guard=guard)
    try:
        with pytest.raises(ProviderFailure) as failure:
            await agent.ask("weather", function_cards(list(direct_catalogs(data())[0].catalog)), catalog_name="test")
        assert str(failure.value) == expected
        assert len(guard.calls) == 1
        assert guard.calls[0]["error"] == expected
        assert "sk-" not in json.dumps(guard.calls)
        assert "message" not in json.dumps(guard.calls)
    finally:
        agent.close()


async def test_guard_refuses_before_call_and_guards_retry(
    tmp_path: Path,
    fake_decision_model: Any,
    axis_embedder: Any,
) -> None:
    attempts = 0

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("this private error text must not be recorded")
        return ModelResponse(parts=[TextPart("none")], usage=RequestUsage(input_tokens=100, output_tokens=1))

    functions = function_cards(list(direct_catalogs(data())[0].catalog))
    refused = CachedAgent(FunctionModel(answer), path=tmp_path / "refused.sqlite", guard=SpendGuard(cap_usd=0))
    try:
        with pytest.raises(SpendLimit):
            await refused.ask("weather", functions, catalog_name="test")
        assert attempts == 0
    finally:
        refused.close()
    guard = SpendGuard()
    retried = CachedAgent(FunctionModel(answer), path=tmp_path / "retry.sqlite", guard=guard)
    try:
        await retried.ask("weather", functions, catalog_name="test")
        assert attempts == 2
        assert len(guard.calls) == 2
        assert guard.calls[0]["error"] == "TimeoutError"
        assert guard.calls[0]["usd"] is None
        assert guard.run_usd > guard.calls[1]["usd"]
        assert "private error" not in json.dumps(guard.calls)
    finally:
        retried.close()
    stopped = SpendGuard(cap_usd=0)
    fake = fake_decision_model(limits=JEV_LIMITS)
    wrapped = GuardedDecisionModel(fake, guard=stopped)
    with pytest.raises(SpendLimit):
        await wrapped.ask(
            DecisionRequest(
                state="pick",
                questions={
                    "tool": ChoiceQuestion(instructions="pick", options={"a": "first", "b": "second"}),
                },
            )
        )
    assert not fake.asks
    embedder = axis_embedder(model_id="text-embedding-3-small")
    with pytest.raises(SpendLimit):
        await GuardedEmbedder(embedder, guard=stopped).embed(["weather"], kind="query")
    assert not embedder.calls


def test_cached_prices_and_purpose_prefix(tmp_path: Path) -> None:
    assert openai_usd(
        model=AGENT_MODEL, input_tokens=10000, cache_read_tokens=8000, output_tokens=100
    ) == pytest.approx((2000 * 0.40 + 8000 * 0.10 + 100 * 1.60) / 1_000_000)
    ledger = tmp_path / "ledger.jsonl"
    for purpose, cost in (("F2a: decision", 2), ("P1: direct choice pilot", 1), ("P1-other", 10)):
        append_ledger(
            LedgerEntry(
                timestamp=datetime.fromisoformat("2026-09-29T00:00:00Z"),
                run_id="test",
                provider="openai",
                model="test",
                input_tokens=1,
                usd=cost,
                purpose=purpose,
            ),
            path=ledger,
        )
    entries = read_ledger(ledger)
    assert f2a_spend(entries).usd == 2
    assert f2a_spend(entries, purpose_prefix="P1:").usd == 1


async def test_runner_report_symmetric_scoring_and_replay_exclusion(tmp_path: Path, fake_decision_model: Any) -> None:
    selected = direct_catalogs(
        data(sources={"webtools_spotify": "restgpt-spotify"}, count=2), sources={"webtools_spotify": "restgpt-spotify"}
    )

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        matching = [tool for tool in info.function_tools if "Weather forecast" in (tool.description or "")]
        parts: list[ModelResponsePart] = [ToolCallPart(matching[0].name, {})] if matching else [TextPart("none")]
        return ModelResponse(parts=parts, usage=RequestUsage(input_tokens=100, cache_read_tokens=50, output_tokens=1))

    fake = fake_decision_model(model_id="jev-fake@test", limits=JEV_LIMITS, favourite="weather tool", none_weight=100)
    estimates = await estimate_direct(selected, retriever=BM25Retriever(), models={"jev": fake})
    for suffix in ("first", "replay"):
        guard = SpendGuard()
        decision = CachedDecisionModel(GuardedDecisionModel(fake, guard=guard), path=tmp_path / "jev.sqlite")
        agent = CachedAgent(FunctionModel(answer, model_name="fake-agent"), path=tmp_path / "agent.sqlite", guard=guard)
        try:
            directory = await DirectRunner(
                retriever=BM25Retriever(), models={"jev": decision}, agent=agent, guard=guard
            ).run(
                selected,
                out_dir=tmp_path / "runs",
                run_id=suffix,
                pilot=False,
                estimate=estimates,
            )
            summary = build_direct_report(directory, out_dir=tmp_path / "results" / suffix)
            assert summary["manifest"]["completed"]
            pooled = {row["arm"]: row for row in summary["rows"] if row["catalog"] == "pooled"}
            assert pooled["hybrid@20+jev"]["relevant_pick_rate"]["value"] == 0
            assert pooled["agent-all"]["relevant_pick_rate"]["value"] == 1
            assert pooled["hybrid@20+jev"]["ranking_ignoring_abstention"]["value"] == 1
            assert pooled["hybrid@20+jev"]["none_option"]["abstained"] == 3
            assert pooled["agent-all"]["agent_text_none"] == 1
            assert pooled["agent-all"]["agent_other_text_abstentions"] == 0
            if suffix == "replay":
                assert not guard.calls
                assert summary["run_cost"]["verified_usd"] == 0
                assert pooled["agent-all"]["positive_cost"]["cold"]["billed_usd"] is None
                assert pooled["agent-all"]["positive_cost"]["warm"]["cache_share"] is None
        finally:
            agent.close()
            decision.close()


def test_dry_run_has_no_provider_or_ledger_side_effects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry run attempted key loading or provider I/O")

    def load_toolret(*, cache_dir: Path) -> ToolRetData:
        return data()

    monkeypatch.setattr(cli, "load_toolret", load_toolret)
    monkeypatch.setattr(cli, "load_dotenv", forbidden)
    monkeypatch.setattr(OpenAIEmbedder, "embed", forbidden)
    monkeypatch.setattr(cli, "EMBEDDING_CACHE_PATH", tmp_path / "embedding.sqlite")
    monkeypatch.setattr(cli, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    output = tmp_path / "estimate.json"
    result = CliRunner().invoke(cli.app, ["direct", "--dry-run", "--estimate-out", str(output)])
    assert result.exit_code == 0, result.output
    estimates = json.loads(output.read_text())
    assert estimates["full"]["requests_per_arm"] == 22
    assert estimates["full"]["usd"] > 0
    agent_lines = {line["arm"]: line for line in estimates["full"]["lines"] if line.get("arm", "").startswith("agent")}
    assert set(agent_lines) == {"agent@20", "agent-all"}
    assert {arm: line["requests"] for arm, line in agent_lines.items()} == {"agent@20": 22, "agent-all": 18}
    assert estimates["full"]["planned_requests_by_arm"]["agent-all"] == 18
    assert not (tmp_path / "ledger.jsonl").exists()


@pytest.mark.parametrize("fail_applicable", [False, True])
async def test_non_applicable_runner_and_estimator_contract(
    tmp_path: Path, fake_decision_model: Any, fail_applicable: bool
) -> None:
    sources = {"metatool_which": "metatool", "webtools_spotify": "restgpt-spotify"}
    original = data(sources=sources, count=10)
    catalog = ToolCatalog(
        [
            *original.catalog,
            *(
                ToolCard(id=f"{source}_extra_{i}", name=f"extra_{i}", description="Unrelated utility.", source=source)
                for source in sources
                for i in range(18)
            ),
        ]
    )
    selected = direct_catalogs(
        ToolRetData(catalog=catalog, tasks=original.tasks, raw_text={}, mapping_stats={}), sources=sources
    )
    exclusion = {
        ("agent-all", "metatool_which"): {
            "reason": "fixture rejection",
            "source": "offline fixture",
            "date": "2026-09-30",
        }
    }
    fake = fake_decision_model(model_id="jev-fake@test", limits=JEV_LIMITS, favourite="weather tool")
    estimate = await estimate_direct(
        selected, retriever=BM25Retriever(), models={"jev": fake}, not_applicable=exclusion
    )
    unrestricted = await estimate_direct(selected, retriever=BM25Retriever(), models={"jev": fake}, not_applicable={})
    assert estimate.planned_requests_by_arm == {arm: 15 if arm == "agent-all" else 30 for arm in DIRECT_ARMS}
    assert unrestricted.planned_requests_by_arm["agent-all"] == 30
    assert unrestricted.usd > estimate.usd
    guard = SpendGuard()

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert (guard.context["arm"], guard.context["catalog"]) not in exclusion
        if fail_applicable and guard.context["arm"] == "agent-all":
            raise ModelHTTPError(400, AGENT_MODEL, {"code": "array_above_max_length", "param": "tools"})
        return ModelResponse(parts=[TextPart("none")], usage=RequestUsage(input_tokens=100, output_tokens=1))

    decision = CachedDecisionModel(GuardedDecisionModel(fake, guard=guard), path=tmp_path / "jev.sqlite")
    agent = CachedAgent(FunctionModel(answer), path=tmp_path / "agent.sqlite", guard=guard)
    try:
        directory = await DirectRunner(
            retriever=BM25Retriever(), models={"jev": decision}, agent=agent, guard=guard, not_applicable=exclusion
        ).run(selected, out_dir=tmp_path, run_id="applicability", pilot=True, estimate=estimate)
        manifest = json.loads((directory / "manifest.json").read_text())
        records = [json.loads(line) for line in (directory / "run.jsonl").read_text().splitlines()]
        assert manifest["not_applicable"] == [
            {"arm": "agent-all", "catalog": "metatool_which", **exclusion[("agent-all", "metatool_which")]}
        ]
        assert manifest["planned_requests_by_arm"]["agent-all"] == 15
        assert not any((r["arm"], r["catalog"]) in exclusion for r in records)
        assert not any((c["arm"], c["catalog"]) in exclusion for c in guard.calls)
        assert manifest["completed"] is not fail_applicable
        if fail_applicable:
            assert manifest["counts"]["agent-all"] == 1
            assert manifest["stop_reason"] == "more than 5% errored requests in agent-all"
        else:
            assert manifest["counts"] == dict(estimate.planned_requests_by_arm)
    finally:
        decision.close()
        agent.close()


async def test_historical_rejections_common_pool_unknown_cost_and_jev_cache(
    tmp_path: Path, fake_decision_model: Any
) -> None:
    sources = {"webtools_spotify": "restgpt-spotify", "metatool_which": "metatool"}
    selected = direct_catalogs(data(sources=sources, count=2), sources=sources)
    fake = fake_decision_model(model_id="jev-fake@test", limits=JEV_LIMITS, favourite="weather tool")
    estimate = await estimate_direct(selected, retriever=BM25Retriever(), models={"jev": fake})
    guard = SpendGuard()

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("none")], usage=RequestUsage(input_tokens=100, output_tokens=1))

    decision = CachedDecisionModel(GuardedDecisionModel(fake, guard=guard), path=tmp_path / "jev.sqlite")
    agent = CachedAgent(FunctionModel(answer), path=tmp_path / "agent.sqlite", guard=guard)
    try:
        directory = await DirectRunner(
            retriever=BM25Retriever(), models={"jev": decision}, agent=agent, guard=guard
        ).run(selected, out_dir=tmp_path, run_id="historical", pilot=True, estimate=estimate)
    finally:
        decision.close()
        agent.close()
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest.pop("not_applicable")
    manifest["completed"] = False
    manifest["stop_reason"] = "more than 5% errored requests in agent-all"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    records = [json.loads(line) for line in (directory / "run.jsonl").read_text().splitlines()]
    calls = [json.loads(line) for line in (directory / "calls.jsonl").read_text().splitlines()]
    reference = next(r for r in records if r["arm"] == "agent-all")
    reference_call = next(c for c in calls if c["arm"] == "agent-all")
    for i in range(4):
        failed_call = {
            **reference_call,
            "catalog": "metatool_which",
            "usd": None,
            "list_usd": None,
            "error": "ModelHTTPError 400 code=array_above_max_length param=tools",
            "budget_charge_usd": 0.04,
        }
        calls.append(failed_call)
        records.append(
            {
                **reference,
                "catalog": "metatool_which",
                "task": f"rejected_{i}",
                "phase": "warm",
                "error": failed_call["error"],
                "abstained": None,
                "pick": None,
                "provider_calls": [failed_call],
            }
        )
    (directory / "run.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (directory / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    original = (directory / "manifest.json").read_bytes()
    output = tmp_path / "report"
    summary = build_direct_report(directory, out_dir=output)
    assert (directory / "manifest.json").read_bytes() == original
    assert summary["manifest"]["completed"] is False
    assert summary["manifest"]["stop_reason"] == manifest["stop_reason"]
    assert summary["applicability"]["common_catalogs"] == ["webtools_spotify"]
    pooled = [r for r in summary["rows"] if r["catalog"] == "pooled"]
    assert len(pooled) == 5
    assert all(r["positives"] == 2 and r["catalogs"] == ["webtools_spotify"] for r in pooled)
    secondary = [r for r in summary["rows"] if r["catalog"] == "all_catalogs"]
    assert len(secondary) == 4
    assert all(r["positives"] == 4 for r in secondary)
    excluded = next(r for r in summary["rows"] if r["status"] == "not applicable")
    assert excluded["excluded_rejected_attempts"] == 4
    assert excluded["relevant_pick_rate"] is None
    assert excluded["none_option"] is None
    unknown = excluded["excluded_attempt_cost"]["warm"]
    assert unknown["unpriced_attempts"] == 4
    assert all(
        unknown[field] is None for field in ("billed_usd", "list_usd", "billed_usd_per_1000", "list_usd_per_1000")
    )
    validation = summary["validation"]["agent-all"]
    assert validation["raw_errors"] == 4
    assert validation["errors"] == 0
    assert validation["raw_requests"] == 7
    assert validation["requests"] == validation["planned_requests"] == 3
    assert validation["classified_successful"] == 3
    assert validation["classified_fraction"] == 1
    assert summary["run_cost"]["projected_remaining_usd"] >= 0
    assert summary["run_cost"]["observed_remaining_usd"] is not None
    assert summary["run_cost"]["projected_p1_total_usd"] == pytest.approx(
        manifest.get("prior_p1_usd", 0)
        + summary["run_cost"]["budget_charge_usd"]
        + summary["run_cost"]["projected_remaining_usd"]
    )
    jev = next(r for r in pooled if r["arm"] == "jev-all")
    assert jev["positive_cost"]["cold"]["input_tokens"] > 0
    assert jev["positive_cost"]["cold"]["cache_share"] is None
    published = (output / "README.md").read_text()
    assert "unknown | unknown | unknown | unknown" in published
    assert "Original run completed: False" in published
    assert "4 rejected attempts, excluded" in published
    baseline_rows = [
        line
        for line in published.split("## Observed provider cost")[0].splitlines()
        if line.startswith("| hybrid@20 |")
    ]
    assert len(baseline_rows) == 4
    assert all(line.split(" | ")[3] == "—" for line in baseline_rows)


@pytest.mark.parametrize("collision", [False, True])
async def test_replay_identity_tracks_card_mapping_without_changing_provider_request(
    tmp_path: Path, collision: bool
) -> None:
    cards = [ToolCard(id="a", name="same?", description="same")]
    if collision:
        cards.append(ToolCard(id="b", name="same?", description="same"))
    changed = list(reversed(cards)) if collision else [replace(cards[0], id="b")]
    original_functions, changed_functions = function_cards(cards), function_cards(changed)
    assert wire_request("weather", original_functions) == wire_request("weather", changed_functions)
    assert agent_payload("weather", original_functions, catalog_name="test") == agent_payload(
        "weather", changed_functions, catalog_name="test"
    )
    calls = 0

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        assert info.model_settings == dict(AGENT_SETTINGS) | {"openai_prompt_cache_key": "test"}
        return ModelResponse(
            parts=[ToolCallPart(info.function_tools[0].name, {})], usage=RequestUsage(input_tokens=100)
        )

    guard = SpendGuard()
    model = FunctionModel(answer)
    agent = CachedAgent(model, path=tmp_path / "identity.sqlite", guard=guard)
    try:
        original, replayed = await agent.ask("weather", original_functions, catalog_name="test")
        assert not replayed
        assert original.pick == "a"
        legacy = agent_payload("weather", original_functions, catalog_name="test") | {"model": model.model_name}
        legacy_key = hashlib.sha256(
            json.dumps(legacy, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        db = sqlite3.connect(tmp_path / "identity.sqlite")
        try:
            db.execute("INSERT INTO agent_responses VALUES (?, ?)", (legacy_key, json.dumps(asdict(original))))
            db.commit()
        finally:
            db.close()
        updated, replayed = await agent.ask("weather", changed_functions, catalog_name="test")
        assert not replayed
        assert updated.pick == "b"
        again, replayed = await agent.ask("weather", original_functions, catalog_name="test")
        assert replayed
        assert again == original
        assert calls == len(guard.calls) == 2
        if collision:
            assert original_functions.card_ids == {"same_": "a", "same__2": "b"}
            assert changed_functions.card_ids == {"same_": "b", "same__2": "a"}
    finally:
        agent.close()


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("lexical_matches", [0, 2, 25])
async def test_direct_estimate_bounds_short_missing_dense_candidates_only(
    tmp_path: Path, fake_decision_model: Any, axis_embedder: Any, cached: bool, lexical_matches: int
) -> None:
    source = "webtools_spotify"
    original = data(sources={source: "restgpt-spotify"}, count=2)
    catalog = ToolCatalog(
        [
            *original.catalog,
            *(
                ToolCard(
                    id=f"extra_{i}",
                    name=f"utility_{i}",
                    description=("Long weather payload " if lexical_matches == 25 else "Long unrelated payload ")
                    * (i + 20),
                    source=source,
                )
                for i in range(22)
            ),
        ]
    )
    tasks = tuple(
        ToolRetTask(task.id, task.subtask, "weather" if lexical_matches else "zxqv", "", task.relevant)
        for task in original.tasks
    )
    selected = direct_catalogs(
        ToolRetData(catalog=catalog, tasks=tasks, raw_text={}, mapping_stats={}), sources={source: "restgpt-spotify"}
    )
    embedder = axis_embedder(model_id="text-embedding-3-small")
    embeddings = CachedEmbedder(embedder, path=tmp_path / "embeddings.sqlite")
    try:
        if cached:
            from toolhunch import default_search_text

            await embeddings.embed([default_search_text(card) for card in catalog], kind="document")
            await embeddings.embed([tasks[0].query], kind="query")
        model = fake_decision_model(limits=JEV_LIMITS)
        embedding_calls = len(embedder.calls)
        free = CacheOnlyRetrieval(CatalogOrderRetriever(), stand_in=BM25Retriever(), embeddings=embeddings)
        shared = SharedRetrieval(free)
        estimate = await estimate_direct(selected, retriever=shared, models={"jev": model}, embeddings=embeddings)
        reference = await estimate_direct(
            selected,
            retriever=SharedRetrieval(CatalogOrderRetriever() if cached else BM25Retriever()),
            models={"jev": model},
            embeddings=embeddings,
        )
        assert not model.asks
        assert len(embedder.calls) == embedding_calls
        lines = {line.get("arm"): line for line in estimate.lines}
        reference_lines = {line.get("arm"): line for line in reference.lines}
        assert lines["hybrid@20+jev"]["requests"] == estimate.planned_requests_by_arm["hybrid@20+jev"] == 3
        assert lines["agent@20"]["requests"] == 3
        if cached:
            assert estimate.lines == reference.lines
            assert estimate.stand_in_searches == 0
            assert estimate.candidate_bound_searches == {"hybrid@20+jev": 0, "agent@20": 0}
        elif lexical_matches == 25:
            assert estimate.lines == reference.lines
            assert estimate.stand_in_searches == len(free.stand_in_queries) == 1
            assert estimate.candidate_bound_searches == {"hybrid@20+jev": 0, "agent@20": 0}
        else:
            assert estimate.stand_in_searches == len(free.stand_in_queries) == 1
            assert not SharedRetrieval(free).stand_in_queries
            assert estimate.candidate_bound_searches == {"hybrid@20+jev": 2, "agent@20": 2}
            for arm in ("hybrid@20+jev", "agent@20"):
                assert lines[arm]["input_tokens"] > reference_lines[arm]["input_tokens"]
                assert lines[arm]["usd"] > reference_lines[arm]["usd"]
            if lexical_matches == 0:
                assert reference_lines["hybrid@20+jev"]["requests"] == 0
            assert "actual Jev planner limits/detail retained" in estimate.candidate_bound_method
    finally:
        embeddings.close()


async def test_full_and_pilot_estimate_fallback_counts_are_scoped_to_each_wrapper(
    tmp_path: Path, fake_decision_model: Any, axis_embedder: Any
) -> None:
    sources = {"webtools_spotify": "restgpt-spotify"}
    original = data(sources=sources, count=4)
    tasks = tuple(replace(task, query=f"weather number_{i}") for i, task in enumerate(original.tasks))
    full = direct_catalogs(replace(original, tasks=tasks), sources=sources)
    pilot = [replace(full[0], positives=full[0].positives[:1], negatives=())]
    embedder = axis_embedder(model_id="text-embedding-3-small")
    embeddings = CachedEmbedder(embedder, path=tmp_path / "embeddings.sqlite")
    model = fake_decision_model(limits=JEV_LIMITS)
    free = CacheOnlyRetrieval(CatalogOrderRetriever(), stand_in=BM25Retriever(), embeddings=embeddings)
    try:
        full_estimate = await estimate_direct(full, retriever=SharedRetrieval(free), models={"jev": model})
        pilot_estimate = await estimate_direct(pilot, retriever=SharedRetrieval(free), models={"jev": model})
        assert full_estimate.stand_in_searches == len(free.stand_in_queries) == 4
        assert pilot_estimate.stand_in_searches == 1
        assert asdict(pilot_estimate)["stand_in_searches"] == 1
        assert not embedder.calls
        assert not model.asks
    finally:
        embeddings.close()


@pytest.mark.parametrize(
    ("pilot", "completed", "coverage", "unpriced", "expected_remaining"),
    [
        (False, False, "absent", False, 10.0),
        (False, False, "partial", False, 1.68),
        (False, True, "partial", False, 0.0),
        (True, False, "partial", False, 1.68),
        (True, False, "partial", True, 10.0),
        (False, False, "unknown_embedding", False, 10.0),
    ],
)
async def test_projection_applicable_evidence_sunk_charges_and_completion(
    tmp_path: Path, pilot: bool, completed: bool, coverage: str, unpriced: bool, expected_remaining: float
) -> None:
    estimate = {"usd": 10.0, "lines": [{"model": "text-embedding-3-small", "usd": 0.1}]}
    manifest = {
        "pilot": pilot,
        "completed": completed,
        "prior_p1_usd": 5.3,
        "catalogs": [{"source": "webtools_spotify", "positives": 3, "negatives": 2}],
        "full_catalogs": [{"source": "webtools_spotify", "positives": 3, "negatives": 2}],
        "estimate": estimate,
        "full_estimate": estimate,
    }
    excluded = {
        "arm": "agent-all",
        "catalog": "metatool_which",
        "model": AGENT_MODEL,
        "provider": "openai",
        "usd": None,
        "list_usd": None,
        "budget_charge_usd": 0.7,
        "error": "rejected",
        "input_tokens": None,
        "cache_read_tokens": None,
    }
    calls: list[dict[str, Any]] = [excluded]
    records: list[dict[str, Any]] = []
    if coverage != "absent":
        calls.append(
            {
                **excluded,
                "arm": "hybrid@20",
                "catalog": "webtools_spotify",
                "model": "text-embedding-3-small",
                "usd": None if coverage == "unknown_embedding" else 0.02,
                "budget_charge_usd": 0.12 if coverage == "unknown_embedding" else 0.02,
            }
        )
        for arm in DIRECT_ARMS:
            for task, variant, phase in (
                ("p0", "positive", "cold"),
                ("p1", "positive", "warm"),
                ("p0", "negative", "negative"),
            ):
                unknown = unpriced and arm == "agent@20" and phase == "warm"
                paid = (
                    []
                    if arm == "hybrid@20"
                    else [
                        {
                            **excluded,
                            "arm": arm,
                            "catalog": "webtools_spotify",
                            "usd": None if unknown else 0.2,
                            "list_usd": None if unknown else 0.2,
                            "budget_charge_usd": 0.9 if unknown else 0.2,
                            "error": None,
                            "input_tokens": 100,
                            "cache_read_tokens": 0,
                        }
                    ]
                )
                calls.extend(paid)
                records.append(
                    {
                        "arm": arm,
                        "catalog": "webtools_spotify",
                        "task": task,
                        "variant": variant,
                        "phase": phase,
                        "error": None,
                        "abstained": False,
                        "pick": "tool",
                        "relevant": ["tool"],
                        "candidates": ["tool"],
                        "ranked": ["tool"],
                        "detail": ["FULL"],
                        "extra_calls": 0,
                        "provider_calls": paid,
                        "search_usd": 0,
                        "search_seconds": 0,
                        "decision_seconds": 0,
                        "replayed": False,
                    }
                )
    raw = tmp_path / "run"
    raw.mkdir()
    (raw / "manifest.json").write_text(json.dumps(manifest))
    (raw / "run.jsonl").write_text("\n".join(map(json.dumps, records)))
    (raw / "calls.jsonl").write_text("\n".join(map(json.dumps, calls)))
    summary = build_direct_report(raw, out_dir=tmp_path / "result")
    cost = summary["run_cost"]
    assert cost["projected_remaining_usd"] == pytest.approx(expected_remaining)
    assert cost["budget_charge_usd"] == pytest.approx(sum(c["budget_charge_usd"] for c in calls))
    assert cost["projected_p1_total_usd"] == pytest.approx(5.3 + cost["budget_charge_usd"] + expected_remaining)
    assert cost["unpriced_attempts"] == 1 + unpriced + (coverage == "unknown_embedding")
    assert summary["manifest"] == manifest
    if not pilot and completed:
        assert cost["projection_method"] == "completed full run"
        assert cost["uncached_remaining_budget_reference_usd"] == 0
    else:
        assert "completed full run" not in cost["projection_method"]
        assert cost["remaining_requests_by_arm"] == dict.fromkeys(DIRECT_ARMS, 5 if coverage == "absent" else 2)
        assert cost["uncached_remaining_budget_reference_usd"] == 10
        assert (cost["observed_remaining_usd"] is not None) == (coverage == "partial" and not unpriced)
    baseline_rows = [
        line
        for line in (tmp_path / "result/README.md").read_text().split("## Observed provider cost")[0].splitlines()
        if line.startswith("| hybrid@20 |") and " / " in line
    ]
    assert baseline_rows
    assert all(line.split(" | ")[3] == "—" for line in baseline_rows)
    if records:
        assert "0.000 (0.000 to 0.000)" in (tmp_path / "result/README.md").read_text()


async def test_direct_runs_a_registered_decider_in_rounds(tmp_path: Path, fake_decision_model: Any) -> None:
    # A three-tool catalog plus the reserved option overflows a three-option cap: positives need two rounds, while a
    # negative (gold tools removed, one tool left) fits one question.
    selected = direct_catalogs(
        data(sources={"webtools_spotify": "restgpt-spotify"}, count=2), sources={"webtools_spotify": "restgpt-spotify"}
    )
    limits = STRANDS_LIMITS.model_copy(update={"max_options_per_choice": 3})
    model = fake_decision_model(model_id="strands-fake@127.0.0.1:8000", limits=limits, favourite="weather tool")
    arms = decider_arms(["strands"])
    assert arms == ("hybrid@20+strands", "strands-all")
    assert [arm_decider(arm) for arm in (*arms, "hybrid@20+jev", "jev-all", "agent-all", "hybrid@20")] == [
        "strands",
        "strands",
        "jev",
        "jev",
        None,
        None,
    ]
    estimate = await estimate_direct(selected, retriever=BM25Retriever(), models={"strands": model}, arms=arms)
    guard = SpendGuard()

    directory = await DirectRunner(
        retriever=BM25Retriever(), models={"strands": model}, agent=None, guard=guard, arms=arms
    ).run(selected, out_dir=tmp_path / "runs", run_id="strands", pilot=False, estimate=estimate)

    records = [json.loads(line) for line in (directory / "run.jsonl").read_text().splitlines()]
    everything = [r for r in records if r["arm"] == "strands-all"]
    assert all(len(r["detail"]) == 2 for r in everything if r["variant"] == "positive")
    assert all(len(r["detail"]) == 1 for r in everything if r["variant"] == "negative")
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["deciders"]["strands"]["model"] == "strands-fake@127.0.0.1:8000"
    summary = build_direct_report(directory, out_dir=tmp_path / "report")
    [row] = [row for row in summary["rows"] if row["arm"] == "strands-all" and row["catalog"] == "pooled"]
    positives = sum(r["variant"] == "positive" for r in everything)
    assert row["rounds"] == {"one_question": len(everything) - positives, "two_rounds": positives}
    assert row["physical_requests"] == 2 * positives + (len(everything) - positives)
    assert "| strands-all, in rounds | pooled |" in (tmp_path / "report" / "README.md").read_text()
