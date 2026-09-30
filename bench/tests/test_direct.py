import json
from collections.abc import Sequence
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
from toolhunch.decision import JEV_LIMITS, ChoiceQuestion, DecisionRequest
from toolhunch_bench import cli
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.direct import CATALOGS, DIRECT_ARMS, DirectRunner, direct_catalogs, estimate_direct
from toolhunch_bench.direct_agent import CachedAgent, function_cards, wire_request
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
        ({"code": "x", "param": "tools", "message": "sk-SECRET"}, "ModelHTTPError 400 code=x param=tools"),
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
    estimates = await estimate_direct(selected, retriever=BM25Retriever(), jev_model=fake)
    for suffix in ("first", "replay"):
        guard = SpendGuard()
        decision = CachedDecisionModel(GuardedDecisionModel(fake, guard=guard), path=tmp_path / "jev.sqlite")
        agent = CachedAgent(FunctionModel(answer, model_name="fake-agent"), path=tmp_path / "agent.sqlite", guard=guard)
        try:
            directory = await DirectRunner(retriever=BM25Retriever(), jev_model=decision, agent=agent, guard=guard).run(
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
    estimate = await estimate_direct(selected, retriever=BM25Retriever(), jev_model=fake, not_applicable=exclusion)
    unrestricted = await estimate_direct(selected, retriever=BM25Retriever(), jev_model=fake, not_applicable={})
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
            retriever=BM25Retriever(), jev_model=decision, agent=agent, guard=guard, not_applicable=exclusion
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
    estimate = await estimate_direct(selected, retriever=BM25Retriever(), jev_model=fake)
    guard = SpendGuard()

    def answer(messages: Sequence[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("none")], usage=RequestUsage(input_tokens=100, output_tokens=1))

    decision = CachedDecisionModel(GuardedDecisionModel(fake, guard=guard), path=tmp_path / "jev.sqlite")
    agent = CachedAgent(FunctionModel(answer), path=tmp_path / "agent.sqlite", guard=guard)
    try:
        directory = await DirectRunner(retriever=BM25Retriever(), jev_model=decision, agent=agent, guard=guard).run(
            selected, out_dir=tmp_path, run_id="historical", pilot=True, estimate=estimate
        )
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
    assert summary["run_cost"]["projected_remaining_usd"] is None
    jev = next(r for r in pooled if r["arm"] == "jev-all")
    assert jev["positive_cost"]["cold"]["input_tokens"] > 0
    assert jev["positive_cost"]["cold"]["cache_share"] is None
    published = (output / "README.md").read_text()
    assert "unknown | unknown | unknown | unknown" in published
    assert "Original run completed: False" in published
    assert "4 rejected attempts, excluded" in published
