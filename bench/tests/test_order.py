import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio
import pytest
from typer.testing import CliRunner

from toolhunch import DetailLevel, OpenAIEmbedder, ScoredCard, ToolCard, ToolCatalog
from toolhunch.decision import (
    CLEF_FLASH_LIMITS,
    JEV_LIMITS,
    LOGPROB_LIMITS,
    STRANDS_LIMITS,
    ChoiceQuestion,
    DecisionError,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
    OpenAILogprobModel,
)
from toolhunch.retrieval import Retrieval
from toolhunch_bench import cli
from toolhunch_bench import order as order_module
from toolhunch_bench.datasets.toolret import TOOLRET_SUBTASKS, ToolRetData, write_task_file
from toolhunch_bench.decision import (
    JEV_MODEL,
    LOGPROB_MODEL,
    SharedRetrieval,
    build_decision_arms,
    estimate_decisions,
    run_decisions,
)
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.direct_cost import SpendGuard, SpendLimit
from toolhunch_bench.ledger import BUDGET
from toolhunch_bench.order import ORDER_DETAILS, ORDER_SEEDS, OrderDecisionModel, order_ledger_entries
from toolhunch_bench.order_report import build_order_report, pilot_gates


class CatalogRetriever:
    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        return Retrieval(matches=tuple(ScoredCard(card, 0.0) for card in list(catalog)[:k]))


@pytest.fixture
def order_data(toolret_data: ToolRetData) -> ToolRetData:
    extras = [ToolCard(id=f"padding_{i}", name=f"padding_{i}", description="Other task") for i in range(8)]
    return replace(toolret_data, catalog=ToolCatalog([*toolret_data.catalog, *extras]), tasks=toolret_data.tasks[:2])


async def recorded_run(
    tmp_path: Path, data: ToolRetData, fake_factory: Any, *, fail: bool = False, refuse: bool = False
) -> tuple[Path, SpendGuard]:
    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, data.tasks, seed=0)
    run_dir = tmp_path / "runs/synthetic"

    def persist(call: dict[str, Any]) -> None:
        with (run_dir / "calls.jsonl").open("a") as journal:
            journal.write(json.dumps(call) + "\n")

    guard = SpendGuard(prior_usd=1.3, sink=persist)
    fakes = {
        "jev": fake_factory(
            model_id=JEV_MODEL,
            limits=JEV_LIMITS,
            favourite="get_weather",
            fail_on="Rome" if fail else None,
            refuse_on="Rome" if refuse else None,
        ),
        "logprob": fake_factory(model_id=LOGPROB_MODEL, limits=LOGPROB_LIMITS, favourite="get_weather"),
    }
    models = {name: OrderDecisionModel(fake, decider=name, guard=guard) for name, fake in fakes.items()}
    arms = build_decision_arms(["jev", "logprob"], ks=[20], models=models, max_detail=ORDER_DETAILS)
    await run_decisions(
        arms,
        data,
        data.tasks,
        retriever=SharedRetrieval(CatalogRetriever()),
        split="heldout",
        sources=["plain"],
        model_queries=None,
        negatives=True,
        repeat=1,
        out_dir=tmp_path / "runs",
        task_file=task_file,
        model_queries_file=None,
        run_id="synthetic",
        order_seeds=ORDER_SEEDS,
        manifest_extra={
            "completed": not fail,
            "experiment": "order-sensitivity-v1",
            "pilot": True,
            "estimate": {"usd": 1.0},
            "prior_p1_usd": 1.3,
            "cap_usd": 7.0,
        },
        before_search=lambda context: setattr(guard, "context", context),
    )
    return run_dir, guard


@pytest.mark.anyio
async def test_five_orders_are_fresh_and_record_real_round_slot_mappings(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    run_dir, guard = await recorded_run(tmp_path, order_data, fake_decision_model)
    assert pilot_gates(run_dir)["passed"] is True
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    rows = [row for row in records if row["record"] == "search" and row["decider"]]
    assert len(rows) == 40
    assert len(guard.calls) == 60  # 20 Jev asks plus two asks per logprob decision
    for row in rows:
        assert len(row["candidates"]) == 20
        if row["variant"] == "negative":
            assert not set(row["relevant"]) & set(row["candidates"])
        for exchange in row["exchanges"]:
            assert exchange["candidate_order"] == list(exchange["option_card_ids"].values())
            assert set(exchange["candidate_order"]) <= set(row["candidates"])
        if row["decider"] == "logprob":
            assert [exchange["round"] for exchange in row["exchanges"]] == [1, 2]
            assert row["exchanges"][0]["candidate_order"] == row["candidates"]
    summary = build_order_report(run_dir, out_dir=tmp_path / "report", published_summary=None)
    assert "clm" not in json.dumps(summary).lower()
    assert summary["deciders"]["jev"]["top_card_stability"]["value"] == 1.0
    logprob = summary["deciders"]["logprob"]
    assert logprob["permutation_averaging"]["production_decisions_per_search"] == 5
    assert logprob["permutation_averaging"]["physical_attempts_per_search"] == 10
    assert logprob["chosen_slots_positive"]["p_value"] is None
    assert logprob["abstention_by_order"]["0"]["negative_share"] == 0.5
    assert sum(logprob["abstention_by_order"]["0"][key] for key in ("correct", "wrong", "abstained")) == 4
    assert logprob["identity_drift"]["comparable"] is False


@pytest.mark.anyio
async def test_order_estimate_asks_no_models_and_counts_every_round(
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    models = {
        "jev": fake_decision_model(model_id=JEV_MODEL, limits=JEV_LIMITS),
        "logprob": fake_decision_model(model_id=LOGPROB_MODEL, limits=LOGPROB_LIMITS),
    }
    arms = build_decision_arms(["jev", "logprob"], ks=[20], models=models, max_detail=ORDER_DETAILS)
    lines = await estimate_decisions(
        arms,
        order_data,
        order_data.tasks,
        retriever=SharedRetrieval(CatalogRetriever()),
        sources=["plain"],
        model_queries=None,
        negatives=True,
        repeat=1,
        order_seeds=ORDER_SEEDS,
    )
    assert [line.calls for line in lines] == [20, 40]
    assert all(not fake.asks for fake in models.values())


@pytest.mark.anyio
@pytest.mark.parametrize("defect", ["incomplete", "identical", "reserved", "unknown", "expensive"])
async def test_pilot_gate_rejects_bad_evidence(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
    defect: str,
) -> None:
    run_dir, _ = await recorded_run(tmp_path, order_data, fake_decision_model)
    if defect == "incomplete":
        path = run_dir / "manifest.json"
        manifest = json.loads(path.read_text()) | {"completed": False}
        path.write_text(json.dumps(manifest))
    elif defect in ("identical", "reserved"):
        path = run_dir / "run.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        own = [row for row in rows if row.get("decider") == "jev" and row.get("record") == "search"]
        if defect == "identical":
            own[1]["candidates"] = own[0]["candidates"]
        else:
            options = own[0]["exchanges"][-1]["request"]["questions"]["tool"]["options"]
            options.insert(0, options.pop())
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    else:
        path = run_dir / "calls.jsonl"
        calls = [json.loads(line) for line in path.read_text().splitlines()]
        calls[0]["usd"] = None if defect == "unknown" else 2.0
        path.write_text("\n".join(json.dumps(call) for call in calls) + "\n")
    assert pilot_gates(run_dir)["passed"] is False


@pytest.mark.anyio
async def test_order_failure_stops_immediately_and_keeps_unknown_usage_reserve(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    with pytest.raises(DecisionError, match="stopped after an errored search"):
        await recorded_run(tmp_path, order_data, fake_decision_model, fail=True)
    run_dir = tmp_path / "runs/synthetic"
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1
    assert calls[0]["usd"] is None
    assert calls[0]["budget_charge_usd"] > 0
    assert calls[0]["request"]["questions"]
    assert json.loads((run_dir / "manifest.json").read_text())["completed"] is False
    assert pilot_gates(run_dir)["passed"] is False


@pytest.mark.anyio
async def test_a_refusal_is_the_model_s_answer_the_run_goes_on_and_the_report_leaves_its_task_out(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    run_dir, guard = await recorded_run(tmp_path, order_data, fake_decision_model, refuse=True)

    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    refused = [row for row in records if row.get("refused")]
    assert {(row["decider"], row["task"]) for row in refused} == {("jev", refused[0]["task"])}
    assert {row["order_seed"] for row in refused} == set(ORDER_SEEDS)  # every order of that task, and the run went on
    refusals = [call for call in guard.calls if call["error"] == "DecisionRefused"]
    assert len(refusals) == len(refused)
    assert all(call["usd"] is not None and call["input_tokens"] == 10 for call in refusals)  # billed at its usage
    gates = pilot_gates(run_dir)
    assert gates["passed"] is True
    assert (gates["errors"], gates["refused"]) == (0, len(refused))
    summary = build_order_report(run_dir, out_dir=tmp_path / "report", published_summary=None)
    assert summary["refused_groups"] == len({(row["task"], row["variant"]) for row in refused})
    assert summary["excluded_incomplete_or_error_groups"] == summary["refused_groups"]
    assert "refused by the model in at least one order" in (tmp_path / "report/README.md").read_text()


@pytest.mark.anyio
async def test_order_mode_refuses_decision_replays_before_any_asks(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    fake = fake_decision_model(limits=JEV_LIMITS)
    cached = CachedDecisionModel(fake, path=tmp_path / "decisions.sqlite")
    arms = build_decision_arms(["jev"], ks=[20], models={"jev": cached}, max_detail=ORDER_DETAILS)
    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, order_data.tasks, seed=0)
    try:
        with pytest.raises(ValueError, match="bypass local cache"):
            await run_decisions(
                arms,
                order_data,
                order_data.tasks,
                retriever=SharedRetrieval(CatalogRetriever()),
                split="heldout",
                sources=["plain"],
                model_queries=None,
                negatives=True,
                repeat=1,
                out_dir=tmp_path / "runs",
                task_file=task_file,
                model_queries_file=None,
                order_seeds=ORDER_SEEDS,
            )
        assert not fake.asks
    finally:
        cached.close()


@pytest.mark.anyio
async def test_concurrent_round_attempts_share_a_serial_budget_gate(fake_decision_model: Any) -> None:
    fake = fake_decision_model(model_id=JEV_MODEL, limits=JEV_LIMITS, fail_on="bad")
    guard = SpendGuard(prior_usd=6.998, cap_usd=7.0)
    wrapper = OrderDecisionModel(fake, decider="jev", guard=guard)
    # One full-cap reservation fits only in the first attempt; its unknown failure consumes the remaining room.
    guard.prior_usd = 7.0 - 0.003
    from toolhunch.decision import ChoiceQuestion, DecisionRequest

    request = DecisionRequest(
        state="bad", questions={"tool": ChoiceQuestion(instructions="Pick", options={"a": "a", "b": "b"})}
    )
    failures: list[type[Exception]] = []

    async def ask() -> None:
        try:
            await wrapper.ask(request)
        except (DecisionError, SpendLimit) as error:
            failures.append(type(error))

    async with anyio.create_task_group() as group:
        group.start_soon(ask)
        group.start_soon(ask)
    assert failures == [DecisionError, SpendLimit]
    assert len(guard.calls) == 1
    [entry] = order_ledger_entries(guard, run_id="failed", pilot=True)
    assert entry.usd == guard.run_usd
    assert entry.purpose == "P1: order sensitivity pilot"


def test_order_cli_rejects_incompatible_config_before_loading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_file = tmp_path / "tasks.json"
    task_file.write_text("{}")

    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("dataset loading must not occur")

    monkeypatch.setattr(cli, "load_toolret", forbidden)
    result = CliRunner().invoke(
        cli.app,
        [
            "decision",
            "--tasks",
            str(task_file),
            "--split",
            "heldout",
            "--deciders",
            "jev,logprob",
            "--k",
            "50",
            "--sources",
            "plain",
            "--order-sensitivity",
            "--dry-run",
        ],
    )
    assert result.exit_code == 2
    assert "order mode requires" in result.output


@pytest.mark.anyio
async def test_zero_spend_dry_run_estimates_pilot_and_full_with_prior_charge(
    tmp_path: Path,
    order_data: ToolRetData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = tuple(replace(order_data.tasks[i % 2], id=f"query_{i}") for i in range(200))
    data = replace(order_data, tasks=tasks)
    task_file = tmp_path / "heldout.json"
    write_task_file(task_file, tasks, seed=0)
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    (cache_dir / "corpus.parquet").touch()
    for name in TOOLRET_SUBTASKS:
        for kind in ("qrels", "queries"):
            (cache_dir / f"{name}-{kind}.parquet").touch()
    summary_path = tmp_path / "results/2026-09-toolret-decision/summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "role": "main",
                        "split": "heldout",
                        "manifest": {
                            "tasks": {"sha256": hashlib.sha256(task_file.read_bytes()).hexdigest()},
                            "dataset": {"catalog_fingerprint": data.catalog.fingerprint},
                        },
                    }
                ]
            }
        )
    )
    monkeypatch.setattr(order_module, "BENCH_DIR", tmp_path)

    def local_data(**kwargs: Any) -> ToolRetData:
        return data

    monkeypatch.setattr(order_module, "load_toolret", local_data)

    class FreeRetrieval(CatalogRetriever):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.searched: set[tuple[str, ...]] = set()
            self.stand_in_queries: set[tuple[str, ...]] = set()

        async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
            self.searched.add(tuple(queries))
            return await super().retrieve(queries, catalog, k=k)

    monkeypatch.setattr(order_module, "CacheOnlyRetrieval", FreeRetrieval)

    async def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run must never ask a provider")

    def forbidden_dotenv(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a dry run must not load keys")

    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", forbidden_dotenv)
    monkeypatch.setattr(OpenAIEmbedder, "embed", forbidden)
    monkeypatch.setattr(OpenAILogprobModel, "ask", forbidden)
    monkeypatch.setattr(order_module.OrderDecisionModel, "ask", forbidden)
    from toolhunch.decision import JevWireModel

    monkeypatch.setattr(JevWireModel, "ask", forbidden)
    ledger = tmp_path / "ledger.jsonl"
    output: list[str] = []
    result = await order_module.order_experiment(
        task_file,
        cache_dir=cache_dir,
        runs_dir=tmp_path / "runs",
        ledger_path=ledger,
        dry_run=True,
        pilot=False,
        pilot_run=None,
        estimate_out=tmp_path / "estimate.json",
        echo=output.append,
        embedding_cache_path=tmp_path / "embeddings.sqlite",
    )
    assert result is None
    assert not ledger.exists()
    estimate = json.loads((tmp_path / "estimate.json").read_text())
    assert estimate["pilot"]["searches"] == 200
    assert estimate["full"]["searches"] == 4000
    assert estimate["pilot_full_plus_prior_usd"] == estimate["full"]["usd"] + estimate["pilot"]["usd"]
    assert "zero provider calls" in "\n".join(output)


@pytest.mark.anyio
async def test_identity_drift_requires_matching_published_raw_population_and_payload(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    run_dir, _ = await recorded_run(tmp_path, order_data, fake_decision_model)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["arms"]["historical-local"] = {"decider": "clm", "model_id": "local@private.example"}
    public_manifest = copy.deepcopy(manifest)
    public_manifest["arms"]["historical-local"]["model_id"] = "local@modal"
    rows = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    identities = [
        row for row in rows if row.get("record") == "search" and row.get("decider") and row["order_seed"] == 0
    ]
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "manifest.json").write_text(json.dumps(manifest))
    (reference / "run.jsonl").write_text("\n".join(json.dumps(row) for row in identities) + "\n")
    published = tmp_path / "published.json"
    published.write_text(
        json.dumps(
            {
                "runs": [{"role": "main", "split": "heldout", "manifest": public_manifest}],
                "heldout": {
                    "run_id": "reference",
                    "rows": [
                        {"arm": f"hybrid+{name}@20", "source": "plain", "p_at_1": 0.5} for name in ("jev", "logprob")
                    ],
                },
            }
        )
    )
    summary = build_order_report(
        run_dir, out_dir=tmp_path / "matched", published_summary=published, reference_run=reference
    )
    assert summary["deciders"]["jev"]["identity_drift"]["comparable"] is True
    published_data = json.loads(published.read_text())
    published_data["runs"][0]["manifest"]["versions"]["python"] = "genuinely different"
    published.write_text(json.dumps(published_data))
    summary = build_order_report(
        run_dir, out_dir=tmp_path / "manifest-changed", published_summary=published, reference_run=reference
    )
    assert summary["deciders"]["jev"]["identity_drift"]["comparable"] is False
    published_data["runs"][0]["manifest"] = public_manifest
    published.write_text(json.dumps(published_data))
    identities[0]["context"] = "Changed query context"
    (reference / "run.jsonl").write_text("\n".join(json.dumps(row) for row in identities) + "\n")
    summary = build_order_report(
        run_dir, out_dir=tmp_path / "changed", published_summary=published, reference_run=reference
    )
    assert summary["deciders"]["jev"]["identity_drift"]["comparable"] is False


@pytest.mark.anyio
async def test_report_exposes_first_slot_bias_and_averages_by_card_id(
    tmp_path: Path,
    order_data: ToolRetData,
    fake_decision_model: Any,
) -> None:
    run_dir, _ = await recorded_run(tmp_path, order_data, fake_decision_model)
    path = run_dir / "run.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    positives: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record") != "search" or row.get("decider") != "jev":
            continue
        row["ranked"] = row["candidates"]
        row["probabilities"] = dict.fromkeys(row["candidates"], 0.0)
        row["probabilities"][row["candidates"][0]] = 1.0
        row["none_probability"], row["abstained"] = 0.0, False
        exchange = row["exchanges"][0]
        probabilities = dict.fromkeys(exchange["response"]["answers"]["tool"], 0.0)
        probabilities[next(iter(exchange["option_card_ids"]))] = 1.0
        exchange["response"]["answers"]["tool"] = probabilities
        if row["variant"] == "positive":
            positives.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = build_order_report(run_dir, out_dir=tmp_path / "position", published_summary=None)
    jev = summary["deciders"]["jev"]
    assert jev["chosen_slots_positive"]["slot_1"]["value"] == 1.0
    assert jev["chosen_slots_positive"]["chi_square"] == pytest.approx(152)
    assert jev["top_card_stability"]["value"] == 0.0
    shuffled = [row for row in positives if row["order_seed"] != 0]
    assert jev["shuffle_p_at_1"]["mean"]["value"] == sum(row["ranked"][0] in row["relevant"] for row in shuffled) / len(
        shuffled
    )
    averaged_correct = 0
    for task in order_data.tasks:
        own = [row for row in positives if row["task"] == task.id]
        cards = own[0]["candidates"]
        chosen = max(cards, key=lambda card: sum(row["ranked"][0] == card for row in own))
        averaged_correct += chosen in task.relevant
    assert jev["permutation_averaging"]["p_at_1"]["value"] == averaged_correct / len(order_data.tasks)


@pytest.mark.anyio
async def test_order_call_prices_reported_cache_reads_but_keeps_list_price(fake_decision_model: Any) -> None:
    from toolhunch.decision import ChoiceQuestion, DecisionRequest
    from toolhunch_bench.direct_cost import openai_usd

    fake = fake_decision_model(model_id=LOGPROB_MODEL, limits=LOGPROB_LIMITS)
    request = DecisionRequest(
        state="weather", questions={"tool": ChoiceQuestion(instructions="Pick", options={"a": "a", "b": "b"})}
    )
    response = await fake.ask(request)
    response = replace(response, raw={"usage": {"prompt_tokens_details": {"cached_tokens": 8}}})

    async def cached_response(request: Any, **kwargs: Any) -> Any:
        return response

    fake.ask = cached_response
    guard = SpendGuard()
    await OrderDecisionModel(fake, decider="logprob", guard=guard).ask(request)
    [call] = guard.calls
    assert call["cache_read_tokens"] == 8
    assert call["usd"] == openai_usd(model=LOGPROB_MODEL, input_tokens=10, output_tokens=1, cache_read_tokens=8)
    assert call["list_usd"] == openai_usd(model=LOGPROB_MODEL, input_tokens=10, output_tokens=1)
    assert call["list_usd"] > call["usd"]


@pytest.fixture
def noise_data(order_data: ToolRetData) -> ToolRetData:
    cards = [
        ToolCard(id=f"zz_{card.id}", name=card.name, description=card.description)
        if card.id.startswith("padding_")
        else card
        for card in order_data.catalog
    ]
    return replace(order_data, catalog=ToolCatalog(cards))


def noise_reference(run_dir: Path, target: Path) -> Path:
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text())
    manifest |= {"run_id": target.name, "repeat": 3, "git": {"commit": "prior", "dirty": True}}
    for name in ("jev", "logprob"):
        manifest["arms"][f"hybrid+{name}@20"]["model"] = "CachedDecisionModel(..., bypass=True)"
    rows: list[dict[str, Any]] = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    repeats: list[dict[str, Any]] = []
    for row in rows:
        if row.get("record") != "search" or not row.get("decider") or row["variant"] != "positive" or row["order_seed"]:
            continue
        correct = next(card for card in row["candidates"] if card in row["relevant"])
        wrong = next(card for card in row["candidates"] if card not in row["relevant"])
        for index in range(3):
            replica = copy.deepcopy(row)
            chosen = wrong if row["task"] == manifest["task_ids"][0] and index == 2 else correct
            replica["repeat"] = index
            replica["ranked"] = [chosen, *[card for card in row["candidates"] if card != chosen]]
            repeats.append(replica)
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(manifest))
    (target / "run.jsonl").write_text("\n".join(json.dumps(row) for row in repeats) + "\n")
    return target


@pytest.mark.anyio
async def test_report_generates_matched_same_order_noise_with_task_cluster_intervals(
    tmp_path: Path, noise_data: ToolRetData, fake_decision_model: Any
) -> None:
    run_dir, _ = await recorded_run(tmp_path, noise_data, fake_decision_model)
    reference = noise_reference(run_dir, tmp_path / "noise")
    summary = build_order_report(
        run_dir, out_dir=tmp_path / "noise-report", published_summary=None, noise_reference_run=reference
    )
    noise = summary["same_order_noise_baseline"]
    assert noise["reference_git"] == {"commit": "prior", "dirty": True}
    assert noise["reference_records_sha256"] == hashlib.sha256((reference / "run.jsonl").read_bytes()).hexdigest()
    assert noise["repeat_count"] == 3
    for name in ("jev", "logprob"):
        row = noise["deciders"][name]
        assert row["comparable"] is True
        assert row["matched_positive_tasks"] == 2
        assert row["pairwise_agreement"]["value"] == pytest.approx(2 / 3)
        assert row["pairwise_agreement"]["ci95"] == pytest.approx([1 / 3, 1])
        assert [value["value"] for value in row["p_at_1_by_repeat"].values()] == [1, 1, 0.5]
        assert row["repeat_p_at_1_bounds"]["min"]["value"] == 0.5
        assert row["repeat_p_at_1_bounds"]["max"]["value"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("defect", ["model", "candidate_order", "missing_repeat", "error", "query"])
async def test_noise_baseline_refuses_non_matching_or_incomplete_repeats(
    tmp_path: Path, noise_data: ToolRetData, fake_decision_model: Any, defect: str
) -> None:
    run_dir, _ = await recorded_run(tmp_path, noise_data, fake_decision_model)
    reference = noise_reference(run_dir, tmp_path / "noise")
    if defect == "model":
        path = reference / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["arms"]["hybrid+jev@20"]["model_id"] = "another-model"
        path.write_text(json.dumps(manifest))
    else:
        path = reference / "run.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if defect == "candidate_order":
            rows[0]["candidates"].reverse()
        elif defect == "missing_repeat":
            rows.pop(0)
        elif defect == "error":
            rows[0]["error"] = "provider failed"
        else:
            rows[0]["queries"] = ["different"]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = build_order_report(
        run_dir, out_dir=tmp_path / "noise-report", published_summary=None, noise_reference_run=reference
    )
    assert summary["same_order_noise_baseline"]["deciders"]["jev"]["comparable"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("names", [("strands", "clef-flash"), ("strands",)], ids=["two", "one"])
async def test_order_report_covers_any_decider(
    tmp_path: Path, order_data: ToolRetData, fake_decision_model: Any, names: tuple[str, ...]
) -> None:
    # Strands runs locally and Clef-flash is billed per declared token: neither is one of P1's two deciders. P2 asks
    # one decider per order run (one local server at a time), so the gates count the deciders the run asked.
    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, order_data.tasks, seed=0)
    run_dir = tmp_path / "runs/p2"

    def persist(call: dict[str, Any]) -> None:
        with (run_dir / "calls.jsonl").open("a") as journal:
            journal.write(json.dumps(call) + "\n")

    guard = SpendGuard(prior_usd=0.5, cap_usd=BUDGET.cap_usd, sink=persist)
    fakes = {
        "strands": fake_decision_model(
            model_id="strands-decider-2B-hobson-v19@127.0.0.1:8000", limits=STRANDS_LIMITS, favourite="get_weather"
        ),
        "clef-flash": fake_decision_model(
            model_id="clef-flash@api.cloudflare.com", limits=CLEF_FLASH_LIMITS, favourite="get_weather"
        ),
    }
    fakes = {name: fakes[name] for name in names}
    details = {name: {"strands": DetailLevel.BRIEF, "clef-flash": DetailLevel.FULL}[name] for name in names}
    models = {name: OrderDecisionModel(fake, decider=name, guard=guard) for name, fake in fakes.items()}
    arms = build_decision_arms(list(fakes), ks=[20], models=models, max_detail=details)
    await run_decisions(
        arms,
        order_data,
        order_data.tasks,
        retriever=SharedRetrieval(CatalogRetriever()),
        split="heldout",
        sources=["plain"],
        model_queries=None,
        negatives=True,
        repeat=1,
        out_dir=tmp_path / "runs",
        task_file=task_file,
        model_queries_file=None,
        run_id="p2",
        order_seeds=ORDER_SEEDS,
        manifest_extra={
            "completed": True,
            "experiment": "order-sensitivity-v1",
            "pilot": True,
            "estimate": {"usd": 0.01},
            "prior_p1_usd": 0.5,
            "cap_usd": BUDGET.cap_usd,
        },
        before_search=lambda context: setattr(guard, "context", context),
    )

    gates = pilot_gates(run_dir)
    assert gates["expected_searches"] == len(order_data.tasks) * len(names) * 2 * len(ORDER_SEEDS)
    assert gates["passed"] is True
    summary = build_order_report(run_dir, out_dir=tmp_path / "report", published_summary=None)
    assert list(summary["deciders"]) == list(names)
    # The README names each decider's configuration and its caveats; an order run reports no latency, so no machine.
    readme = (tmp_path / "report" / "README.md").read_text()
    configured = " and ".join(
        {"strands": "Strands Decider 2B BRIEF", "clef-flash": "Clef-flash FULL"}[n] for n in names
    )
    assert f"K=20, plain queries, {configured}, reserved option last." in readme
    assert ("**Clef-flash.** Workers AI serves the current Clef-flash" in readme) == ("clef-flash" in names)
    assert "**Local deciders.**" not in readme
    # A local decider's cost reads "local", never $0 (spec D9), in its section and, alone in a run, in the run cost.
    assert "measured decision cost per 1,000 searches local;" in readme
    assert ("Verified run cost local." in readme) == (names == ("strands",))
    assert summary["deciders"]["strands"]["top_card_stability"]["value"] == 1.0
    assert summary["deciders"]["strands"]["verified_decision_usd"] == 0
    if "clef-flash" in names:
        clef_calls = [call for call in guard.calls if call["provider"] == "cloudflare"]
        assert summary["deciders"]["clef-flash"]["verified_decision_usd"] == pytest.approx(
            sum(call["input_tokens"] for call in clef_calls) * 0.09 / 1_000_000
        )
    # Neither has published F2a evidence or a same-order repeats run: both say so instead of borrowing Jev's.
    assert not summary["deciders"]["strands"]["identity_drift"]["comparable"]
    assert set(summary["same_order_noise_baseline"]["deciders"]) == set(names)
    # Only the paid decider reaches the ledger, under the current phase.
    entries = order_ledger_entries(guard, run_id="p2", pilot=True, budget=BUDGET)
    paid = [("cloudflare", f"{BUDGET.prefix} order sensitivity pilot")] if "clef-flash" in names else []
    assert [(entry.provider, entry.purpose) for entry in entries] == paid


@pytest.mark.anyio
async def test_order_model_charges_an_unpriceable_reply(fake_decision_model: Any) -> None:
    # A priced decider whose reply reports no input tokens (a renamed usage field defaults to 0) is a failure charged
    # at its reservation, never a $0 success (Review Focus 2).
    base = fake_decision_model(model_id="clef-flash@api.cloudflare.com", limits=CLEF_FLASH_LIMITS, favourite="weather")

    class Unpriced:
        model_id, limits, question_kinds, prompt_version = (
            base.model_id,
            base.limits,
            base.question_kinds,
            base.prompt_version,
        )

        async def ask(self, request: DecisionRequest, /, **options: Any) -> DecisionResponse:
            return replace(await base.ask(request, **options), usage=DecisionUsage(1, 0, 0))

    guard = SpendGuard(prior_usd=0.0, cap_usd=BUDGET.cap_usd)
    model = OrderDecisionModel(Unpriced(), decider="clef-flash", guard=guard)
    request = DecisionRequest(
        state="Request: weather in Rome",
        questions={"tool": ChoiceQuestion(instructions="Which?", options={"a": "weather", "none": "None fits."})},
    )

    with pytest.raises(DecisionError):
        await model.ask(request)

    [call] = guard.calls
    assert call["error"] == "priced reply without input tokens"
    assert call["usd"] is None
    assert call["budget_charge_usd"] == pytest.approx(65_536 * 0.09 / 1_000_000)
