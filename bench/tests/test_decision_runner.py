import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import httpx2
import pytest
from genai_prices import Usage, calc_price
from inline_snapshot import snapshot

from toolhunch import (
    BM25Retriever,
    DenseRetriever,
    DetailLevel,
    HybridRetriever,
    ScoredCard,
    ToolCard,
    ToolCatalog,
    default_search_text,
)
from toolhunch.decision import (
    CLM_LIMITS,
    JEV_LIMITS,
    LOGPROB_LIMITS,
    ChoiceQuestion,
    DecisionRequest,
    ModelLimits,
    clm,
)
from toolhunch.decision import decider as decider_module
from toolhunch.decision.planner import NONE_KEY
from toolhunch.retrieval import Retrieval
from toolhunch_bench import decision as decision_module
from toolhunch_bench.datasets.model_queries import WrittenQueries, save_model_queries
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask, write_task_file
from toolhunch_bench.decision import (
    CLM_DEPLOYMENT,
    JEV_REQUEST_OVERHEAD_TOKENS,
    LOGPROB_MODEL,
    CacheOnlyRetrieval,
    DecisionArm,
    EstimateLine,
    ExcludingRetriever,
    SharedRetrieval,
    Split,
    build_decision_arms,
    estimate_decisions,
    run_decisions,
    warm_up_clm,
)
from toolhunch_bench.decision_cache import CachedDecisionModel
from toolhunch_bench.decision_report import build_decision_report
from toolhunch_bench.embedding_cache import CachedEmbedder
from toolhunch_bench.retrieval import build_arms

pytestmark = pytest.mark.anyio


class RecordingRetriever:
    """Passes every request to `inner` and records its queries and `k`."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[tuple[tuple[str, ...], int]] = []

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        self.calls.append((tuple(queries), k))
        return await self.inner.retrieve(queries, catalog, k=k)


def ids(retrieval: Retrieval) -> list[str]:
    return [match.card.id for match in retrieval.matches]


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return manifest, [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]


# What an agent searched for on each task; "mail my boss" made no search, so its queries are the request itself.
WRITTEN = {
    "t_query_0": WrittenQueries("t_query_0", ("weather forecast", "rain alerts"), fallback=False),
    "t_query_1": WrittenQueries("t_query_1", ("mail my boss",), fallback=True),
    "t_query_2": WrittenQueries("t_query_2", ("diary entry",), fallback=False),
    "t_query_3": WrittenQueries("t_query_3", ("get user", "user lookup"), fallback=False),
}


async def test_excluding_retriever_drops_gold_and_keeps_k(toolret_data: ToolRetData) -> None:
    catalog = toolret_data.catalog
    inner = RecordingRetriever(BM25Retriever())
    full = ids(await inner.retrieve(["weather forecast"], catalog, k=7))
    gold = {"web_tool_0", "web_tool_1"}  # get_weather and weather_alerts, the two best matches
    assert gold <= set(full[:2])

    excluding = ExcludingRetriever(inner, exclude=gold)
    kept = ids(await excluding.retrieve(["weather forecast"], catalog, k=5))

    assert inner.calls[-1] == (("weather forecast",), 7)  # k plus one per excluded id
    assert kept == [card_id for card_id in full if card_id not in gold][:5]  # the rest, in retrieval order
    # An excluded id the retriever does not return leaves the first k as they are.
    far = ExcludingRetriever(inner, exclude={"web_tool_17"})
    assert ids(await far.retrieve(["weather forecast"], catalog, k=3)) == full[:3]


async def test_shared_retrieval_calls_the_inner_retriever_once_per_query(axis_embedder: Any) -> None:
    topics = ("weather", "email", "calendar", "diary", "image", "user")
    catalog = ToolCatalog(
        ToolCard(id=f"tool_{i:03}", name=f"tool_{i:03}", description=f"{topics[i % 6]} and {topics[i * 7 % 5]} {i}")
        for i in range(120)
    )

    def hybrid() -> HybridRetriever:
        return HybridRetriever([BM25Retriever(), DenseRetriever(axis_embedder())])

    inner = RecordingRetriever(hybrid())
    shared = SharedRetrieval(inner)

    top_20 = await shared.retrieve(["weather email"], catalog, k=20)
    top_50 = await shared.retrieve(["weather email"], catalog, k=50)
    await shared.retrieve(["diary"], catalog, k=20)
    await shared.retrieve(["weather email"], catalog, k=20)

    assert inner.calls == [(("weather email",), 100), (("diary",), 100)]
    # The first k of the depth-100 retrieval are what the hybrid retriever returns for k itself.
    fresh = hybrid()
    assert ids(top_20) == ids(await fresh.retrieve(["weather email"], catalog, k=20))
    assert ids(top_50) == ids(await fresh.retrieve(["weather email"], catalog, k=50))
    assert shared.first_seconds(["weather email"]) > 0
    # Deeper than what was kept: the inner retriever is asked again, at the new depth.
    assert len((await shared.retrieve(["diary"], catalog, k=110)).matches) == 110
    assert inner.calls[-1] == (("diary",), 110)


@pytest.fixture
def run_files(tmp_path: Path, toolret_data: ToolRetData) -> tuple[Path, Path]:
    """The task file and the model-queries file of `toolret_data`'s tasks."""
    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, toolret_data.tasks, seed=0)
    queries_file = tmp_path / "tasks.model-queries.json"
    save_model_queries(queries_file, list(WRITTEN.values()), writer={"model": "writer@test"})
    return task_file, queries_file


async def test_run_records_every_arm_source_variant_and_the_negatives_exclude_gold(
    tmp_path: Path,
    toolret_data: ToolRetData,
    axis_embedder: Any,
    fake_decision_model: Any,
    run_files: tuple[Path, Path],
) -> None:
    data = toolret_data
    task_file, queries_file = run_files
    [hybrid] = build_arms(["hybrid"], embedder=axis_embedder(), raw_text=data.raw_text)
    inner = RecordingRetriever(hybrid.retriever)
    fakes = {
        "jev": fake_decision_model(model_id="jev-fake@test", favourite="get_weather"),
        "clm": fake_decision_model(model_id="clm-fake@test"),
    }
    models = {name: CachedDecisionModel(fake, path=tmp_path / f"{name}.sqlite") for name, fake in fakes.items()}
    arms = build_decision_arms(["jev", "clm"], ks=[3, 5], models=models, max_detail={"clm": DetailLevel.BRIEF})

    run_dir = await run_decisions(
        arms,
        data,
        data.tasks,
        retriever=SharedRetrieval(inner, config=hybrid.config),
        split="dev",
        sources=("plain", "model"),
        model_queries=WRITTEN,
        negatives=True,
        repeat=1,
        out_dir=tmp_path / "runs",
        task_file=task_file,
        model_queries_file=queries_file,
        run_id="run-1",
    )

    manifest, records = read_run(run_dir)
    names = ["hybrid@3", "hybrid+jev@3", "hybrid+clm@3", "hybrid@5", "hybrid+jev@5", "hybrid+clm@5"]
    assert [arm.name for arm in arms] == names
    # Arm-major: each arm's 16 searches (2 sources x 4 tasks x 2 variants), then its own record.
    assert [record["arm"] for record in records] == [name for name in names for _ in range(17)]
    arm_records = [record for record in records if record["record"] == "arm"]
    assert [(r["arm"], r["searches"], r["errors"]) for r in arm_records] == [(name, 16, 0) for name in names]
    searches = [record for record in records if record["record"] == "search"]
    assert len(searches) == len(arms) * 2 * 4 * 2
    # Within an arm, a request asked before comes from the cache: a fallback repeats the plain request, and a
    # negative whose positive had no gold among its candidates repeats the positive's. Retrieval alone asks nothing.
    for record in arm_records:
        own = [search for search in searches if search["arm"] == record["arm"]]
        requests = {(tuple(search["queries"]), search["context"], tuple(search["candidates"])) for search in own}
        if record["arm"].startswith("hybrid@"):
            assert record["cache_hits"] is record["cache_misses"] is None
        else:
            assert (record["cache_misses"], record["cache_hits"]) == (len(requests), len(own) - len(requests))
            assert record["cache_hits"] > 0

    for record in searches:
        relevant = set(record["relevant"])
        assert record["context"] == data.tasks[int(record["task"][-1])].query
        assert record["gold_in_candidates"] == bool(relevant & set(record["candidates"]))
        if record["variant"] == "negative":
            assert not relevant & set(record["candidates"])
            assert len(record["candidates"]) == record["k"]
        if record["source"] == "model":
            written = WRITTEN[record["task"]]
            assert (record["queries"], record["fallback"]) == (list(written.queries), written.fallback)
        else:
            assert (record["queries"], record["fallback"]) == ([record["context"]], None)
        if record["decider"] is None:
            assert record["ranked"] == record["candidates"]
            assert record["probabilities"] == {}
            assert record["key"] is record["abstained"] is record["exchanges"] is record["usage"] is None
        else:
            assert sorted(record["ranked"]) == sorted(record["candidates"])
            assert record["key"].startswith(f"{record['decider']}-fake@test|tool-choice-v1|")
    assert sum(record["gold_in_candidates"] for record in searches if record["variant"] == "positive") > 0
    # Every arm, source and variant searched one list of queries through one retrieval: 4 plain lists and 3 written
    # ones (the fallback's is the plain one), each at depth 100.
    assert sorted(inner.calls) == sorted({(tuple(record["queries"]), 100) for record in searches})
    assert len(inner.calls) == 7

    [weather] = [
        r
        for r in searches
        if (r["arm"], r["source"], r["variant"], r["task"]) == ("hybrid+jev@3", "model", "positive", "t_query_0")
    ]
    assert weather["retrieval_seconds"] > 0
    del weather["retrieval_seconds"]
    assert weather == snapshot(
        {
            "record": "search",
            "arm": "hybrid+jev@3",
            "decider": "jev",
            "k": 3,
            "source": "model",
            "split": "dev",
            "variant": "positive",
            "repeat": 0,
            "task": "t_query_0",
            "queries": ["weather forecast", "rain alerts"],
            "fallback": False,
            "context": "will it rain in Rome",
            "candidates": ["web_tool_0", "web_tool_1", "web_tool_12"],
            "relevant": ["web_tool_0", "web_tool_1"],
            "gold_in_candidates": True,
            "ranked": ["web_tool_0", "web_tool_1", "web_tool_12"],
            "probabilities": {
                "web_tool_0": 0.7272727272727273,
                "web_tool_1": 0.09090909090909091,
                "web_tool_12": 0.09090909090909091,
            },
            "none_probability": 0.09090909090909091,
            "abstained": False,
            "key": "jev-fake@test|tool-choice-v1|1edb74d749cc426c|choice",
            "shape": {
                "candidates": 3,
                "questions": [[4]],
                "reserved_option": True,
                "max_detail": "FULL",
                "finalists_per_chunk": 2,
                "questions_per_request": 1,
                "option_keys": "names-v1",
            },
            "state_cut": False,
            "exchanges": [
                {
                    "round": 1,
                    "detail": "FULL",
                    "estimated_input_tokens": 134,
                    "input_tokens": 10,
                    "output_tokens": 1,
                    "seconds": 0.1,
                    "server_seconds": 0.01,
                    "key_only_options": [],
                }
            ],
            "usage": {"requests": 1, "input_tokens": 10, "output_tokens": 1},
            "decision_seconds": 0.1,
            "decision_sequential_seconds": 0.1,
            "server_seconds": 0.01,
            "not_applicable": None,
            "failed_card": None,
            "error": None,
            "refused": False,
        }
    )

    assert manifest["split"] == "dev"
    assert manifest["sources"] == ["plain", "model"]
    assert (manifest["negatives"], manifest["repeat"]) == (True, 1)
    assert manifest["model_queries"] == {
        "file": queries_file.name,  # outside the repository: the name alone
        "sha256": hashlib.sha256(queries_file.read_bytes()).hexdigest(),
        "writer": {"model": "writer@test"},
    }
    assert {name: arm["model_id"] for name, arm in manifest["arms"].items()} == {
        "hybrid@3": None,
        "hybrid+jev@3": "jev-fake@test",
        "hybrid+clm@3": "clm-fake@test",
        "hybrid@5": None,
        "hybrid+jev@5": "jev-fake@test",
        "hybrid+clm@5": "clm-fake@test",
    }
    assert manifest["arms"]["hybrid+clm@5"] == snapshot(
        {
            "k": 5,
            "decider": "clm",
            "model_id": "clm-fake@test",
            "model": "CachedDecisionModel(FakeDecisionModel('clm-fake@test'), bypass=False)",
            "limits": {
                "max_options_per_choice": None,
                "max_request_tokens": None,
                "max_state_plus_question_tokens": None,
                "max_text_tokens": None,
                "max_question_tokens": None,
                "max_option_tokens": None,
                "max_questions_per_request": None,
                "score_levels": None,
                "price_input_per_mtok": None,
                "price_output_per_mtok": None,
                "source": "test",
                "checked": "2026-09-29",
            },
            "prompt_version": "tool-choice-v1",
            "max_detail": "BRIEF",
            "min_detail": "NAME",
            "tokenizer": "HeuristicTokenizer(bytes_per_token=3.0)",
            "reserved_option": True,
        }
    )
    assert manifest["retriever"] == {**hybrid.config, "shared_depth": 100}
    assert manifest["clm_deployment"] == CLM_DEPLOYMENT


async def test_a_decision_error_is_recorded_and_the_run_goes_on(
    tmp_path: Path,
    toolret_data: ToolRetData,
    axis_embedder: Any,
    fake_decision_model: Any,
    run_files: tuple[Path, Path],
) -> None:
    data = toolret_data
    task_file, _ = run_files
    [hybrid] = build_arms(["hybrid"], embedder=axis_embedder(), raw_text=data.raw_text)
    inner = RecordingRetriever(hybrid.retriever)
    jev = fake_decision_model(fail_on="boss")  # nothing usable for "mail my boss"
    arms = build_decision_arms(["jev"], ks=[3], models={"jev": jev}, max_detail={})

    run_dir = await run_decisions(
        arms,
        data,
        data.tasks,
        retriever=SharedRetrieval(inner, config=hybrid.config),
        split="heldout",
        sources=("plain",),
        model_queries=None,
        negatives=True,
        repeat=1,
        out_dir=tmp_path / "runs",
        task_file=task_file,
        model_queries_file=None,
    )

    manifest, records = read_run(run_dir)
    searches = [record for record in records if record["record"] == "search"]
    failed = [record for record in searches if record["error"] is not None]
    assert [(r["arm"], r["task"], r["variant"]) for r in failed] == [
        ("hybrid+jev@3", "t_query_1", "positive"),
        ("hybrid+jev@3", "t_query_1", "negative"),
    ]
    for record in failed:
        assert record["error"] == "fake@test: no option letter among the top logprobs"
        [baseline] = [
            r
            for r in searches
            if (r["arm"], r["task"], r["variant"]) == ("hybrid@3", record["task"], record["variant"])
        ]
        assert record["candidates"] == baseline["candidates"]  # the candidates the decider was given
        assert (record["ranked"], record["probabilities"], record["abstained"], record["key"]) == (None, {}, None, None)
    # The searches after the failed ones were decided.
    later = [r for r in searches if r["arm"] == "hybrid+jev@3" and r["task"] in ("t_query_2", "t_query_3")]
    assert len(later) == 4
    assert all(r["error"] is None and r["key"] is not None for r in later)
    [jev_arm] = [record for record in records if record["record"] == "arm" and record["arm"] == "hybrid+jev@3"]
    assert (jev_arm["searches"], jev_arm["errors"]) == (8, 2)
    assert len(inner.calls) == 4  # the failed searches' candidates came from the shared retrieval
    assert (manifest["split"], manifest["sources"], manifest["model_queries"]) == ("heldout", ["plain"], None)
    assert manifest["clm_deployment"] is None  # no CLM arm


class GoldFirst:
    """Ranks a request's gold tools first, then every other tool in catalog order, leaving `hidden` out."""

    def __init__(self, tasks: Sequence[ToolRetTask], *, hidden: str) -> None:
        self._gold = {task.query: sorted(task.relevant) for task in tasks}
        self._hidden = hidden

    async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
        gold = self._gold[queries[0]]
        cards = {card.id: card for card in catalog}
        rest = [card_id for card_id in cards if card_id not in gold and card_id != self._hidden]
        return Retrieval(matches=tuple(ScoredCard(cards[card_id], 1.0) for card_id in [*gold, *rest][:k]))


async def test_lists_two_rounds_cannot_hold_are_not_applicable_and_a_card_over_the_cap_fails(
    tmp_path: Path, fake_decision_model: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Ten tools whose names take 5 heuristic tokens, and one whose key alone takes 19: the gold tool of the last task.
    pdf = ToolCard(id="pdf", name="convert_a_portable_document_into_plain_text", description="PDF to text.")
    catalog = ToolCatalog([*(ToolCard(id=f"t{i}", name=f"tool_number_{i:02d}") for i in range(10)), pdf])
    tasks = tuple(
        ToolRetTask(f"q{i}", "t", f"request {i}", "Retrieve tools", frozenset({gold}))
        for i, gold in enumerate(["t0", "t1", "t2", "t3", "pdf"])
    )
    data = ToolRetData(catalog=catalog, raw_text={}, tasks=tasks, mapping_stats={})
    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, tasks, seed=0)

    async def run(
        model: Any, *, k: int, searched: Sequence[ToolRetTask], run_id: str, split: Split = "heldout"
    ) -> tuple[Path, list[dict[str, Any]]]:
        arms = build_decision_arms(["strands"], ks=[k], models={"strands": model}, max_detail={})
        retriever = GoldFirst(tasks, hidden="pdf")
        [line] = await estimate_decisions(
            arms, data, searched, retriever=retriever, sources=("plain",), model_queries=None, negatives=True, repeat=1
        )
        run_dir = await run_decisions(
            arms,
            data,
            searched,
            retriever=SharedRetrieval(retriever),
            split=split,
            sources=("plain",),
            model_queries=None,
            negatives=True,
            repeat=1,
            out_dir=tmp_path / "runs",
            task_file=task_file,
            model_queries_file=None,
            run_id=run_id,
        )
        _, records = read_run(run_dir)
        decided = [record for record in records if record["record"] == "search" and record["decider"] == "strands"]
        assert (line.not_applicable, line.would_fail) == (
            sum(record["not_applicable"] is not None for record in decided),
            sum(record["error"] is not None for record in decided),
        )  # the estimate counted them as the run records them
        return run_dir, decided

    def report(heldout: Path, *, dev: Sequence[Path] = ()) -> tuple[dict[str, Any], str]:
        build_decision_report(list(dev), heldout, out_dir=tmp_path / f"{heldout.name}-report")
        summary = json.loads((tmp_path / f"{heldout.name}-report" / "summary.json").read_text())
        return summary, (tmp_path / f"{heldout.name}-report" / "README.md").read_text()

    # Three options a question hold two rounds of at most 3 x 2 candidates, with the reserved option in the final:
    # eight are not applicable, and nothing is asked. A dev run of the same searches is not applicable too.
    unasked = fake_decision_model(
        limits=ModelLimits(max_options_per_choice=3, source="test", checked=date(2026, 10, 5))
    )
    dev, _ = await run(unasked, k=8, searched=tasks[:4], run_id="too-many-dev", split="dev")
    heldout, decided = await run(unasked, k=8, searched=tasks[:4], run_id="too-many")
    summary, readme = report(heldout, dev=[dev])
    assert len(decided) == 8
    reason = "at most 6 candidates fit two rounds for this model"
    assert all(r["not_applicable"] == reason and r["error"] is r["failed_card"] is r["ranked"] is None for r in decided)
    assert unasked.asks == []
    unfit = {"status": "not applicable", "not_applicable": {"searches": 8, "reason": reason}}
    [row] = [row for row in summary["heldout"]["rows"] if row["arm"] == "hybrid+strands@8"]
    assert ({key: row[key] for key in unfit}, row["errors"]) == (unfit, 0)
    assert row["p_at_1"] is row["p_at_1_ci95"] is row["delta_p_at_1"] is None
    assert set(row["rules"].values()) == {None}
    [curve] = summary["heldout"]["risk_coverage"]
    assert ({key: curve[key] for key in unfit}, curve["grid"]) == (unfit, None)
    [cell] = [entry for entry in summary["dev"]["precision"] if entry["arm"] == "hybrid+strands@8"]
    assert ({key: cell[key] for key in unfit}, cell["p_at_1"]) == (unfit, None)
    [errors] = [entry for entry in summary["dev"]["errors"] if entry["arm"] == "hybrid+strands@8"]
    assert (errors["errors"], errors["not_applicable"]) == (0, unfit["not_applicable"])
    held_out_part, dev_part = readme.split("## Dev")
    assert "| not applicable (8) |" in held_out_part
    assert "| hybrid+strands@8 | plain | not applicable (8) |" in dev_part  # never a P@1 of the lists that fit
    assert f"8 searches not applicable: {reason}" in held_out_part

    # Under a 12-token option window every name would take its option past it, so each goes as its key alone; the PDF
    # tool's key alone is over it, so the one search that retrieves it fails. Four options a question split five
    # candidates into groups of three and two, asked in 0.3 and 0.1 s, and a final of the two winners and the reserved
    # option, asked at once. Each call takes 0.01 s by the server's clock.
    def seconds(request: DecisionRequest) -> float:
        [question] = request.questions.values()
        assert isinstance(question, ChoiceQuestion)
        return 0.0 if NONE_KEY in question.options else {3: 0.3, 2: 0.1}[len(question.options)]

    laya_like = ModelLimits(
        max_options_per_choice=4,
        max_question_tokens=100,
        max_option_tokens=12,
        source="test",
        checked=date(2026, 10, 5),
    )
    heldout, decided = await run(
        fake_decision_model(limits=laya_like, seconds=seconds), k=5, searched=tasks, run_id="key-alone"
    )
    summary, readme = report(heldout)
    [failed] = [record for record in decided if record["error"] is not None]
    assert (failed["task"], failed["variant"], failed["not_applicable"]) == ("q4", "positive", None)
    assert failed["failed_card"] == "pdf"
    assert failed["error"].startswith("card 'pdf': its option key alone takes 19 tokens, over max_option_tokens=12")
    answered = [record for record in decided if record["error"] is None]
    assert all(
        [(exchange["round"], exchange["seconds"]) for exchange in record["exchanges"]] == [(1, 0.3), (1, 0.1), (2, 0.0)]
        and (record["decision_seconds"], record["decision_sequential_seconds"]) == (0.3, pytest.approx(0.4))
        and record["failed_card"] is None
        for record in answered
    )
    [row] = [row for row in summary["heldout"]["rows"] if row["arm"] == "hybrid+strands@5"]
    assert "status" not in row
    assert (row["errors"], row["failed_card_searches"]) == (1, 1)
    assert (row["positives"], row["p_at_1"]) == (4, 1.0)  # the other positives keep their P@1
    # All 7 card options of each search went as their key alone (3 + 2 in round one, 2 in the final); the reserved
    # option is not a card.
    assert row["key_only_option_share"] == 1.0
    # Strands is asked one request at a time, so every latency adds its calls: a decision takes 0.3 + 0.1 + 0 s, not
    # its critical path of 0.3, and 3 x 0.01 s on the server; the arm's 9 decisions take 3.6 s.
    assert row["latency_ms"]["decision_basis"] == "sum of calls"
    assert row["latency_ms"]["decision"]["p50"] == pytest.approx(400.0)
    assert row["latency_ms"]["server"]["p50"] == pytest.approx(30.0)
    [arm] = [arm for arm in summary["heldout"]["arms"] if arm["arm"] == "hybrid+strands@5"]
    assert (arm["decision_seconds"], arm["decision_basis"]) == (pytest.approx(3.6), "sum of calls")
    assert "| sum of calls |" in readme
    assert "100.0% of the card options sent as their key alone; 1 failed search: card 'pdf'" in readme

    # A final question that does not fit after round one, which the plan rules out, fails the search that asked it:
    # it is not "not applicable", since round one's calls were made.
    def no_final(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(decider_module, "plan_question", no_final)
    late = fake_decision_model(limits=laya_like)
    _, decided = await run(late, k=5, searched=tasks[:1], run_id="late")
    assert len(late.asks) == 2 * 2  # two searches, each asking its two groups
    assert all(
        r["error"].startswith("the final question over the 2 finalists")
        and r["not_applicable"] is r["failed_card"] is None
        for r in decided
    )


class Clock:
    """What the runner reads of the `time` module, `perf_counter`, on a clock that only the test moves."""

    def __init__(self) -> None:
        self.now = 0.0

    def perf_counter(self) -> float:
        return self.now


async def test_a_hook_runs_before_each_arm_outside_its_wall_time(
    tmp_path: Path,
    toolret_data: ToolRetData,
    fake_decision_model: Any,
    run_files: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = toolret_data
    clock = Clock()
    monkeypatch.setattr(decision_module, "time", clock)
    events: list[str] = []

    class TimedRetrieval(SharedRetrieval):
        """Each retrieval takes a second on the clock."""

        async def retrieve(self, queries: Sequence[str], catalog: ToolCatalog, *, k: int) -> Retrieval:
            events.append("retrieve")
            clock.now += 1.0
            return await super().retrieve(queries, catalog, k=k)

    async def before_arm(arm: DecisionArm) -> None:
        events.append(arm.name)
        clock.now += 600.0  # a cold start

    arms = build_decision_arms(["jev"], ks=[3], models={"jev": fake_decision_model()}, max_detail={})

    run_dir = await run_decisions(
        arms,
        data,
        data.tasks,
        retriever=TimedRetrieval(BM25Retriever()),
        split="dev",
        sources=("plain",),
        model_queries=None,
        negatives=True,
        repeat=1,
        out_dir=tmp_path / "runs",
        task_file=run_files[0],
        model_queries_file=None,
        before_arm=before_arm,
    )

    _, records = read_run(run_dir)
    # The hook runs once per arm, before its first search: each of the 8 searches retrieves twice, for the search
    # and then for the record's candidates.
    assert events == [event for arm in arms for event in [arm.name, *["retrieve"] * 16]]
    # An arm's wall time is its 16 seconds of retrieval, without the 600 of its hook.
    assert [record["wall_seconds"] for record in records if record["record"] == "arm"] == [16.0, 16.0]


async def test_warm_up_waits_for_health_then_sends_one_request() -> None:
    seen: list[tuple[str, str]] = []
    health = iter([503, 503, 200])

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx2.Response(next(health))
        [(question_id, question)] = json.loads(request.content)["questions"].items()
        options = list(question["criteria"])
        answer = {"type": "choice", "probabilities": {key: 1 / len(options) for key in options}}
        return httpx2.Response(200, json={"answers": {question_id: answer}, "usage": {"input_tokens": 12}})

    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        await anyio.lowlevel.checkpoint()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = clm("http://clm.test", api_key_env=None, http_client=client)
        waited = await warm_up_clm(model, base_url="http://clm.test", poll=10.0, client=client, sleep=sleep)

    assert seen == [("GET", "/health")] * 3 + [("POST", "/v1/systemone")]
    assert slept == [10.0, 10.0]
    assert waited >= 0.0


async def test_the_estimate_counts_every_ask_and_asks_no_model(
    toolret_data: ToolRetData, axis_embedder: Any, fake_decision_model: Any
) -> None:
    data = toolret_data
    models = {
        "jev": fake_decision_model(limits=JEV_LIMITS),
        "clm": fake_decision_model(limits=CLM_LIMITS),
        "logprob": fake_decision_model(limits=LOGPROB_LIMITS, prompt_version="letters-v1"),
    }

    async def estimate(k: int) -> list[EstimateLine]:
        return await estimate_decisions(
            build_decision_arms(["jev", "clm", "logprob"], ks=[k], models=models, max_detail={}),
            data,
            data.tasks,
            retriever=SharedRetrieval(DenseRetriever(axis_embedder())),  # it ranks every tool, not just matching ones
            sources=("plain",),
            model_queries=None,
            negatives=True,
            repeat=3,
        )

    jev, clm_line, logprob = await estimate(20)

    assert all(model.asks == [] for model in models.values())
    # Every one of the 20 tools is a candidate of a positive search: 21 options with the reserved one, over the
    # logprob model's cap of 20, so it asks in two rounds. A negative search has at most 19 candidates: one question.
    searches = 4 * 2 * 3  # tasks x variants x repeats
    assert (jev.provider, jev.model, jev.calls) == ("typesafe", "jev-1.13.0", searches)
    assert (clm_line.provider, clm_line.model, clm_line.calls, clm_line.usd) == ("modal", "clm-latest", searches, None)
    assert (logprob.provider, logprob.model, logprob.calls) == ("openai", LOGPROB_MODEL, 4 * (2 + 1) * 3)
    # Jev and CLM are asked the same questions here, and CLM's line counts their tokens without an overhead.
    assert jev.input_tokens == clm_line.input_tokens + JEV_REQUEST_OVERHEAD_TOKENS * jev.calls
    assert jev.usd == pytest.approx(jev.input_tokens * 0.042 / 1_000_000)
    usage = Usage(input_tokens=logprob.input_tokens, output_tokens=logprob.calls)  # one output token per call
    price = calc_price(usage, model_ref=LOGPROB_MODEL, provider_id="openai")
    assert logprob.usd == pytest.approx(float(price.total_price))  # pyright: ignore[reportUnknownMemberType]
    assert "7 GPU seconds" in clm_line.note  # 24 calls at 0.3 s
    # At K = 5 every decider asks one question per search, the same one: logprob adds 120 tokens per ask.
    _, clm_line, logprob = await estimate(5)
    assert logprob.calls == clm_line.calls == searches
    assert logprob.input_tokens == clm_line.input_tokens + 120 * logprob.calls


async def test_the_estimate_retrieval_embeds_nothing(
    tmp_path: Path, toolret_data: ToolRetData, axis_embedder: Any
) -> None:
    catalog = toolret_data.catalog
    paid = axis_embedder()
    cache = CachedEmbedder(paid, path=tmp_path / "embeddings.sqlite")
    await cache.embed([default_search_text(card) for card in catalog], kind="document")
    await cache.embed(["weather forecast"], kind="query")
    embedded = len(paid.calls)
    hybrid = HybridRetriever([BM25Retriever(), DenseRetriever(cache)])
    free = CacheOnlyRetrieval(hybrid, stand_in=BM25Retriever(), embeddings=cache)

    cached = await free.retrieve(["weather forecast"], catalog, k=5)
    uncached = await free.retrieve(["email inbox"], catalog, k=5)
    extended = ToolCatalog([*catalog, ToolCard(name="rain_radar", description="Weather radar.")])
    new_card = await free.retrieve(["weather forecast"], extended, k=5)

    assert len(paid.calls) == embedded  # nothing reached the paid embedder
    assert ids(cached) == ids(await hybrid.retrieve(["weather forecast"], catalog, k=5))
    # A query, or a card, the cache has no vector for: BM25 alone finds the candidates.
    assert ids(uncached) == ids(await BM25Retriever().retrieve(["email inbox"], catalog, k=5))
    assert ids(new_card) == ids(await BM25Retriever().retrieve(["weather forecast"], extended, k=5))
    assert free.stand_in_queries == {("email inbox",), ("weather forecast",)}
    assert len(paid.calls) == embedded
    cache.close()
