import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import tiktoken
from genai_prices import Usage, calc_price
from inline_snapshot import snapshot
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from typer.testing import CliRunner

from toolhunch import ToolCard, ToolCatalog
from toolhunch.decision import CLM_LIMITS, JEV_LIMITS, LOGPROB_LIMITS, DecisionUsage
from toolhunch_bench import cli
from toolhunch_bench.datasets.model_queries import (
    WRITER_MODEL,
    WRITER_SETTINGS,
    WrittenQueries,
    model_queries_path,
    save_model_queries,
)
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask, write_task_file
from toolhunch_bench.deciders import DeciderName, decision_model
from toolhunch_bench.ledger import BUDGET, LedgerEntry, append_ledger, f2a_spend, read_ledger

TOOLS = {"web_tool_0": ("get_weather", "Weather forecast."), "web_tool_1": ("send_email", "Send an email.")}
DATA = ToolRetData(
    catalog=ToolCatalog(ToolCard(id=i, name=name, description=text) for i, (name, text) in TOOLS.items()),
    raw_text={i: json.dumps({"name": name, "description": text}) for i, (name, text) in TOOLS.items()},
    tasks=tuple(
        ToolRetTask(f"{subtask}_query_{i}", subtask, query, f"Given a `{subtask}` task", frozenset({tool}))
        for subtask, query, tool in [("weather", "rain in Rome", "web_tool_0"), ("email", "mail my boss", "web_tool_1")]
        for i in range(2)
    ),
    mapping_stats={"name": 1.0, "description": 1.0, "properties": 0.0},
)


def test_sample_run_report_and_check_from_the_command_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def load_toolret(*, cache_dir: Path) -> ToolRetData:
        return DATA

    monkeypatch.setattr(cli, "load_toolret", load_toolret)
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path / "runs")
    runner = CliRunner()
    task_file = tmp_path / "tasks.json"

    def invoke(*args: str) -> str:
        result = runner.invoke(cli.app, list(args))
        assert result.exit_code == 0, result.output
        return result.output

    invoke("toolret", "sample", "--n", "2", "--out", str(task_file))
    assert len(json.loads(task_file.read_text())["ids"]) == 2  # one per subtask
    assert "no paid arms" in invoke("retrieval", "--tasks", str(task_file), "--arms", "bm25", "--dry-run")
    invoke("retrieval", "--tasks", str(task_file), "--arms", "keywords,bm25")
    [run_dir] = (tmp_path / "runs").iterdir()
    invoke("report", str(run_dir), "--out", str(tmp_path / "report"))
    assert (tmp_path / "report" / "README.md").read_text().count("| bm25 |") == 2
    invoke("check", "bm25", "--tasks", str(task_file), "--out", str(tmp_path / "checks.json"))
    assert json.loads((tmp_path / "checks.json").read_text())["bm25"]["max_relative_difference"] < 1e-4


def test_sample_leaves_out_the_ids_of_an_exclude_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def load_toolret(*, cache_dir: Path) -> ToolRetData:
        return DATA

    monkeypatch.setattr(cli, "load_toolret", load_toolret)
    dev, heldout = tmp_path / "dev.json", tmp_path / "heldout.json"

    for arguments in (["--out", str(dev)], ["--exclude", str(dev), "--out", str(heldout)]):
        result = CliRunner().invoke(cli.app, ["toolret", "sample", "--n", "2", *arguments])
        assert result.exit_code == 0, result.output

    dev_ids = set(json.loads(dev.read_text())["ids"])
    heldout_ids = set(json.loads(heldout.read_text())["ids"])
    assert len(heldout_ids) == 2  # the two tasks the dev sample left, one per subtask
    assert not dev_ids & heldout_ids


class WordEncoding:
    """A stand-in for a tiktoken encoding, so that the estimate never downloads a vocabulary."""

    name = "words"

    def encode(self, text: str, *, disallowed_special: Any = ()) -> list[int]:
        return list(range(len(text.split())))


def word_encoding(name: str) -> WordEncoding:
    return WordEncoding()


def no_dotenv(*args: object, **kwargs: object) -> bool:
    return False


@pytest.fixture
def writer_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The temporary ledger `queries write` uses; nothing here reaches the network, the real ledger or `.env`."""

    def load_toolret(*, cache_dir: Path) -> ToolRetData:
        return DATA

    ledger = tmp_path / "cost-ledger.jsonl"
    monkeypatch.setattr(cli, "load_toolret", load_toolret)
    monkeypatch.setattr(cli, "LEDGER_PATH", ledger)
    monkeypatch.setattr(cli, "load_dotenv", no_dotenv)
    monkeypatch.setattr(tiktoken, "get_encoding", word_encoding)
    return ledger


@pytest.fixture
def writer_task_file(tmp_path: Path) -> Path:
    path = tmp_path / "tasks.json"
    write_task_file(path, [DATA.tasks[0], DATA.tasks[2]], seed=0)  # "rain in Rome", then "mail my boss"
    return path


def f2a_entry(usd: float) -> LedgerEntry:
    return LedgerEntry(
        timestamp=datetime(2026, 9, 29, 9, 0, tzinfo=UTC),
        run_id="20260929T090000Z",
        provider="openai",
        model="gpt-4.1-mini",
        input_tokens=1_000,
        usd=usd,
        purpose="F2a: an earlier paid run",
    )


PHASE = BUDGET.prefix.rstrip(":")


def phase_entry(usd: float) -> LedgerEntry:
    """An earlier paid run of the current phase, whose budget the paid commands check."""
    return LedgerEntry(
        timestamp=datetime(2026, 9, 29, 9, 0, tzinfo=UTC),
        run_id="20260929T090000Z",
        provider="openai",
        model="gpt-4.1-mini",
        input_tokens=1_000,
        usd=usd,
        purpose=f"{BUDGET.prefix} an earlier paid run",
    )


def forbid_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(*args: object, **kwargs: object) -> object:
        raise AssertionError("no model may be built")

    monkeypatch.setattr(cli, "OpenAIChatModel", build)


def install_writer(
    monkeypatch: pytest.MonkeyPatch, respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]
) -> list[tuple[str, Any]]:
    """Put a `FunctionModel` where the command builds its OpenAI model; returns the arguments it was built with."""
    built: list[tuple[str, Any]] = []

    def build(name: str, *, settings: Any) -> FunctionModel:
        built.append((name, settings))
        return FunctionModel(respond)

    monkeypatch.setattr(cli, "OpenAIChatModel", build)
    return built


def searching_for_rain(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Searches when the request mentions rain; otherwise answers in text. Every request bills 300 + 40 tokens."""
    [request] = messages
    assert isinstance(request, ModelRequest)
    [prompt] = request.parts
    assert isinstance(prompt, UserPromptPart)
    usage = RequestUsage(input_tokens=300, output_tokens=40)
    if "rain" in str(prompt.content):
        return ModelResponse(
            parts=[ToolCallPart("search_tools", {"queries": ["weather forecast", "rain forecast"]})], usage=usage
        )
    return ModelResponse(parts=[TextPart("Sure.")], usage=usage)


def test_the_writer_dry_run_prints_the_estimate_and_spends_nothing(
    writer_ledger: Path, writer_task_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    append_ledger(f2a_entry(1.5), path=writer_ledger)
    before = writer_ledger.read_text()
    forbid_model(monkeypatch)

    result = CliRunner().invoke(cli.app, ["queries", "write", "--tasks", str(writer_task_file), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "estimate" in result.output.lower()
    assert "upper bound" in result.output  # the output tokens are assumed, not measured
    assert "$1.5000" in result.output  # the F2a total the cap is checked against
    assert writer_ledger.read_text() == before
    assert not model_queries_path(writer_task_file).exists()


def test_the_writer_stops_before_spending_when_the_estimate_passes_the_cap(
    writer_ledger: Path, writer_task_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    append_ledger(f2a_entry(6.999), path=writer_ledger)
    before = writer_ledger.read_text()
    forbid_model(monkeypatch)

    result = CliRunner().invoke(cli.app, ["queries", "write", "--tasks", str(writer_task_file)])

    assert result.exit_code == 2
    assert "$6.9990" in result.output  # the F2a total ...
    assert "$7.00 cap" in result.output  # ... and the cap it would pass
    assert writer_ledger.read_text() == before
    assert not model_queries_path(writer_task_file).exists()


def test_the_writer_saves_the_queries_and_records_what_it_billed(
    writer_ledger: Path, writer_task_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built = install_writer(monkeypatch, searching_for_rain)

    result = CliRunner().invoke(cli.app, ["queries", "write", "--tasks", str(writer_task_file)])

    assert result.exit_code == 0, result.output
    assert built == [(WRITER_MODEL, WRITER_SETTINGS)]
    saved = json.loads(model_queries_path(writer_task_file).read_text())
    assert saved["queries"] == snapshot(
        {
            "weather_query_0": {"queries": ["weather forecast", "rain forecast"], "fallback": False},
            "email_query_0": {"queries": ["mail my boss"], "fallback": True},
        }
    )  # in the task file's order
    assert saved["writer"]["model"] == WRITER_MODEL
    [entry] = read_ledger(writer_ledger)
    price = calc_price(Usage(input_tokens=600, output_tokens=80), model_ref=WRITER_MODEL, provider_id="openai")
    assert (entry.provider, entry.model, entry.input_tokens, entry.output_tokens) == ("openai", WRITER_MODEL, 600, 80)
    assert entry.usd == pytest.approx(float(price.total_price))  # pyright: ignore[reportUnknownMemberType]
    assert entry.purpose == "F2a: model-written queries for tasks.json"
    assert "Fallbacks (no search in the first response): 1 of 2" in result.output
    assert "email_query_0: Sure." in result.output  # what the model answered instead of searching


def test_a_failing_writer_run_still_records_what_it_billed(
    writer_ledger: Path, writer_task_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fails_on_the_second_task(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if "mail" in str(messages):
            raise RuntimeError("the API went away")
        return searching_for_rain(messages, info)

    install_writer(monkeypatch, fails_on_the_second_task)

    result = CliRunner().invoke(cli.app, ["queries", "write", "--tasks", str(writer_task_file)])

    assert isinstance(result.exception, RuntimeError)
    [entry] = read_ledger(writer_ledger)
    assert (entry.input_tokens, entry.output_tokens) == (300, 40)  # the task that got an answer
    assert not model_queries_path(writer_task_file).exists()  # a partial file would read as a complete one


@dataclass
class DecisionSetup:
    """What the `decision` command was given: its task file, its ledger and the models it would pay for."""

    task_file: Path
    ledger: Path
    runs: Path
    decision_cache: Path
    models: dict[str, Any]
    embedders: list[Any]
    warmed: list[tuple[Any, str]]


@pytest.fixture
def decision_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    toolret_data: ToolRetData,
    axis_embedder: Any,
    fake_decision_model: Any,
) -> DecisionSetup:
    """The `decision` command kept off the network, the real ledger, the real caches and `.env`.

    Fakes with each real model's limits stand where the command builds Jev, CLM, the logprob model and the embedder,
    and where it warms CLM up, so a test can check that none of them was asked anything.
    """

    def load_toolret(*, cache_dir: Path) -> ToolRetData:
        return toolret_data

    task_file = tmp_path / "tasks.json"
    write_task_file(task_file, toolret_data.tasks, seed=0)
    written = [
        WrittenQueries(
            task.id, ("weather forecast",) if task.id == "t_query_0" else (task.query,), task.id != "t_query_0"
        )
        for task in toolret_data.tasks
    ]
    save_model_queries(model_queries_path(task_file), written, writer={"model": "writer@test"})
    models = {
        "jev": fake_decision_model(model_id="jev-1.13.0@api.typesafe.ai", limits=JEV_LIMITS),
        "clm": fake_decision_model(model_id="clm-latest@clm.test", limits=CLM_LIMITS),
        "logprob": fake_decision_model(
            model_id="gpt-4.1-mini-2025-04-14@api.openai.com", limits=LOGPROB_LIMITS, prompt_version="letters-v1"
        ),
    }
    embedders: list[Any] = []

    def embedder(model_name: str, **settings: object) -> Any:
        embedders.append(axis_embedder(model_id=model_name))
        return embedders[-1]

    warmed: list[tuple[Any, str]] = []

    async def warm_up_clm(model: Any, *, base_url: str) -> float:
        warmed.append((model, base_url))
        return 42.0

    setup = DecisionSetup(
        task_file=task_file,
        ledger=tmp_path / "cost-ledger.jsonl",
        runs=tmp_path / "runs",
        decision_cache=tmp_path / "decisions.sqlite",
        models=models,
        embedders=embedders,
        warmed=warmed,
    )
    monkeypatch.setattr(cli, "load_toolret", load_toolret)
    monkeypatch.setattr(cli, "LEDGER_PATH", setup.ledger)
    monkeypatch.setattr(cli, "RUNS_DIR", setup.runs)
    monkeypatch.setattr(cli, "EMBEDDING_CACHE_PATH", tmp_path / "embeddings.sqlite")
    monkeypatch.setattr(cli, "DECISION_CACHE_PATH", setup.decision_cache)
    monkeypatch.setattr(cli, "load_dotenv", no_dotenv)
    monkeypatch.setattr(tiktoken, "get_encoding", word_encoding)

    def model(name: DeciderName, *, max_retries: int = 3) -> Any:
        # Deciders without a fake are built for real: building one contacts nothing.
        return models[name] if name in models else decision_model(name, max_retries=max_retries)

    monkeypatch.setattr(cli, "decision_model", model)
    monkeypatch.setattr(cli, "OpenAIEmbedder", embedder)
    monkeypatch.setattr(cli, "warm_up_clm", warm_up_clm)
    monkeypatch.setenv("CLM_BASE_URL", "http://clm.test")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")  # checked by name before a run; the fakes never send it
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return setup


def decide(setup: DecisionSetup, *options: str) -> Any:
    arguments = ["decision", "--tasks", str(setup.task_file), "--split", "dev", "--deciders", "jev,clm,logprob"]
    return CliRunner().invoke(cli.app, [*arguments, "--k", "3,5", "--sources", "plain,model", *options])


def spent_nothing(setup: DecisionSetup) -> bool:
    """No model was asked, no text embedded and CLM not woken; no run was written and no decision cache opened."""
    asked = any(model.asks for model in setup.models.values()) or any(e.calls for e in setup.embedders)
    return not asked and not setup.warmed and not setup.runs.exists() and not setup.decision_cache.exists()


def test_decision_dry_run_prints_an_estimate_and_spends_nothing(decision_setup: DecisionSetup) -> None:
    append_ledger(phase_entry(1.5), path=decision_setup.ledger)
    before = decision_setup.ledger.read_text()

    result = decide(decision_setup, "--dry-run")

    assert result.exit_code == 0, result.output
    assert "estimate" in result.output.lower()
    for line in ("typesafe jev-1.13.0: $", "openai gpt-4.1-mini-2025-04-14: $", "modal clm-latest: Modal credits"):
        assert line in result.output
    assert "GPU seconds" in result.output  # CLM is counted in time, not in dollars
    assert "BM25 alone" in result.output  # the cache holds no query embedding yet
    assert "$1.5000" in result.output  # the phase total the cap is checked against
    assert f"${BUDGET.cap_usd:.2f} cap" in result.output
    assert decision_setup.ledger.read_text() == before
    assert spent_nothing(decision_setup)


def test_decision_dry_run_estimates_new_deciders_from_the_registry(
    decision_setup: DecisionSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-test")
    monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test-token")
    arguments = [
        "decision",
        "--tasks",
        str(decision_setup.task_file),
        "--split",
        "dev",
        "--deciders",
        "clef-flash,strands",
    ]

    result = CliRunner().invoke(cli.app, [*arguments, "--k", "3", "--sources", "plain", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "  cloudflare clef-flash: $" in result.output  # priced at the declared $0.09/M
    assert "at the declared $0.09/M" in result.output
    assert "  local strands-decider-2B-hobson-v19: $0.0000; " in result.output  # a local model costs nothing
    assert "acct-test" not in result.output
    assert spent_nothing(decision_setup)


def test_decision_stops_before_spending_when_the_ledger_is_over_the_cap(decision_setup: DecisionSetup) -> None:
    append_ledger(phase_entry(8.5), path=decision_setup.ledger)
    before = decision_setup.ledger.read_text()

    result = decide(decision_setup)  # a paid run, not a dry run

    assert result.exit_code == 2
    assert f"Over the cap: nothing was spent. {PHASE} total $8.5000, estimate $" in result.output
    assert decision_setup.ledger.read_text() == before
    assert spent_nothing(decision_setup)


def test_a_decision_run_warms_clm_up_and_records_what_it_billed(
    decision_setup: DecisionSetup, fake_decision_model: Any
) -> None:
    append_ledger(phase_entry(1.5), path=decision_setup.ledger)
    # Neither Jev nor the logprob model answers anything usable for "mail my boss": 2 arms x 2 sources x 2 variants
    # of that task end in an error. The logprob model takes 4 options, so at K = 5 it asks two questions in round
    # one, and both fail: its 8 failed searches are 12 failed asks.
    jev_model = fake_decision_model(model_id="jev-1.13.0@api.typesafe.ai", limits=JEV_LIMITS, fail_on="boss")
    logprob_model = fake_decision_model(
        model_id="gpt-4.1-mini-2025-04-14@api.openai.com",
        limits=LOGPROB_LIMITS.model_copy(update={"max_options_per_choice": 4}),
        prompt_version="letters-v1",
        fail_on="boss",
    )
    decision_setup.models |= {"jev": jev_model, "logprob": logprob_model}

    result = decide(decision_setup)

    assert result.exit_code == 0, result.output
    models = decision_setup.models
    # CLM was woken before each of its two arms, since it may have scaled to zero during the arms in between, and
    # through the model itself: an answer replayed from the cache would wake nothing.
    assert decision_setup.warmed == [(models["clm"], "http://clm.test")] * 2
    assert "hybrid+clm@5: CLM answered after 42 s of warm-up" in result.output
    [run_dir] = decision_setup.runs.iterdir()
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines()]
    earlier, *entries = read_ledger(decision_setup.ledger)
    assert earlier == phase_entry(1.5)
    assert {entry.purpose for entry in entries} == {
        f"{BUDGET.prefix} decision dev (jev, clm, logprob; k 3/5; plain/model) on tasks.json"
    }
    by_model = {(entry.provider, entry.model): entry for entry in entries}
    assert sorted(by_model) == [
        ("modal", "clm-latest"),
        ("openai", "gpt-4.1-mini-2025-04-14"),
        ("openai", "text-embedding-3-small"),
        ("typesafe", "jev-1.13.0"),
    ]
    jev = by_model["typesafe", "jev-1.13.0"]
    # Jev bills per declared token, so its asks go through the spend guard, which retries a failed attempt once:
    # its 8 failed searches are 16 failed attempts.
    assert jev.input_tokens == 10 * (len(jev_model.asks) - 16)  # 10 input tokens for every ask it answered
    # The guard charged each failed attempt its reservation, which the ledger keeps, as direct and order runs do: the
    # provider may have billed those attempts, and the next run's cap check reads the ledger.
    reservation = JEV_LIMITS.estimate_usd(DecisionUsage(1, JEV_LIMITS.max_request_tokens or 0, 0)) or 0.0
    assert jev.usd == pytest.approx(jev.input_tokens * 0.042 / 1_000_000 + 16 * reservation)
    assert ", 8 failed," in jev.note
    logprob = by_model["openai", "gpt-4.1-mini-2025-04-14"]
    assert logprob.input_tokens == 10 * (len(logprob_model.asks) - 12)
    assert ", 12 failed," in logprob.note
    usage = Usage(input_tokens=logprob.input_tokens, output_tokens=logprob.output_tokens)
    price = calc_price(usage, model_ref="gpt-4.1-mini-2025-04-14", provider_id="openai")
    assert logprob.usd == pytest.approx(float(price.total_price))  # pyright: ignore[reportUnknownMemberType]
    modal = by_model["modal", "clm-latest"]
    clm_seconds = sum(r["wall_seconds"] for r in records if r["record"] == "arm" and "+clm@" in r["arm"])
    assert modal.usd == pytest.approx(clm_seconds * 0.80 / 3600)
    assert modal.note.startswith("Modal credits: ")
    assert f2a_spend(read_ledger(decision_setup.ledger), purpose_prefix=BUDGET.prefix).modal_usd == pytest.approx(
        modal.usd
    )  # apart from the cap
    assert by_model["openai", "text-embedding-3-small"].input_tokens > 0  # the texts the fresh cache lacked
    assert all(model.closed for model in models.values())
    assert all(embedder.closed for embedder in decision_setup.embedders)
    assert f"{PHASE} total now: $" in result.output


def test_a_failing_decision_run_still_records_what_it_billed(
    decision_setup: DecisionSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def gone(request: Any, /, **options: Any) -> Any:
        raise RuntimeError("the API went away")

    monkeypatch.setattr(decision_setup.models["logprob"], "ask", gone)

    result = decide(decision_setup)

    assert isinstance(result.exception, RuntimeError)
    # The arms before the first logprob one had run: Jev, CLM and the query embeddings were paid for. The logprob
    # model answered nothing, but its failed ask may have been billed, so it has a line too.
    billed = {(entry.provider, entry.model): entry for entry in read_ledger(decision_setup.ledger)}
    assert sorted(billed) == [
        ("modal", "clm-latest"),
        ("openai", "gpt-4.1-mini-2025-04-14"),
        ("openai", "text-embedding-3-small"),
        ("typesafe", "jev-1.13.0"),
    ]
    logprob = billed["openai", "gpt-4.1-mini-2025-04-14"]
    assert (logprob.input_tokens, logprob.usd) == (0, 0.0)
    assert logprob.note.startswith("asks: 0 answered by the model, 0 by the cache, 1 failed,")
    assert all(model.closed for model in decision_setup.models.values())


@pytest.mark.parametrize(
    ("status", "attempts", "reason"),
    [(401, 1, "typesafe refused with HTTP 401"), (429, 3, "3 failed attempts in a row at typesafe")],
)
def test_a_decision_run_stops_when_its_provider_refuses_or_is_out(
    decision_setup: DecisionSetup, fake_decision_model: Any, status: int, attempts: int, reason: str
) -> None:
    # A refused key stops the run at once; a quota or an outage after three failed attempts in a row. Going on would
    # only charge more reservations for asks that cannot be answered (P2's Clef dev run, refused at its daily quota).
    append_ledger(phase_entry(1.5), path=decision_setup.ledger)
    jev_model = fake_decision_model(
        model_id="jev-1.13.0@api.typesafe.ai", limits=JEV_LIMITS, fail_on="", fail_status=status
    )
    decision_setup.models |= {"jev": jev_model}

    result = decide(decision_setup)

    assert result.exit_code == 2, result.output
    assert f"Run stopped: {reason}" in result.output
    assert len(jev_model.asks) == attempts
    [run_dir] = decision_setup.runs.iterdir()
    assert reason in json.loads((run_dir / "manifest.json").read_text())["stop_reason"]
    billed = {(entry.provider, entry.model): entry for entry in read_ledger(decision_setup.ledger)}
    reservation = JEV_LIMITS.estimate_usd(DecisionUsage(1, JEV_LIMITS.max_request_tokens or 0, 0)) or 0.0
    assert billed["typesafe", "jev-1.13.0"].usd == pytest.approx(attempts * reservation)
    assert all(model.closed for model in decision_setup.models.values())


@pytest.mark.parametrize("name", ["logprob", "luna", "clm"])
def test_direct_refuses_deciders_it_cannot_guard_or_ledger(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    # The direct path guards and ledgers deciders billed per declared token and local ones only: OpenAI's would spend
    # without a cap or a ledger line, and Modal's CLM would spend credits unledgered. It refuses before loading data.
    def never(**_: Any) -> None:
        raise AssertionError("ToolRet loaded before the deciders were checked")

    monkeypatch.setattr(cli, "load_toolret", never)

    result = CliRunner().invoke(cli.app, ["direct", "--deciders", name, "--dry-run"])

    assert result.exit_code == 2, result.output
    assert f"unknown {name}" in result.output
