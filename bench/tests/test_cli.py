import json
from collections.abc import Callable
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
from toolhunch_bench import cli
from toolhunch_bench.datasets.model_queries import WRITER_MODEL, WRITER_SETTINGS, model_queries_path
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask, write_task_file
from toolhunch_bench.ledger import LedgerEntry, append_ledger, read_ledger

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
