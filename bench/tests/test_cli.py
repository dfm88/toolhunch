import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from toolhunch import ToolCard, ToolCatalog
from toolhunch_bench import cli
from toolhunch_bench.datasets.toolret import ToolRetData, ToolRetTask

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
