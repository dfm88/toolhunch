"""`compare-runs`: one model's searches on two deployments, matched and compared without dropping anything."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from toolhunch_bench import cli


def search(task: str, *, top: str, p: float, none: float = 0.1, error: str | None = None) -> dict[str, Any]:
    other = "b" if top == "a" else "a"
    return {
        "record": "search",
        "decider": "clm",
        "task": task,
        "variant": "positive",
        "source": "plain",
        "k": 20,
        "repeat": 0,
        "ranked": [] if error else [top, other],
        "probabilities": {} if error else {top: p, other: 1 - p - none},
        "none_probability": none,
        "error": error,
    }


def write(run_dir: Path, records: list[dict[str, Any]], *, decider: str) -> Path:
    run_dir.mkdir(parents=True)
    lines = [json.dumps(record | {"decider": decider}) for record in records]
    (run_dir / "run.jsonl").write_text("\n".join([*lines, json.dumps({"record": "arm", "arm": "x"})]) + "\n")
    return run_dir


def test_compare_runs_counts_what_it_cannot_match(tmp_path: Path) -> None:
    reference = write(
        tmp_path / "modal",
        [
            search("t1", top="a", p=0.80),
            search("t2", top="a", p=0.70),
            search("t3", top="a", p=0.60),
            search("t4", top="a", p=0.90),
        ],
        decider="clm",
    )
    candidate = write(
        tmp_path / "mac",
        [
            search("t1", top="a", p=0.78),  # same top card, |dp| 0.02
            search("t2", top="b", p=0.75),  # another top card: a 0.15 vs 0.70, b 0.75 vs 0.20
            search("t3", top="a", p=0.60, error="DecisionError: gone"),  # errored on one side
            search("t5", top="a", p=0.50),  # only in the candidate; t4 only in the reference
        ],
        decider="clm-local",
    )

    result = CliRunner().invoke(
        cli.app,
        [
            "compare-runs",
            str(reference),
            str(candidate),
            "--reference-decider",
            "clm",
            "--candidate-decider",
            "clm-local",
            "--min-top1",
            "0.98",
            "--max-median-dp",
            "0.01",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert (summary["matched"], summary["errored"], summary["compared"]) == (3, 1, 2)
    assert (summary["unmatched_reference"], summary["unmatched_candidate"]) == (1, 1)
    assert summary["top1_agreement"] == 0.5
    assert summary["max_abs_dp"] == pytest.approx(0.55)
    assert summary["max_abs_dp_search"]["task"] == "t2"
    assert summary["median_abs_dp"] == pytest.approx((0.02 + 0.55) / 2)
    assert summary["gate"]["passed"] is False
