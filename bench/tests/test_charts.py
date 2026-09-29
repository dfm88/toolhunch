import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from toolhunch_bench.cli import app


def test_decision_charts_use_only_plain_jev_and_logprob_rows(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for decider in (None, "jev", "logprob", "clm"):
        for k in (20, 50):
            for source in ("plain", "model"):
                rows.append(
                    {
                        "decider": decider,
                        "k": k,
                        "source": source,
                        "p_at_1": 0.3,
                        "p_at_1_ci95": [0.2, 0.4],
                        "records": 4,
                        "negatives": 2,
                        "cost": {"usd_per_1000_searches": 0.05 if decider == "jev" else 0.5},
                        "risk_coverage": {
                            "dev_taus": [0.5],
                            "grid": [
                                {
                                    "tau": 0.0,
                                    "coverage": 1.0,
                                    "selective_accuracy": 0.25,
                                    "correct": 1,
                                    "wrong": 3,
                                    "abstained": 0,
                                },
                                {
                                    "tau": 0.5,
                                    "coverage": 0.5,
                                    "selective_accuracy": 0.5,
                                    "correct": 1,
                                    "wrong": 1,
                                    "abstained": 2,
                                },
                            ],
                        },
                    }
                )
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"heldout": {"rows": rows}}))
    result = CliRunner().invoke(app, ["decision-charts", str(summary), "--out", str(tmp_path / "figures")])
    assert result.exit_code == 0, result.output
    for name in ("decision-precision-cost.svg", "decision-coverage-accuracy.svg"):
        text = (tmp_path / "figures" / name).read_text()
        assert "<svg" in text
        assert "Jev" in text
        assert "Logprob" in text
        assert "CLM" not in text
        assert "clm" not in text
        assert "ratel" not in text
    text = (tmp_path / "figures" / "decision-coverage-accuracy.svg").read_text()
    assert all(label in text for label in ("Correct", "Wrong", "Abstained", "Negatives", "50%", "dev τ=0.50"))
