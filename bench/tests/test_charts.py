import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from toolhunch_bench.cli import app
from toolhunch_bench.deciders import DECIDERS, DeciderName, DeciderSpec


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


def test_cost_latency_marks_local_deciders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # P2's figure puts the new deciders next to the earlier ones. A local decider is drawn hollow with the machine in
    # the caption; latency is the decision's alone, since later runs searched with cached query embeddings; CLM stays
    # out until its authors confirm parity. Colors and that exclusion are registry data, so the figure draws a new
    # decider without an edit of its own.
    registry = cast("dict[DeciderName, DeciderSpec]", DECIDERS)
    monkeypatch.setitem(registry, DeciderName.STRANDS, replace(registry[DeciderName.STRANDS], color="#123456"))
    hardware = {"chip": "Apple M5 Max", "memory_gb": 128, "os": "macOS 26.1"}

    def pick(arm: str, rate: float, ms: float | None, usd: float | None, *, local: bool = False) -> dict[str, Any]:
        warm = {"decision_latency_p50_ms": ms, "billed_usd_per_1000": usd} | ({"local": True} if local else {})
        interval = [rate - 0.05, rate + 0.05]
        return {
            "arm": arm,
            "catalog": "pooled",
            "relevant_pick_rate": {"value": rate, "ci95": interval},
            "positive_cost": {"warm": warm},
        }

    def ranked(arm: str, decider: str | None, p: float, ms: float | None, usd: float | None) -> dict[str, Any]:
        cost = {"usd_per_1000_searches": usd} | ({"local": hardware} if usd is None and decider else {})
        return {
            "arm": arm,
            "decider": decider,
            "k": 20,
            "source": "plain",
            "p_at_1": p,
            "p_at_1_ci95": [p - 0.05, p + 0.05],
            "cost": cost,
            "latency_ms": {"decision": None if ms is None else {"p50": ms}},
        }

    provenance = {"provenance": {"hardware": hardware}}
    added = {"started": "2026-10-04T19:00:00+00:00", "deciders": {"strands": provenance, "laya": provenance}}
    direct = {
        "manifest": {"started": "2026-09-30T05:00:00+00:00"},
        "added_runs": [{"manifest": added}],
        "rows": [
            pick("hybrid@20", 0.55, None, 0.0),
            pick("jev-all", 0.74, 320, 0.28),
            pick("hybrid@20+jev", 0.71, 280, 0.08),
            pick("strands-all", 0.62, 116, None, local=True) | {"lower_detail": 375},
            pick("hybrid@20+strands", 0.61, 157, None, local=True),
            pick("clef-all", 0.80, 1430, 1.63),
            pick("hybrid@20+clef", 0.77, 612, 0.44),
            pick("hybrid@20+clm-local", 0.15, 55, None, local=True),
            pick("laya-all", 0.36, 4200, None, local=True),  # no label offset of its own, slower than 3 s
        ],
    }
    f2a = {
        "runs": [{"manifest": {"started": "2026-09-29T15:00:00+00:00"}}],
        "heldout": {
            "rows": [ranked("hybrid@20", None, 0.22, None, 0.0), ranked("hybrid+jev@20", "jev", 0.33, 300, 0.05)]
        },
    }
    p2 = {
        "runs": [{"manifest": {"started": "2026-10-04T19:00:00+00:00"}}],
        "heldout": {
            "rows": [
                ranked("hybrid@20", None, 0.22, None, 0.0),
                ranked("hybrid+strands@20", "strands", 0.285, 80, None),
                ranked("hybrid+clef@20", "clef", 0.31, 517, 0.27),
                ranked("hybrid+clm-local@20", "clm-local", 0.13, 339, None),
            ]
        },
    }
    paths: dict[str, Path] = {}
    for name, summary in (("direct", direct), ("f2a", f2a), ("p2", p2)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(summary))
    arguments = ["cost-latency-chart", "--direct", str(paths["direct"]), "--out", str(tmp_path / "figures")]
    arguments += ["--decision", str(paths["f2a"]), "--decision", str(paths["p2"])]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    text = (tmp_path / "figures" / "cost-latency.svg").read_text()
    for point in ("jev-all", "hybrid-20-jev", "clef-all", "hybrid-20-clef", "hybrid-jev-20", "hybrid-clef-20"):
        assert f'id="point-{point}"' in text
    for point in ("strands-all", "hybrid-20-strands", "hybrid-strands-20", "laya-all"):
        assert f'id="point-{point}-local"' in text
    assert "clm" not in text
    assert "CLM" not in text
    assert "#123456" in text  # the registry's color for Strands
    # Text is drawn as paths; the caption is also the SVG's description, which states the machine and the points
    # whose cards were sent below full detail to fit a window.
    assert "Apple M5 Max, 128 GB, macOS 26.1" in text
    assert "Lower detail: Strands Decider 2B (all tools)" in text


def test_readme_charts_draw_every_published_decider(tmp_path: Path) -> None:
    # The README figures read every published summary, so a decider added to the registry and the results appears
    # in each without an edit of its own; CLM, unpublished, appears in none.
    result = CliRunner().invoke(app, ["readme-charts", "--out", str(tmp_path)])

    assert result.exit_code == 0, result.output
    direct = (tmp_path / "direct-choice.svg").read_text()
    for arm in (
        "agent-luna-all",
        "hybrid-20-jev",
        "clef-all",
        "strands-all",
        "rizzo-flow-all",
        "hybrid-20-laya-wide",
        "luna-decisions-all",
    ):
        assert f'id="bar-{arm}"' in direct
    assert 'id="bar-hybrid-20"' in direct
    rerank = (tmp_path / "rerank-44k.svg").read_text()
    for arm in (
        "hybrid-jev-20",
        "hybrid-luna-20",
        "hybrid-clef-flash-20",
        "hybrid-rizzo-flow-20",
        "hybrid-laya-wide-20",
        "hybrid-luna-decisions-20",
    ):
        assert f'id="bar-{arm}"' in rerank
    order = (tmp_path / "order-sensitivity.svg").read_text()
    assert 'id="order-jev-same"' in order
    assert 'id="order-rizzo-flow-shuffled"' in order
    assert 'id="order-gpt-6-luna-decisions-shuffled"' in order
    assert 'id="order-clef-same"' not in order  # no same-order repeats for a decider that never varied
    for text in (direct, rerank, order):
        assert "clm" not in text
