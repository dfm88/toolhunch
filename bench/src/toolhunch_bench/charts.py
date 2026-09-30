"""Offline figures from a generated decision summary, using held-out plain requests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import matplotlib
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["build_decision_charts"]

_COLORS = {"jev": "#2166ac", "logprob": "#b35806"}
_LABELS = {"jev": "Jev", "logprob": "Logprob"}


class _DecisionCharts:
    def __init__(self, summary: dict[str, Any], *, out_dir: Path) -> None:
        heldout = summary.get("heldout")
        if heldout is None:
            raise ValueError("Decision charts require a held-out summary")
        self.rows = [row for row in heldout["rows"] if row["source"] == "plain"]
        self.out_dir = out_dir

    def write(self) -> tuple[Path, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Matplotlib's keyword-heavy plotting signatures contain untyped extensions.
        with cast("Any", matplotlib).rc_context({"svg.fonttype": "none", "svg.hashsalt": "toolhunch-decision-v1"}):
            return self._precision(), self._coverage()

    def _save(self, figure: Figure, *, name: str) -> Path:
        path = self.out_dir / name
        cast("Any", figure).savefig(path, format="svg", metadata={"Date": None}, bbox_inches="tight")
        figure.clear()
        return path

    def _precision(self) -> Path:
        figure: Any = Figure(figsize=(8, 5.5), layout="constrained")
        axes: Any = figure.subplots()
        baseline = next((row for row in self.rows if row["decider"] is None and row["k"] == 20), None)
        if baseline is not None and baseline["p_at_1"] is not None:
            axes.axhline(baseline["p_at_1"], color="#666666", linestyle="--", label="Hybrid alone")
        for row in self.rows:
            decider = row["decider"]
            if decider not in _LABELS or row["k"] not in (20, 50):
                continue
            cost_data: dict[str, Any] = row["cost"] or {}
            cost = cost_data.get("usd_per_1000_searches")
            value, interval = row["p_at_1"], row["p_at_1_ci95"]
            if cost is None or cost <= 0 or value is None or interval is None:
                continue
            axes.errorbar(
                x=cost,
                y=value,
                yerr=[[max(0.0, value - interval[0])], [max(0.0, interval[1] - value)]],
                fmt="o" if row["k"] == 20 else "s",
                color=_COLORS[decider],
                capsize=4,
                label=f"{_LABELS[decider]} @ {row['k']}",
            )
        axes.set_xscale("log")
        axes.set_xlabel("Decision cost per 1,000 searches (USD, log scale)")
        axes.set_ylabel("P@1: a relevant tool first (95% interval)")
        axes.set_title("ToolRet held-out ranking · plain requests")
        axes.set_ylim(0, 1)
        axes.grid(visible=True, alpha=0.2)
        axes.legend(loc="upper left")
        figure.text(0.5, -0.02, "Decision cost excludes shared retrieval. ", ha="center", fontsize=8)
        return self._save(figure, name="decision-precision-cost.svg")

    def _coverage(self) -> Path:
        figure: Any = Figure(figsize=(8, 6.5))
        axes: Any = figure.subplots()
        figure.subplots_adjust(bottom=0.35)
        count_rows: list[list[str]] = []
        for row in self.rows:
            decider = row["decider"]
            if decider not in _LABELS or row["k"] != 20:
                continue
            curve = row.get("risk_coverage")
            if curve is None:
                raise ValueError("Regenerate the decision summary to add plain-source risk curves")
            points = [point for point in curve["grid"] if point["selective_accuracy"] is not None]
            axes.plot(
                [point["coverage"] for point in points],
                [point["selective_accuracy"] for point in points],
                color=_COLORS[decider],
                label=_LABELS[decider],
            )
            for tau in curve["dev_taus"]:
                point = next(point for point in curve["grid"] if point["tau"] == tau)
                if point["selective_accuracy"] is not None:
                    axes.scatter(point["coverage"], point["selective_accuracy"], color=_COLORS[decider], marker="s")
                    axes.annotate(
                        f"dev τ={tau:.2f}",
                        (point["coverage"], point["selective_accuracy"]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=9,
                    )
                count_rows.append(
                    [
                        _LABELS[decider],
                        f"{tau:.2f}",
                        str(point["correct"]),
                        str(point["wrong"]),
                        str(point["abstained"]),
                        f"{row['negatives'] / row['records']:.0%}",
                    ]
                )
        axes.set_xlabel("Coverage: answered / requests")
        axes.set_ylabel("Selective accuracy: correct / answered")
        axes.set_title("Held-out curve, for reading; τ was chosen on dev")
        axes.set_xlim(0, 1)
        axes.set_ylim(0, 1)
        axes.grid(visible=True, alpha=0.2)
        axes.legend(loc="upper right")
        if count_rows:
            table = axes.table(
                cellText=count_rows,
                colLabels=["K=20, plain", "dev τ", "Correct", "Wrong", "Abstained", "Negatives"],
                cellLoc="center",
                bbox=(0, -0.55, 1, 0.28),
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
        figure.text(
            0.5, 0.035, "Counts at the marked thresholds; positives and negatives pooled.", ha="center", fontsize=8
        )
        figure.text(0.5, 0.01, ha="center", fontsize=8)
        return self._save(figure, name="decision-coverage-accuracy.svg")


def build_decision_charts(summary_path: Path, *, out_dir: Path) -> tuple[Path, Path]:
    """Write ranking-cost and coverage-accuracy SVGs for Jev and logprob from a summary.

    The ranking figure uses K=20 and K=50; coverage uses K=20 and the plain-source
    grid, marking only the thresholds already chosen on dev. No provider is called.
    """
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    return _DecisionCharts(summary, out_dir=out_dir).write()
