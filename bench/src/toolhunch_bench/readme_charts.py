"""README figures from the generated direct-choice, decision and order summaries, without provider calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["build_readme_charts"]

_SEARCH, _JEV, _GPT = "#8C939B", "#2F6DB5", "#C0652B"
_GOOD, _BAD, _NEUTRAL = "#2E8B57", "#C8483B", "#C9CED4"
_INK, _MUTED, _GRID = "#1F2328", "#5B636B", "#E6E8EB"
_STYLE = {
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 15,
    "text.color": _INK,
    "axes.edgecolor": _MUTED,
    "xtick.color": _MUTED,
    "ytick.color": _INK,
    "svg.fonttype": "path",
    "svg.hashsalt": "toolhunch-readme-v1",
}


@dataclass(frozen=True, slots=True)
class _Row:
    key: str
    name: str
    detail: str
    color: str


# Top to bottom in the direct-choice figures.
_DIRECT = (
    _Row("jev-all", "Jev", "reads the whole catalog", _JEV),
    _Row("hybrid@20+jev", "Jev", "reads 20 searched tools", _JEV),
    _Row("agent-all", "GPT-4.1 mini agent", "gets the whole catalog as functions", _GPT),
    _Row("agent@20", "GPT-4.1 mini agent", "gets 20 searched tools as functions", _GPT),
    _Row("hybrid@20", "Search only", "BM25 + embeddings, top result", _SEARCH),
)
_RERANK = (
    _Row("hybrid@20", "Search only", "BM25 + embeddings, top result", _SEARCH),
    _Row("hybrid+jev@20", "Jev picks", "one choice question", _JEV),
    _Row("hybrid+logprob@20", "GPT-4.1 mini picks", "logprobs over option letters", _GPT),
)


class _ReadmeCharts:
    def __init__(
        self, *, direct: dict[str, Any], decision: dict[str, Any], order: dict[str, Any], out_dir: Path, fmt: str
    ) -> None:
        self.pooled = {row["arm"]: row for row in direct["rows"] if row["catalog"] == "pooled"}
        self.rerank = {row["arm"]: row for row in decision["heldout"]["rows"] if row["source"] == "plain"}
        self.order = order
        self.out_dir = out_dir
        self.fmt = fmt
        manifest = direct["manifest"]
        self.direct_footer = self._footer(
            manifest["started"], manifest["jev"]["model"].split("@")[0], manifest["agent"]["model"]
        )
        arms = order["manifest"]["arms"]
        models = (arms["hybrid+jev@20"]["model_id"].split("@")[0], arms["hybrid+logprob@20"]["model_id"].split("@")[0])
        heldout = next(run for run in decision["runs"] if run["run_id"] == decision["heldout"]["run_id"])
        self.rerank_footer = self._footer(heldout["manifest"]["started"], *models)
        self.order_footer = self._footer(order["manifest"]["started"], *models)

    @staticmethod
    def _footer(started: str, *models: str) -> str:
        return f"toolhunch · ToolRet · run of {started[:10]} · {', '.join(models)}"

    def write(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with cast("Any", matplotlib).rc_context(_STYLE):
            return [self._direct(), self._none_option(), self._rerank(), self._order()]

    def _frame(self, title: str, subtitle: str, footer: str) -> Any:
        figure: Any = Figure(figsize=(16, 9), dpi=100)
        figure.text(0.04, 0.94, title, fontsize=28, weight="bold", va="top")
        figure.text(0.04, 0.879, subtitle, fontsize=16, color=_MUTED, va="top")
        figure.text(0.04, 0.025, footer, fontsize=11, color=_MUTED)
        return figure

    def _save(self, figure: Any, *, name: str) -> Path:
        path = self.out_dir / f"{name}.{self.fmt}"
        metadata = {"Date": None} if self.fmt == "svg" else {"Software": None}
        figure.savefig(path, format=self.fmt, metadata=metadata)
        figure.clear()
        return path

    @staticmethod
    def _label(figure: Any, row: _Row, *, y: float) -> None:
        figure.text(0.29, y + 0.012, row.name, ha="right", weight="bold", fontsize=17)
        figure.text(0.29, y - 0.03, row.detail, ha="right", color=_MUTED, fontsize=13)

    @staticmethod
    def _bar(axes: Any, *, y: float, value: float, interval: list[float], color: str) -> None:
        axes.barh(y, value, color=color, height=0.62)
        axes.errorbar(value, y, xerr=[[value - interval[0]], [interval[1] - value]], color=_INK, capsize=4, lw=1.2)
        axes.text(0.012, y, f"{value:.0%}", color="white", weight="bold", va="center", fontsize=20)

    @staticmethod
    def _percent_axis(axes: Any, *, top: float, rows: int) -> None:
        ticks = [tick / 100 for tick in range(0, round(top * 100) + 1, 20 if top < 1 else 25)]
        axes.set_xlim(0, top)
        axes.set_ylim(-0.6, rows - 0.4)
        axes.set_yticks([])
        axes.set_xticks(ticks, [f"{tick:.0%}" for tick in ticks])
        axes.grid(axis="x", color=_GRID)
        axes.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axes.spines[side].set_visible(False)
        axes.tick_params(axis="y", length=0)

    def _direct(self) -> Path:
        figure = self._frame(
            "Which tool should the agent call?",
            "290 requests, catalogs of 40–101 tools: share where a relevant tool was picked. “None” counts as a miss.",  # noqa: RUF001
            f"Whiskers: 95% interval. Cost and latency: warm requests, median latency. {self.direct_footer}",
        )
        axes: Any = figure.add_axes((0.30, 0.12, 0.47, 0.66))
        for index, row in enumerate(_DIRECT):
            y = len(_DIRECT) - 1 - index
            rate = self.pooled[row.key]["relevant_pick_rate"]
            self._bar(axes, y=y, value=rate["value"], interval=rate["ci95"], color=row.color)
            centre = 0.12 + 0.66 * (y + 0.5) / len(_DIRECT)
            self._label(figure, row, y=centre)
            warm = self.pooled[row.key]["positive_cost"]["warm"]
            usd = warm["billed_usd_per_1000"]
            cost = "≈ $0" if usd < 0.01 else f"${usd:.2f}"
            cached = f"  ({warm['cache_share']:.0%} cached)" if warm["cache_share"] else ""
            latency = warm["latency_p50_ms"] / 1000
            figure.text(0.785, centre - 0.008, f"{cost}{cached}\n{latency:.2f} s", fontsize=14, va="center")
        figure.text(0.785, 0.80, "Per 1,000 requests · latency", fontsize=14, weight="bold", color=_MUTED)
        self._percent_axis(axes, top=1.0, rows=len(_DIRECT))
        return self._save(figure, name="direct-choice")

    def _none_option(self) -> Path:
        figure = self._frame(
            "When the right tool is missing, who says “none”?",
            "Same five strategies. Left: 290 requests with a relevant tool in the catalog. Right: 145 with it removed.",
            f"Removed tools leave approximate negatives: an unlabelled tool might still help. {self.direct_footer}",
        )
        left: Any = figure.add_axes((0.30, 0.14, 0.31, 0.62))
        right: Any = figure.add_axes((0.65, 0.14, 0.31, 0.62))
        for index, row in enumerate(_DIRECT):
            y = len(_DIRECT) - 1 - index
            none = self.pooled[row.key]["none_option"]
            negatives = none["negative_requests"]
            positives = none["requests"] - negatives
            recall = none["abstention_recall"]
            negative_none = round((recall["value"] if recall and recall["value"] is not None else 0.0) * negatives)
            positive_none = none["abstained"] - negative_none
            negative_pick = negatives - negative_none
            positive_wrong = none["wrong"] - negative_pick
            if none["correct"] + positive_wrong + positive_none != positives:
                raise ValueError(f"{row.key}: outcome counts do not split into positives and negatives")
            self._stack(left, y=y, parts=((none["correct"], _GOOD), (positive_wrong, _BAD), (positive_none, _NEUTRAL)))
            self._stack(right, y=y, parts=((negative_none, _GOOD), (negative_pick, _BAD)))
            self._label(figure, row, y=0.14 + 0.62 * (y + 0.5) / len(_DIRECT))
        panels = (
            (
                left,
                "A right tool exists",
                ((_GOOD, "picks a right tool"), (_BAD, "picks a wrong tool"), (_NEUTRAL, "says “none”")),
            ),
            (right, "No right tool", ((_GOOD, "says “none”"), (_BAD, "picks a tool anyway"))),
        )
        for axes, head, legend in panels:
            axes.set_xlim(0, 1)
            axes.set_ylim(-0.6, len(_DIRECT) - 0.4)
            axes.set_axis_off()
            axes.set_title(head, loc="left", fontsize=18, weight="bold", pad=34)
            handles = [Rectangle((0, 0), 1, 1, color=color) for color, _ in legend]
            axes.legend(
                handles,
                [text for _, text in legend],
                loc="lower left",
                bbox_to_anchor=(0, 1.0),
                ncol=3,
                frameon=False,
                fontsize=13,
                handlelength=1,
                columnspacing=1,
            )
        return self._save(figure, name="direct-none-option")

    @staticmethod
    def _stack(axes: Any, *, y: float, parts: tuple[tuple[int, str], ...]) -> None:
        total = sum(count for count, _ in parts)
        start = 0.0
        for count, color in parts:
            share = count / total
            axes.barh(y, share, left=start, color=color, height=0.62)
            if share > 0.07:
                ink = _INK if color == _NEUTRAL else "white"
                axes.text(
                    start + share / 2,
                    y,
                    f"{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=ink,
                    weight="bold",
                )
            start += share

    def _rerank(self) -> Path:
        search = self.rerank["hybrid@20"]
        figure = self._frame(
            "44,453 tools: search 20 candidates, then pick one",
            "200 held-out requests: share with a relevant tool ranked first.",
            f"Whiskers: 95% interval. Decision cost and median latency exclude the shared search. {self.rerank_footer}",
        )
        axes: Any = figure.add_axes((0.30, 0.14, 0.47, 0.62))
        for y, row in enumerate(reversed(_RERANK)):
            data = self.rerank[row.key]
            self._bar(axes, y=y, value=data["p_at_1"], interval=data["p_at_1_ci95"], color=row.color)
            centre = 0.14 + 0.62 * (y + 0.5) / len(_RERANK)
            self._label(figure, row, y=centre)
            if data["decider"]:
                usd = data["cost"]["usd_per_1000_searches"]
                text = f"${usd:.2f}\n{data['latency_ms']['decision']['p50'] / 1000:.2f} s"
            else:
                text = f"search: {data['latency_ms']['retrieval']['p50'] / 1000:.2f} s\n(shared by all rows)"
            figure.text(0.785, centre - 0.008, text, fontsize=14, va="center")
        figure.text(0.785, 0.78, "Per 1,000 searches · latency", fontsize=14, weight="bold", color=_MUTED)
        ceiling = search["ceiling"]
        axes.axvline(ceiling, color=_INK, ls="--", lw=1.4)
        axes.text(
            ceiling + 0.012, 2.55, f"ceiling {ceiling:.0%}:\na relevant tool is\namong the 20", fontsize=13, va="top"
        )
        self._percent_axis(axes, top=0.8, rows=len(_RERANK))
        return self._save(figure, name="rerank-44k")

    def _order(self) -> Path:
        figure = self._frame(
            "Shuffle the 20 candidates: does the top pick change?",
            "How often two runs rank the same tool first, on 200 requests (pairwise agreement).",
            f"Whiskers: 95% interval. “Same order” repeats come from a separate run. {self.order_footer}",
        )
        axes: Any = figure.add_axes((0.08, 0.20, 0.55, 0.54))
        baseline = self.order["same_order_noise_baseline"]["deciders"]
        lines = ["Relevant tool ranked first,", "original → shuffled (mean):", ""]
        for group, (key, name, color) in enumerate((("jev", "Jev", _JEV), ("logprob", "GPT-4.1 mini logprobs", _GPT))):
            decider = self.order["deciders"][key]
            for offset, (stat, label, alpha) in enumerate(
                (
                    (baseline[key]["pairwise_agreement"], "same order", 0.45),
                    (decider["pairwise_agreement"], "shuffled", 1.0),
                )
            ):
                x = group * 2.6 + offset
                value, (low, high) = stat["value"], stat["ci95"]
                axes.bar(x, value, color=color, alpha=alpha, width=0.85)
                axes.errorbar(x, value, yerr=[[value - low], [high - value]], color=_INK, capsize=4, lw=1.2)
                axes.text(x, 0.04, f"{value:.1%}", ha="center", color="white", weight="bold", fontsize=20)
                axes.text(x, -0.07, label, ha="center", fontsize=14, color=_MUTED)
            axes.text(group * 2.6 + 0.5, -0.15, name, ha="center", fontsize=17, weight="bold")
            identity, shuffled = decider["p_at_1"]["0"]["value"], decider["shuffle_p_at_1"]["mean"]["value"]
            lines.append(f"{name.split(' logprobs')[0]}: {identity:.1%} → {shuffled:.1%}")
        lines += [
            "",
            "Both change their pick when shuffled.",
            "GPT's precision fell in all four shuffles;",
            "Jev's did not.",
        ]
        figure.text(0.68, 0.72, "\n".join(lines), fontsize=16, va="top", linespacing=1.5)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(-0.7, 4.3)
        axes.set_xticks([])
        ticks = [0, 0.25, 0.5, 0.75, 1]
        axes.set_yticks(ticks, [f"{tick:.0%}" for tick in ticks])
        axes.grid(axis="y", color=_GRID)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        return self._save(figure, name="order-sensitivity")


def build_readme_charts(
    *, direct_summary: Path, decision_summary: Path, order_summary: Path, out_dir: Path, fmt: str = "svg"
) -> list[Path]:
    """Write the four README figures: direct choice, "none" outcomes, re-ranking at 44,453 tools, candidate order.

    Every number comes from the generated summaries; `fmt` is `svg` for the docs or `png` for social posts.
    """
    if fmt not in ("svg", "png"):
        raise ValueError("fmt must be 'svg' or 'png'")
    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in (direct_summary, decision_summary, order_summary)
    ]
    direct, decision, order = summaries
    return _ReadmeCharts(direct=direct, decision=decision, order=order, out_dir=out_dir, fmt=fmt).write()
