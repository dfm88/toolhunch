"""README figures from the generated direct-choice, decision, order and Luna summaries, without provider calls."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from toolhunch_bench.deciders import DECIDERS, DeciderName
from toolhunch_bench.direct import arm_decider

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["build_cost_latency", "build_readme_charts"]

_SEARCH, _JEV, _GPT, _LUNA = "#8C939B", "#2F6DB5", "#C0652B", "#7A4FB5"
_STRANDS, _CLEF, _CLEF_FLASH = "#16877A", "#B5306B", "#D98BB2"
_COLORS = {"jev": _JEV, "logprob": _GPT, "luna": _LUNA, "strands": _STRANDS, "clef": _CLEF, "clef-flash": _CLEF_FLASH}
_UNPUBLISHED = frozenset({"clm", "clm-local"})
"""Deciders left out of figures: CLM's figures wait for its authors to confirm our deployments match theirs."""
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
    _Row("agent-luna-all", "GPT-6 Luna agent", "gets the whole catalog as functions", _LUNA),
    _Row("agent-luna@20", "GPT-6 Luna agent", "gets 20 searched tools as functions", _LUNA),
    _Row("jev-all", "Jev", "reads the whole catalog", _JEV),
    _Row("hybrid@20+jev", "Jev", "reads 20 searched tools", _JEV),
    _Row("agent-all", "GPT-4.1 mini agent", "gets the whole catalog as functions", _GPT),
    _Row("agent@20", "GPT-4.1 mini agent", "gets 20 searched tools as functions", _GPT),
    _Row("hybrid@20", "Search only", "BM25 + embeddings, top result", _SEARCH),
)
_ORDER_NOTE = (
    "All three change their pick",
    "when the candidates are shuffled;",
    "pale bars: the same order, repeated.",
)
# Name, detail and offset in points from the point, per arm of the precision-latency figure.
_TRADEOFF_LABELS: dict[str, tuple[str, str, float, float]] = {
    "agent-luna-all": ("Luna agent", "all tools", 12, 4),
    "agent-luna@20": ("Luna agent", "20 tools", -12, 0),
    "jev-all": ("Jev", "all tools", -12, 0),
    "hybrid@20+jev": ("Jev", "20 tools", 12, -4),
    "agent-all": ("GPT-4.1 mini agent", "all tools", -12, -2),
    "agent@20": ("GPT-4.1 mini agent", "20 tools", 12, 4),
    "hybrid@20": ("Search only", "", 12, -4),
    "hybrid+jev@20": ("Jev", "", 12, 0),
    "hybrid+luna@20": ("GPT-6 Luna", "", -12, 4),
    "hybrid+logprob@20": ("GPT-4.1 mini", "", 12, -4),
}
_RERANK = (
    _Row("hybrid@20", "Search only", "BM25 + embeddings, top result", _SEARCH),
    _Row("hybrid+jev@20", "Jev picks", "one choice question", _JEV),
    _Row("hybrid+logprob@20", "GPT-4.1 mini picks", "logprobs over option letters", _GPT),
    _Row("hybrid+luna@20", "GPT-6 Luna picks", "one letter, structured output", _LUNA),
)


class _ReadmeCharts:
    def __init__(
        self,
        *,
        direct: dict[str, Any],
        decision: dict[str, Any],
        order: dict[str, Any],
        luna: dict[str, Any],
        out_dir: Path,
        fmt: str,
    ) -> None:
        self.pooled = {row["arm"]: row for row in direct["rows"] if row["catalog"] == "pooled"}
        self.rerank = {row["arm"]: row for row in decision["heldout"]["rows"] if row["source"] == "plain"}
        first = luna["first_pick"]
        self.rerank["hybrid+luna@20"] = {
            "decider": "luna",
            "p_at_1": first["p_at_1"]["value"],
            "p_at_1_ci95": first["p_at_1"]["ci95"],
            "cost": first["cost"],
            "latency_ms": {"decision": first["cost"]["latency_ms"]},
        }
        self.order, self.luna = order, luna
        self.out_dir = out_dir
        self.fmt = fmt
        manifest = direct["manifest"]
        added = [run["manifest"] for run in direct.get("added_runs", [])]
        self.direct_footer = self._footer(
            [manifest["started"], *(run["started"] for run in added)],
            manifest["jev"]["model"].split("@")[0],
            manifest["agent"]["model"],
            *(run["agent"]["model"] for run in added),
        )
        arms = order["manifest"]["arms"]
        models = (arms["hybrid+jev@20"]["model_id"].split("@")[0], arms["hybrid+logprob@20"]["model_id"].split("@")[0])
        heldout = next(run for run in decision["runs"] if run["run_id"] == decision["heldout"]["run_id"])
        runs = luna["runs"]
        self.rerank_footer = self._footer(
            [heldout["manifest"]["started"], runs["orders"]["started"]], *models, luna["model"]
        )
        self.order_footer = self._footer(
            [order["manifest"]["started"], runs["orders"]["started"]], *models, luna["model"]
        )

    @staticmethod
    def _footer(started: list[str], *models: str) -> str:
        dates = sorted({day[:10] for day in started})
        return f"toolhunch · ToolRet · runs of {' and '.join(dates)} · {', '.join(models)}"

    def write(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with cast("Any", matplotlib).rc_context(_STYLE):
            return [self._direct(), self._none_option(), self._rerank(), self._order(), self._tradeoff()]

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
            cached = f"  ({warm['cache_share']:.1%} cached)" if warm["cache_share"] else ""
            latency = warm["latency_p50_ms"] / 1000
            figure.text(0.785, centre - 0.008, f"{cost}{cached}\n{latency:.2f} s", fontsize=14, va="center")
        figure.text(0.785, 0.80, "Per 1,000 requests · latency", fontsize=14, weight="bold", color=_MUTED)
        self._percent_axis(axes, top=1.0, rows=len(_DIRECT))
        return self._save(figure, name="direct-choice")

    def _none_option(self) -> Path:
        figure = self._frame(
            "When the right tool is missing, who says “none”?",
            "Left: 290 requests with a relevant tool in the catalog. Right: 145 with it removed.",
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
        axes: Any = figure.add_axes((0.07, 0.20, 0.60, 0.54))
        baseline = self.order["same_order_noise_baseline"]["deciders"]
        luna = self.luna["order"]
        deciders = [
            (
                name,
                color,
                baseline[key]["pairwise_agreement"],
                self.order["deciders"][key]["pairwise_agreement"],
                self.order["deciders"][key]["p_at_1"]["0"]["value"],
                self.order["deciders"][key]["shuffle_p_at_1"]["mean"]["value"],
            )
            for key, name, color in (("jev", "Jev", _JEV), ("logprob", "GPT-4.1 mini", _GPT))
        ]
        deciders.append(
            (
                "GPT-6 Luna",
                _LUNA,
                self.luna["same_order_noise"]["pairwise_agreement"],
                luna["pairwise_agreement"],
                luna["p_at_1"]["0"]["value"],
                luna["shuffle_p_at_1"]["value"],
            )
        )
        lines = ["Relevant tool ranked first,", "original → shuffled (mean):", ""]
        for group, (name, color, same, shuffled_agreement, identity, shuffled) in enumerate(deciders):
            for offset, (stat, label, alpha) in enumerate(
                ((same, "same order", 0.45), (shuffled_agreement, "shuffled", 1.0))
            ):
                x = group * 2.6 + offset
                value, (low, high) = stat["value"], stat["ci95"]
                axes.bar(x, value, color=color, alpha=alpha, width=0.85)
                axes.errorbar(x, value, yerr=[[value - low], [high - value]], color=_INK, capsize=4, lw=1.2)
                axes.text(x, 0.04, f"{value:.1%}", ha="center", color="white", weight="bold", fontsize=20)
                axes.text(x, -0.07, label, ha="center", fontsize=14, color=_MUTED)
            axes.text(group * 2.6 + 0.5, -0.15, name, ha="center", fontsize=17, weight="bold")
            lines.append(f"{name}: {identity:.1%} → {shuffled:.1%}")
        lines += ["", *_ORDER_NOTE]
        figure.text(0.71, 0.72, "\n".join(lines), fontsize=16, va="top", linespacing=1.5)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(-0.7, 6.9)
        axes.set_xticks([])
        ticks = [0, 0.25, 0.5, 0.75, 1]
        axes.set_yticks(ticks, [f"{tick:.0%}" for tick in ticks])
        axes.grid(axis="y", color=_GRID)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        return self._save(figure, name="order-sensitivity")

    def _tradeoff(self) -> Path:
        search = self.rerank["hybrid@20"]
        figure = self._frame(
            "Precision, latency and cost",
            "Higher and further left is better. Labels: cost per 1,000 requests.",
            "Median latency, warm requests. Left includes search; right excludes the search every row shares "
            f"({search['latency_ms']['retrieval']['p50'] / 1000:.2f} s). Whiskers: 95% interval.\n{self.rerank_footer}",
        )
        panels = (
            (
                (0.08, 0.14, 0.40, 0.62),
                "Catalogs of 40–101 tools (290 requests)",  # noqa: RUF001
                [
                    (
                        row,
                        self.pooled[row.key]["relevant_pick_rate"]["value"],
                        self.pooled[row.key]["relevant_pick_rate"]["ci95"],
                        self.pooled[row.key]["positive_cost"]["warm"]["latency_p50_ms"],
                        self.pooled[row.key]["positive_cost"]["warm"]["billed_usd_per_1000"],
                    )
                    for row in _DIRECT
                ],
                (0.5, 0.9),
            ),
            (
                (0.57, 0.14, 0.40, 0.62),
                "44,453 tools, 20 candidates (200 requests)",
                [
                    (
                        row,
                        self.rerank[row.key]["p_at_1"],
                        self.rerank[row.key]["p_at_1_ci95"],
                        self.rerank[row.key]["latency_ms"]["decision"]["p50"],
                        self.rerank[row.key]["cost"]["usd_per_1000_searches"],
                    )
                    for row in _RERANK[1:]
                ],
                (0.15, 0.45),
            ),
        )
        for box, head, points, (low, high) in panels:
            axes: Any = figure.add_axes(box)
            for row, value, interval, milliseconds, usd in points:
                x = milliseconds / 1000
                axes.errorbar(
                    x, value, yerr=[[value - interval[0]], [interval[1] - value]], color=row.color, alpha=0.35, lw=2
                )
                axes.scatter(x, value, s=170, color=row.color, zorder=3, edgecolors="white", linewidths=1.5)
                name, detail, dx, dy = _TRADEOFF_LABELS[row.key]
                cost = "≈ $0" if usd < 0.01 else f"${usd:.2f}"
                align = "left" if dx > 0 else "right"
                for text, shift, va, style in (
                    (name, 1, "bottom", {"weight": "bold", "color": _INK}),
                    (f"{detail} · {cost}" if detail else cost, -1, "top", {"color": _MUTED}),
                ):
                    axes.annotate(
                        text,
                        (x, value),
                        xytext=(dx, dy + shift),
                        textcoords="offset points",
                        ha=align,
                        va=va,
                        fontsize=13,
                        **style,
                    )
            if box[0] > 0.5:
                axes.axhline(search["p_at_1"], color=_SEARCH, ls="--", lw=1.4)
                axes.text(0.105, search["p_at_1"] + 0.006, f"search only {search['p_at_1']:.0%}", color=_MUTED)
            axes.set_xscale("log")
            axes.set_xlim(0.1, 3)
            ticks = [0.1, 0.2, 0.5, 1, 2]
            axes.set_xticks(ticks, [f"{tick:g} s" for tick in ticks])
            axes.minorticks_off()
            axes.set_ylim(low, high)
            steps = [low + step * 0.1 for step in range(round((high - low) / 0.1) + 1)]
            axes.set_yticks(steps, [f"{step:.0%}" for step in steps])
            axes.grid(color=_GRID)
            axes.set_axisbelow(True)
            for side in ("top", "right"):
                axes.spines[side].set_visible(False)
            axes.set_title(head, loc="left", fontsize=17, weight="bold", pad=14)
        return self._save(figure, name="cost-latency")


def build_readme_charts(
    *,
    direct_summary: Path,
    decision_summary: Path,
    order_summary: Path,
    luna_summary: Path,
    out_dir: Path,
    fmt: str = "svg",
) -> list[Path]:
    """Write the README figures: direct choice, "none", re-ranking at 44,453 tools, order, precision against latency.

    Every number comes from the generated summaries; `fmt` is `svg` for the docs or `png` for social posts.
    """
    if fmt not in ("svg", "png"):
        raise ValueError("fmt must be 'svg' or 'png'")
    paths = (direct_summary, decision_summary, order_summary, luna_summary)
    direct, decision, order, luna = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return _ReadmeCharts(direct=direct, decision=decision, order=order, luna=luna, out_dir=out_dir, fmt=fmt).write()


@dataclass(frozen=True, slots=True)
class _Point:
    key: str
    name: str
    detail: str
    color: str
    value: float
    interval: list[float]
    milliseconds: float
    cost: str
    local: bool


# Label offsets in points from the point, per arm of the P2 cost-latency figure; a far label gets a connector.
_P2_OFFSETS: dict[str, tuple[float, float]] = {
    "strands-all": (-10, 42),
    "hybrid@20+strands": (14, -46),
    "hybrid@20+jev": (16, -40),
    "jev-all": (-14, 36),
    "hybrid@20+clef-flash": (-12, 62),
    "hybrid@20+clef": (6, -72),
    "clef-flash-all": (6, 84),
    "clef-all": (10, 46),
    "agent-luna-all": (16, 0),
    "agent-luna@20": (16, -38),
    "agent@20": (16, 26),
    "agent-all": (16, -26),
    "hybrid+strands@20": (12, 0),
    "hybrid+clef@20": (12, -16),
    "hybrid+clef-flash@20": (12, -4),
}


def _usd(usd: float | None, *, local: bool) -> str:
    if local:
        return "local"
    if usd is None:
        return "n/a"
    return "≈ $0" if usd < 0.01 else f"${usd:.2f}"


def _named(arm: str, decider: str) -> tuple[str, str, float, float]:
    """An arm's name, detail and label offset: the P1 figure's when it has the arm, else from the registry."""
    if arm in _TRADEOFF_LABELS:
        return _TRADEOFF_LABELS[arm]
    label = DECIDERS[DeciderName(decider)].label
    detail = (
        ""
        if "@20" in arm and not arm.startswith("hybrid@20+")
        else ("all tools" if arm.endswith("-all") else "20 tools")
    )
    return (label, detail, *_P2_OFFSETS.get(arm, (12, 0)))


def _slug(arm: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", arm.lower()).strip("-")


def _direct_points(direct: dict[str, Any]) -> list[_Point]:
    colors = {row.key: row.color for row in _DIRECT}
    points: list[_Point] = []
    for row in direct["rows"]:
        arm = row["arm"]
        costs: dict[str, Any] = row.get("positive_cost") or {}
        warm: dict[str, Any] = costs.get("warm") or {}
        decider = arm_decider(arm)
        agent = arm.startswith("agent")
        if row["catalog"] != "pooled" or warm.get("decision_latency_p50_ms") is None:
            continue
        if not agent and (decider is None or decider in _UNPUBLISHED):
            continue
        name, detail, _, _ = _named(arm, decider or "luna")
        local = bool(warm.get("local"))
        points.append(
            _Point(
                key=arm,
                name=name,
                detail=detail,
                color=colors.get(arm) or _COLORS[decider or "luna"],
                value=row["relevant_pick_rate"]["value"],
                interval=row["relevant_pick_rate"]["ci95"],
                milliseconds=warm["decision_latency_p50_ms"],
                cost=_usd(warm.get("billed_usd_per_1000"), local=local),
                local=local,
            )
        )
    return points


def _rerank_points(decisions: Sequence[dict[str, Any]], luna: dict[str, Any] | None) -> list[_Point]:
    points: dict[str, _Point] = {}
    for summary in decisions:
        for row in summary["heldout"]["rows"]:
            decider = row["decider"]
            if decider is None or decider in _UNPUBLISHED or (row["k"], row["source"]) != (20, "plain"):
                continue
            if row["arm"] in points or row["latency_ms"]["decision"] is None:
                continue
            name, _, _, _ = _named(row["arm"], decider)
            local = "local" in row["cost"]
            points[row["arm"]] = _Point(
                key=row["arm"],
                name=name,
                detail="",
                color=_COLORS[decider],
                value=row["p_at_1"],
                interval=row["p_at_1_ci95"],
                milliseconds=row["latency_ms"]["decision"]["p50"],
                cost=_usd(row["cost"]["usd_per_1000_searches"], local=local),
                local=local,
            )
    if luna is not None:
        first = luna["first_pick"]
        points["hybrid+luna@20"] = _Point(
            key="hybrid+luna@20",
            name=_TRADEOFF_LABELS["hybrid+luna@20"][0],
            detail="",
            color=_LUNA,
            value=first["p_at_1"]["value"],
            interval=first["p_at_1"]["ci95"],
            milliseconds=first["cost"]["latency_ms"]["p50"],
            cost=_usd(first["cost"]["usd_per_1000_searches"], local=False),
            local=False,
        )
    return list(points.values())


def _hardware(direct: dict[str, Any]) -> str | None:
    for run in direct.get("added_runs", []):
        deciders: dict[str, dict[str, Any]] = run["manifest"].get("deciders") or {}
        for entry in deciders.values():
            provenance: dict[str, Any] = entry.get("provenance") or {}
            if hardware := provenance.get("hardware"):
                return f"{hardware['chip']}, {hardware['memory_gb']} GB, {hardware['os']}"
    return None


def _panel(axes: Any, points: Sequence[_Point], *, head: str, search: float) -> None:
    for point in points:
        x = point.milliseconds / 1000
        low, high = point.interval
        axes.errorbar(
            x, point.value, yerr=[[point.value - low], [high - point.value]], color=point.color, alpha=0.35, lw=2
        )
        axes.scatter(
            x,
            point.value,
            s=170,
            zorder=3,
            color="white" if point.local else point.color,
            edgecolors=point.color if point.local else "white",
            linewidths=2.5 if point.local else 1.5,
            gid=f"point-{_slug(point.key)}" + ("-local" if point.local else ""),
        )
        dx, dy = _P2_OFFSETS.get(point.key) or _TRADEOFF_LABELS[point.key][2:]
        align = "left" if dx > 0 else "right"
        if abs(dy) > 20:
            axes.annotate(
                "",
                (x, point.value),
                xytext=(dx, dy),
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": _MUTED, "lw": 0.8, "shrinkA": 0, "shrinkB": 8},
            )
        for text, shift, va, style in (
            (point.name, 1, "bottom", {"weight": "bold", "color": _INK}),
            (f"{point.detail} · {point.cost}" if point.detail else point.cost, -1, "top", {"color": _MUTED}),
        ):
            axes.annotate(
                text,
                (x, point.value),
                xytext=(dx, dy + shift),
                textcoords="offset points",
                ha=align,
                va=va,
                fontsize=13,
                **style,
            )
    axes.axhline(search, color=_SEARCH, ls="--", lw=1.4)
    axes.text(0.052, search + 0.006, f"search only {search:.0%}", color=_MUTED)
    values = [point.interval[0] for point in points] + [point.interval[1] for point in points] + [search]
    low, high = math.floor(min(values) * 10) / 10, math.ceil(max(values) * 10) / 10
    axes.set_xscale("log")
    axes.set_xlim(0.05, 3)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    axes.set_xticks(ticks, [f"{tick:g} s" for tick in ticks])
    axes.minorticks_off()
    axes.set_ylim(low, high)
    steps = [low + step * 0.1 for step in range(round((high - low) / 0.1) + 1)]
    axes.set_yticks(steps, [f"{step:.0%}" for step in steps])
    axes.grid(color=_GRID)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.set_title(head, loc="left", fontsize=17, weight="bold", pad=14)


def build_cost_latency(
    *,
    direct_summary: Path,
    decision_summaries: Sequence[Path],
    luna_summary: Path | None = None,
    out_dir: Path,
    fmt: str = "svg",
) -> Path:
    """Draw precision against the decision's latency, labelled with cost, for every published decider.

    The direct-choice panel reads `direct_summary`; the 44,453-tool panel reads the held-out K20 plain rows of each
    decision summary (the first holding an arm wins) and GPT-6 Luna's first pick. A local decider is drawn hollow,
    its machine in the caption. Latency is the decision's alone: later runs searched with query embeddings an
    earlier run had cached, so totals that include the search would not compare. CLM is left out.
    """
    if fmt not in ("svg", "png"):
        raise ValueError("fmt must be 'svg' or 'png'")
    direct = json.loads(direct_summary.read_text(encoding="utf-8"))
    decisions = [json.loads(path.read_text(encoding="utf-8")) for path in decision_summaries]
    luna = None if luna_summary is None else json.loads(luna_summary.read_text(encoding="utf-8"))
    direct_points, rerank_points = _direct_points(direct), _rerank_points(decisions, luna)
    [direct_search] = [row for row in direct["rows"] if row["arm"] == "hybrid@20" and row["catalog"] == "pooled"]
    rerank_search = next(
        row["p_at_1"]
        for summary in decisions
        for row in summary["heldout"]["rows"]
        if row["arm"] == "hybrid@20" and row["source"] == "plain"
    )
    started = [direct["manifest"]["started"], *(run["manifest"]["started"] for run in direct.get("added_runs", []))]
    started += [run["manifest"]["started"] for summary in decisions for run in summary["runs"]]
    dates = sorted({day[:10] for day in started})
    hardware = _hardware(direct)
    notes = [
        "Median latency of the decision alone, warm requests: later runs searched with cached query embeddings, so "
        "search is left out. Whiskers: 95% interval."
    ]
    points = [*direct_points, *rerank_points]
    unpinned = any(point.key.startswith(("clef", "hybrid@20+clef", "hybrid+clef")) for point in points)
    if any(point.local for point in points) and hardware:
        notes.append(
            f"Hollow: ran on one machine ({hardware}), no money cost; its latency is that machine's."
            + (" Clef and Clef-flash have no pinned version." if unpinned else "")
        )
    elif unpinned:
        notes.append("Clef and Clef-flash have no pinned version.")
    footer = "\n".join([*notes, f"toolhunch · ToolRet · runs of {', '.join(dates)}"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with cast("Any", matplotlib).rc_context(_STYLE):
        figure: Any = Figure(figsize=(16, 9), dpi=100)
        figure.text(0.04, 0.94, "Precision, latency and cost", fontsize=28, weight="bold", va="top")
        figure.text(
            0.04,
            0.879,
            "Higher and further left is better. Labels: cost per 1,000 requests.",
            fontsize=16,
            color=_MUTED,
            va="top",
        )
        figure.text(0.04, 0.02, footer, fontsize=11, color=_MUTED, linespacing=1.5)
        _panel(
            figure.add_axes((0.08, 0.19, 0.40, 0.57)),
            direct_points,
            head="Catalogs of 40–101 tools (290 requests)",  # noqa: RUF001
            search=direct_search["relevant_pick_rate"]["value"],
        )
        _panel(
            figure.add_axes((0.57, 0.19, 0.40, 0.57)),
            rerank_points,
            head="44,453 tools, 20 candidates (200 requests)",
            search=rerank_search,
        )
        path = out_dir / f"cost-latency.{fmt}"
        metadata = {"Date": None, "Description": footer} if fmt == "svg" else {"Software": None}
        figure.savefig(path, format=fmt, metadata=metadata)
        figure.clear()
    return path
