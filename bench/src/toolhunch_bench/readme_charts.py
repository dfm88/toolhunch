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

from toolhunch_bench.deciders import DECIDERS, DeciderName, DeciderSpec
from toolhunch_bench.direct import arm_decider

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

__all__ = ["build_cost_latency", "build_readme_charts"]

_SEARCH = "#8C939B"
_JEV, _GPT, _LUNA = (DECIDERS[name].color for name in (DeciderName.JEV, DeciderName.LOGPROB, DeciderName.LUNA))
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
class _Bar:
    """One row of a bar figure: an arm, its share and interval, and the cost and latency printed beside it."""

    key: str
    name: str
    detail: str
    color: str
    value: float
    interval: list[float]
    side: str


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
# The agents of the direct-choice test, by arm prefix: they are not registry deciders.
_AGENTS = {"agent-luna": ("GPT-6 Luna agent", _LUNA), "agent": ("GPT-4.1 mini agent", _GPT)}
# How each kind of 44,453-tool decider is asked, under its name.
_RERANK_DETAILS = {"logprob": "logprobs over option letters", "luna": "one letter, structured output"}
# Inches of a frame: the title's and the subtitle's tops below the figure's top, the footer above its bottom.
_TITLE_IN, _SUBTITLE_IN, _FOOTER_IN = 0.54, 1.09, 0.22


# A decider's name, color, same-order and shuffled pairwise agreement, P@1 in search order and shuffled (mean).
type _OrderRow = tuple[str, str, dict[str, Any] | None, dict[str, Any], float, float]


def _agent(arm: str) -> tuple[str, str] | None:
    """An agent arm's name and color, or `None` for a decider's arm."""
    prefix = arm.removesuffix("-all").split("@")[0]
    return _AGENTS.get(prefix) if arm.startswith("agent") else None


def _run_dates(started: Sequence[str]) -> str:
    return ", ".join(sorted({day[:10] for day in started}))


class _ReadmeCharts:
    """The README figures, every published decider in each: rows come from the summaries and the registry."""

    def __init__(
        self,
        *,
        direct: dict[str, Any],
        decisions: Sequence[dict[str, Any]],
        orders: Sequence[dict[str, Any]],
        luna: dict[str, Any],
        out_dir: Path,
        fmt: str,
    ) -> None:
        self.direct, self.decisions, self.orders, self.luna = direct, decisions, orders, luna
        self.pooled = {row["arm"]: row for row in direct["rows"] if row["catalog"] == "pooled"}
        self.search = next(
            row
            for summary in decisions
            for row in summary["heldout"]["rows"]
            if row["arm"] == "hybrid@20" and row["source"] == "plain"
        )
        self.out_dir = out_dir
        self.fmt = fmt
        hardware = _hardware(direct)
        self.machine = f"Local: one machine ({hardware}), no money cost. " if hardware else ""
        arms = [*self.pooled, *(row["arm"] for summary in decisions for row in summary["heldout"]["rows"])]
        if any("clef" in arm for arm in arms):
            self.machine += "Clef and Clef-flash have no pinned version."
        started = [direct["manifest"]["started"], *(run["manifest"]["started"] for run in direct.get("added_runs", []))]
        self.direct_footer = f"toolhunch · ToolRet · runs of {_run_dates(started)}"
        started = [run["manifest"]["started"] for summary in decisions for run in summary["runs"]]
        started.append(luna["runs"]["orders"]["started"])
        self.rerank_footer = f"toolhunch · ToolRet · runs of {_run_dates(started)}"
        started = [summary["manifest"]["started"] for summary in orders] + [luna["runs"]["orders"]["started"]]
        self.order_footer = f"toolhunch · ToolRet · runs of {_run_dates(started)}"

    def write(self) -> list[Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with cast("Any", matplotlib).rc_context(_STYLE):
            paths = [self._direct(), self._none_option(), self._rerank(), self._order()]
        return [*paths, _draw_cost_latency(self.direct, self.decisions, self.luna, out_dir=self.out_dir, fmt=self.fmt)]

    @staticmethod
    def _height(rows: int) -> float:
        """A figure tall enough for `rows` two-line labels: 9 inches up to 7 rows, as P1's figures were."""
        return max(9.0, 3.6 + 0.78 * rows)

    def _frame(self, title: str, subtitle: str, footer: str, *, height: float = 9.0) -> Any:
        figure: Any = Figure(figsize=(16, height), dpi=100)
        figure.text(0.04, 1 - _TITLE_IN / height, title, fontsize=28, weight="bold", va="top")
        figure.text(0.04, 1 - _SUBTITLE_IN / height, subtitle, fontsize=16, color=_MUTED, va="top")
        figure.text(0.04, _FOOTER_IN / height, footer, fontsize=11, color=_MUTED, linespacing=1.5)
        return figure

    def _save(self, figure: Any, *, name: str) -> Path:
        path = self.out_dir / f"{name}.{self.fmt}"
        metadata = {"Date": None} if self.fmt == "svg" else {"Software": None}
        figure.savefig(path, format=self.fmt, metadata=metadata)
        figure.clear()
        return path

    @staticmethod
    def _label(figure: Any, bar: _Bar, *, y: float, height: float) -> None:
        figure.text(0.29, y + 0.11 / height, bar.name, ha="right", weight="bold", fontsize=17)
        figure.text(0.29, y - 0.27 / height, bar.detail, ha="right", color=_MUTED, fontsize=13)

    @staticmethod
    def _bar(axes: Any, bar: _Bar, *, y: float) -> None:
        axes.barh(y, bar.value, color=bar.color, height=0.62, gid=f"bar-{_slug(bar.key)}")
        low, high = bar.interval
        axes.errorbar(bar.value, y, xerr=[[bar.value - low], [high - bar.value]], color=_INK, capsize=4, lw=1.2)
        axes.text(0.012, y, f"{bar.value:.0%}", color="white", weight="bold", va="center", fontsize=20)

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

    def _direct_bars(self) -> list[_Bar]:
        """Every published arm of the direct-choice test, grouped by model, the best model first; search last."""
        groups: dict[str, list[_Bar]] = {}
        for arm, row in self.pooled.items():
            agent = _agent(arm)
            decider = arm_decider(arm)
            if agent is None and (decider is None or not _spec(decider).published):
                continue
            warm = row["positive_cost"]["warm"]
            if warm.get("decision_latency_p50_ms") is None:
                continue
            whole = arm.endswith("-all")
            if agent is not None:
                name, color = agent
                detail = f"gets {'the whole catalog' if whole else '20 searched tools'} as functions"
            else:
                spec = _spec(cast("str", decider))
                name, color = spec.label, spec.color
                detail = f"reads {'the whole catalog' if whole else '20 searched tools'}"
            local = bool(warm.get("local"))
            cost = _usd(warm.get("billed_usd_per_1000"), local=local)
            cached = f"  ({warm['cache_share']:.1%} cached)" if warm.get("cache_share") else ""
            side = f"{cost}{cached}\n{warm['decision_latency_p50_ms'] / 1000:.2f} s"
            rate = row["relevant_pick_rate"]
            bar = _Bar(arm, name, detail, color, rate["value"], rate["ci95"], side)
            groups.setdefault(name, []).append(bar)
        ordered = sorted(groups.values(), key=lambda bars: -max(bar.value for bar in bars))
        bars = [bar for group in ordered for bar in sorted(group, key=lambda bar: not bar.key.endswith("-all"))]
        search = self.pooled["hybrid@20"]
        warm = search["positive_cost"]["warm"]
        side = f"{_usd(warm['billed_usd_per_1000'], local=False)}\nsearch {warm['latency_p50_ms'] / 1000:.2f} s"
        rate = search["relevant_pick_rate"]
        bars.append(
            _Bar(
                "hybrid@20", "Search only", "BM25 + embeddings, top result", _SEARCH, rate["value"], rate["ci95"], side
            )
        )
        return bars

    def _direct(self) -> Path:
        bars = self._direct_bars()
        height = self._height(len(bars))
        top, bottom = 2.0 / height, 1.1 / height
        figure = self._frame(
            "Which tool should the agent call?",
            "290 requests, catalogs of 40–101 tools: share where a relevant tool was picked. “None” counts as a miss.",  # noqa: RUF001
            f"Whiskers: 95% interval. Warm requests; latency of the decision alone, median. {self.machine}\n"
            f"{self.direct_footer}",
            height=height,
        )
        span = 1 - top - bottom
        axes: Any = figure.add_axes((0.30, bottom, 0.47, span))
        for index, bar in enumerate(bars):
            y = len(bars) - 1 - index
            self._bar(axes, bar, y=y)
            centre = bottom + span * (y + 0.5) / len(bars)
            self._label(figure, bar, y=centre, height=height)
            figure.text(0.785, centre - 0.07 / height, bar.side, fontsize=14, va="center")
        figure.text(
            0.785, 1 - top + 0.25 / height, "Per 1,000 requests · latency", fontsize=14, weight="bold", color=_MUTED
        )
        self._percent_axis(axes, top=1.0, rows=len(bars))
        return self._save(figure, name="direct-choice")

    def _none_option(self) -> Path:
        bars = self._direct_bars()
        height = self._height(len(bars))
        top, bottom = 2.3 / height, 1.0 / height
        figure = self._frame(
            "When the right tool is missing, who says “none”?",
            "Left: 290 requests with a relevant tool in the catalog. Right: 145 with it removed.",
            f"Removed tools leave approximate negatives: an unlabelled tool might still help.\n{self.direct_footer}",
            height=height,
        )
        span = 1 - top - bottom
        left: Any = figure.add_axes((0.30, bottom, 0.31, span))
        right: Any = figure.add_axes((0.65, bottom, 0.31, span))
        for index, bar in enumerate(bars):
            y = len(bars) - 1 - index
            none = self.pooled[bar.key]["none_option"]
            negatives = none["negative_requests"]
            positives = none["requests"] - negatives
            recall = none["abstention_recall"]
            negative_none = round((recall["value"] if recall and recall["value"] is not None else 0.0) * negatives)
            positive_none = none["abstained"] - negative_none
            negative_pick = negatives - negative_none
            positive_wrong = none["wrong"] - negative_pick
            if none["correct"] + positive_wrong + positive_none != positives:
                raise ValueError(f"{bar.key}: outcome counts do not split into positives and negatives")
            self._stack(left, y=y, parts=((none["correct"], _GOOD), (positive_wrong, _BAD), (positive_none, _NEUTRAL)))
            self._stack(right, y=y, parts=((negative_none, _GOOD), (negative_pick, _BAD)))
            self._label(figure, bar, y=bottom + span * (y + 0.5) / len(bars), height=height)
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
            axes.set_ylim(-0.6, len(bars) - 0.4)
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

    def _rerank_bars(self) -> list[_Bar]:
        """Every published decider's held-out K20 plain P@1, the best first; search last."""
        bars: list[_Bar] = []
        for point in _rerank_points(self.decisions, self.luna):
            decider = point.key.removeprefix("hybrid+").split("@")[0]  # a re-ranking arm is hybrid+<decider>@<K>
            spec = _spec(decider)
            detail = _RERANK_DETAILS.get(decider, "one choice question") + (" · on one machine" if point.local else "")
            mark = "*" if point.summed else ""
            side = f"{point.cost}\n{point.milliseconds / 1000:.2f} s{mark}"
            bars.append(_Bar(point.key, f"{spec.label} picks", detail, point.color, point.value, point.interval, side))
        bars.sort(key=lambda bar: -bar.value)
        search = self.search
        side = f"search: {search['latency_ms']['retrieval']['p50'] / 1000:.2f} s\n(shared by all rows)"
        bars.append(
            _Bar(
                "hybrid@20",
                "Search only",
                "BM25 + embeddings, top result",
                _SEARCH,
                search["p_at_1"],
                search["p_at_1_ci95"],
                side,
            )
        )
        return bars

    def _rerank(self) -> Path:
        bars = self._rerank_bars()
        height = self._height(len(bars))
        top, bottom = 2.0 / height, 1.2 / height
        summed = any(bar.side.endswith("*") for bar in bars)
        figure = self._frame(
            "44,453 tools: search 20 candidates, then pick one",
            "200 held-out requests: share with a relevant tool ranked first.",
            "Whiskers: 95% interval. Decision cost and median latency exclude the shared search."
            + (" *Asked one request at a time: the latency adds every call." if summed else "")
            + f"\n{self.machine}{self.rerank_footer}",
            height=height,
        )
        span = 1 - top - bottom
        axes: Any = figure.add_axes((0.30, bottom, 0.47, span))
        for index, bar in enumerate(bars):
            y = len(bars) - 1 - index
            self._bar(axes, bar, y=y)
            centre = bottom + span * (y + 0.5) / len(bars)
            self._label(figure, bar, y=centre, height=height)
            figure.text(0.785, centre - 0.07 / height, bar.side, fontsize=14, va="center")
        figure.text(
            0.785, 1 - top + 0.25 / height, "Per 1,000 searches · latency", fontsize=14, weight="bold", color=_MUTED
        )
        ceiling = self.search["ceiling"]
        axes.axvline(ceiling, color=_INK, ls="--", lw=1.4)
        axes.text(
            ceiling + 0.012,
            len(bars) - 0.45,
            f"ceiling {ceiling:.0%}:\na relevant tool is\namong the 20",
            fontsize=13,
            va="top",
        )
        self._percent_axis(axes, top=0.8, rows=len(bars))
        return self._save(figure, name="rerank-44k")

    def _order_rows(self) -> list[_OrderRow]:
        """Each published decider: name, color, same-order agreement where measured, shuffled agreement, P@1."""
        rows: list[_OrderRow] = []
        for summary in self.orders:
            baseline = summary["same_order_noise_baseline"]["deciders"]
            for key, decider in summary["deciders"].items():
                spec = _spec(key)
                if not spec.published:
                    continue
                same = baseline[key]["pairwise_agreement"] if baseline[key].get("comparable") else None
                identity = decider["p_at_1"]["0"]["value"]
                shuffled = decider["shuffle_p_at_1"]["mean"]["value"]
                rows.append((spec.label, spec.color, same, decider["pairwise_agreement"], identity, shuffled))
        luna = self.luna["order"]
        rows.append(
            (
                _spec("luna").label,
                _LUNA,
                self.luna["same_order_noise"]["pairwise_agreement"],
                luna["pairwise_agreement"],
                luna["p_at_1"]["0"]["value"],
                luna["shuffle_p_at_1"]["value"],
            )
        )
        return sorted(rows, key=lambda row: -row[3]["value"])

    def _order(self) -> Path:
        rows = self._order_rows()
        figure = self._frame(
            "Shuffle the 20 candidates: does the top pick change?",
            "How often two orders rank the same tool first, on 200 requests (pairwise agreement).",
            f"Whiskers: 95% interval. “Same order” repeats come from a separate run.\n{self.order_footer}",
        )
        axes: Any = figure.add_axes((0.07, 0.24, 0.62, 0.50))
        lines = ["Relevant tool ranked first,", "search order → shuffled (mean):", ""]
        x = 0.0
        for name, color, same, shuffled_agreement, identity, shuffled in rows:
            stats = [(same, "same", 0.45)] if same is not None else []
            stats.append((shuffled_agreement, "shuffled", 1.0))
            start = x
            for stat, label, alpha in stats:
                value, (low, high) = stat["value"], stat["ci95"]
                axes.bar(x, value, color=color, alpha=alpha, width=0.85, gid=f"order-{_slug(name)}-{label}")
                axes.errorbar(x, value, yerr=[[value - low], [high - value]], color=_INK, capsize=3, lw=1.1)
                axes.text(x, 0.03, f"{value:.0%}", ha="center", color="white", weight="bold", fontsize=13)
                axes.text(x, -0.06, label, ha="center", fontsize=11, color=_MUTED)
                x += 1
            wrapped = name.replace(" ", "\n", 1) if len(name) > 13 else name
            axes.text((start + x - 1) / 2, -0.11, wrapped, ha="center", va="top", fontsize=13, weight="bold")
            x += 0.6
            lines.append(f"{name}: {identity:.1%} → {shuffled:.1%}")
        lines += [
            "",
            "Pale bars: the same order asked",
            "again (Jev, GPT-4.1 mini, GPT-6 Luna).",
            "The others gave the same probabilities",
            "when a search was repeated.",
        ]
        figure.text(0.72, 0.76, "\n".join(lines), fontsize=14, va="top", linespacing=1.45)
        axes.set_ylim(0, 1.05)
        axes.set_xlim(-0.7, x - 0.9)
        axes.set_xticks([])
        ticks = [0, 0.25, 0.5, 0.75, 1]
        axes.set_yticks(ticks, [f"{tick:.0%}" for tick in ticks])
        axes.grid(axis="y", color=_GRID)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        return self._save(figure, name="order-sensitivity")


def build_readme_charts(
    *,
    direct_summary: Path,
    decision_summaries: Sequence[Path],
    order_summaries: Sequence[Path],
    luna_summary: Path,
    out_dir: Path,
    fmt: str = "svg",
) -> list[Path]:
    """Write the README figures: direct choice, "none", re-ranking at 44,453 tools, order, precision against latency.

    Each figure shows every decider the registry publishes, read from the summaries: `decision_summaries` and
    `order_summaries` earliest first, the first holding an arm wins. Every number comes from the generated summaries;
    `fmt` is `svg` for the docs or `png` for social posts.
    """
    if fmt not in ("svg", "png"):
        raise ValueError("fmt must be 'svg' or 'png'")

    def read(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    return _ReadmeCharts(
        direct=read(direct_summary),
        decisions=[read(path) for path in decision_summaries],
        orders=[read(path) for path in order_summaries],
        luna=read(luna_summary),
        out_dir=out_dir,
        fmt=fmt,
    ).write()


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
    lowered: bool = False
    summed: bool = False


# Label offsets in points from the point, per arm of the P2 cost-latency figure; a far label gets a connector.
_P2_OFFSETS: dict[str, tuple[float, float]] = {
    "strands-all": (-82, 92),
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

# Labels aligned against the side their offset points to: a long label placed in free space beyond the point.
_P2_LEFT_ALIGNED = frozenset({"strands-all"})

# The P3 figure's offsets, over the P2 ones when a P3 decider is drawn: Laya's low precision stretches the direct
# panel down to 20%, and the P2 offsets, set for a 50-90% axis, crowd its upper half.
_P3_OFFSETS: dict[str, tuple[float, float]] = {
    "strands-all": (-94, 44),
    "hybrid@20+strands": (11, -58),
    "jev-all": (-13, 24),
    "hybrid@20+jev": (8, -35),
    "hybrid@20+clef-flash": (-9, 44),
    "hybrid@20+clef": (-4, 44),
    "clef-flash-all": (44, 52),
    "clef-all": (70, 46),
    "agent-luna@20": (-6, -66),
    "agent@20": (24, -80),
    "agent-all": (40, -125),
    "hybrid@20+rizzo-flow": (12, 4),
    "rizzo-flow-all": (-5, -61),
    "hybrid+rizzo-flow@20": (-30, 80),
}
_P3_LEFT_ALIGNED = frozenset({"hybrid@20+clef", "agent-luna@20"})
_P3_ARMS = re.compile(r"laya|rizzo")


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


def _spec(decider: str) -> DeciderSpec:
    """A decider's registry entry, which gives its color and whether figures show it."""
    return DECIDERS[DeciderName(decider)]


def _slug(arm: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", arm.lower()).strip("-")


def _direct_points(direct: dict[str, Any]) -> list[_Point]:
    points: list[_Point] = []
    for row in direct["rows"]:
        arm = row["arm"]
        costs: dict[str, Any] = row.get("positive_cost") or {}
        warm: dict[str, Any] = costs.get("warm") or {}
        decider = arm_decider(arm)
        agent = arm.startswith("agent")
        if row["catalog"] != "pooled" or warm.get("decision_latency_p50_ms") is None:
            continue
        if not agent and (decider is None or not _spec(decider).published):
            continue
        name, detail, _, _ = _named(arm, decider or "luna")
        local = bool(warm.get("local"))
        points.append(
            _Point(
                key=arm,
                name=name,
                detail=detail,
                color=agent[1] if (agent := _agent(arm)) else _spec(decider or "luna").color,
                value=row["relevant_pick_rate"]["value"],
                interval=row["relevant_pick_rate"]["ci95"],
                milliseconds=warm["decision_latency_p50_ms"],
                cost=_usd(warm.get("billed_usd_per_1000"), local=local),
                local=local,
                lowered=bool(row.get("lower_detail")),
            )
        )
    return points


def _rerank_points(decisions: Sequence[dict[str, Any]], luna: dict[str, Any] | None) -> list[_Point]:
    points: dict[str, _Point] = {}
    for summary in decisions:
        for row in summary["heldout"]["rows"]:
            decider = row["decider"]
            if decider is None or not _spec(decider).published or (row["k"], row["source"]) != (20, "plain"):
                continue
            if row["arm"] in points or row["latency_ms"]["decision"] is None or row["p_at_1"] is None:
                continue  # a cell two rounds cannot hold has no P@1 to plot
            name, _, _, _ = _named(row["arm"], decider)
            local = "local" in row["cost"]
            points[row["arm"]] = _Point(
                key=row["arm"],
                name=name,
                detail="",
                color=_spec(decider).color,
                value=row["p_at_1"],
                interval=row["p_at_1_ci95"],
                milliseconds=row["latency_ms"]["decision"]["p50"],
                cost=_usd(row["cost"]["usd_per_1000_searches"], local=local),
                local=local,
                lowered=bool(row.get("lower_detail_searches")),
                summed=row["latency_ms"].get("decision_basis") == "sum of calls",
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


def _panel(
    axes: Any,
    points: Sequence[_Point],
    *,
    head: str,
    search: float,
    offsets: Mapping[str, tuple[float, float]],
    left_aligned: frozenset[str],
) -> None:
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
        # The figure's offsets first, then the P1 figure's, then the default `_named` gives an arm with neither.
        dx, dy = offsets.get(point.key) or (
            _TRADEOFF_LABELS[point.key][2:] if point.key in _TRADEOFF_LABELS else (12, 0)
        )
        align = "left" if dx > 0 or point.key in left_aligned else "right"
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
            (
                " · ".join(
                    part for part in (point.detail, "lower detail" if point.lowered else "", point.cost) if part
                ),
                -1,
                "top",
                {"color": _MUTED},
            ),
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
    # Up to 3 s as P1 and P2 drew it; a slower point widens the axis, which would otherwise drop it unseen.
    right = max(3.0, 1.3 * max(point.milliseconds for point in points) / 1000)
    axes.set_xlim(0.05, right)
    ticks = [tick for tick in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if tick < right]
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
    earlier run had cached, so totals that include the search would not compare. When a plotted decision's latency
    adds every call, as for a model asked one request at a time, the caption says so. A cell two rounds cannot hold
    has no P@1 and is not drawn. CLM is left out.
    """
    if fmt not in ("svg", "png"):
        raise ValueError("fmt must be 'svg' or 'png'")
    direct = json.loads(direct_summary.read_text(encoding="utf-8"))
    decisions = [json.loads(path.read_text(encoding="utf-8")) for path in decision_summaries]
    luna = None if luna_summary is None else json.loads(luna_summary.read_text(encoding="utf-8"))
    return _draw_cost_latency(direct, decisions, luna, out_dir=out_dir, fmt=fmt)


def _draw_cost_latency(
    direct: dict[str, Any],
    decisions: Sequence[dict[str, Any]],
    luna: dict[str, Any] | None,
    *,
    out_dir: Path,
    fmt: str,
) -> Path:
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
    if lowered := [point for point in points if point.lowered]:
        names = ", ".join(f"{point.name} ({point.detail})" if point.detail else point.name for point in lowered)
        notes.append(f"Lower detail: {names} sent cards below full detail to fit the model's window.")
    if any(point.summed for point in points):
        notes.append("Local deciders are asked one request at a time: their latency adds every call of a search.")
    footer = "\n".join([*notes, f"toolhunch · ToolRet · runs of {', '.join(dates)}"])
    p3 = any(_P3_ARMS.search(point.key) for point in points)
    offsets = _P2_OFFSETS | (_P3_OFFSETS if p3 else {})
    left_aligned = _P2_LEFT_ALIGNED | (_P3_LEFT_ALIGNED if p3 else frozenset[str]())
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
            offsets=offsets,
            left_aligned=left_aligned,
        )
        _panel(
            figure.add_axes((0.57, 0.19, 0.40, 0.57)),
            rerank_points,
            head="44,453 tools, 20 candidates (200 requests)",
            search=rerank_search,
            offsets=offsets,
            left_aligned=left_aligned,
        )
        path = out_dir / f"cost-latency.{fmt}"
        metadata = {"Date": None, "Description": footer} if fmt == "svg" else {"Software": None}
        figure.savefig(path, format=fmt, metadata=metadata)
        figure.clear()
    return path
