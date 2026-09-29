"""Retrieval metrics (binary relevance), abstention metrics and summary statistics, in plain Python."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence, Set

__all__ = [
    "THRESHOLD_GRID",
    "Outcome",
    "SelectiveMetrics",
    "bootstrap_ci",
    "choose_threshold",
    "cluster_bootstrap_ci",
    "linear_fit",
    "ndcg_at_k",
    "percentile",
    "precision_at_1",
    "recall_at_k",
    "selective_metrics",
    "utility",
]

THRESHOLD_GRID = tuple(i / 20 for i in range(20))
"""The abstention thresholds a threshold is chosen from: 0.00, 0.05, ..., 0.95."""


def recall_at_k(ranked: Sequence[str], relevant: Set[str], *, k: int) -> float:
    """Share of the relevant ids found in the first `k` of `ranked`."""
    if not relevant:
        return 0.0
    return len({doc_id for doc_id in ranked[:k] if doc_id in relevant}) / len(relevant)


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], *, k: int = 10) -> float:
    """Normalised DCG with binary gains and a `log2(rank + 1)` discount; the ideal ranks every relevant id first."""
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, doc_id in enumerate(ranked[:k], start=1) if doc_id in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def precision_at_1(ranked: Sequence[str], relevant: Set[str]) -> float:
    """1.0 when the first result is relevant, else 0.0."""
    return 1.0 if ranked and ranked[0] in relevant else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    """The `q`-th percentile (0-100), interpolating linearly between ranks like NumPy's default.

    Raises:
        ValueError: `values` is empty.
    """
    if not values:
        raise ValueError("percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of the mean of `values`, at level `1 - alpha`.

    Raises:
        ValueError: `values` is empty.
    """
    if not values:
        raise ValueError("bootstrap of no values")
    rng = random.Random(seed)
    means = [sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples)]
    return percentile(means, 100 * alpha / 2), percentile(means, 100 * (1 - alpha / 2))


def cluster_bootstrap_ci(
    clusters: Sequence[Sequence[float]], *, resamples: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval of the mean of every value in `clusters`, at level `1 - alpha`.

    Whole clusters are resampled, so values that belong together, such as the repeats of one search, stay together.
    With one value per cluster the interval is the one `bootstrap_ci` gives for those values and the same seed.

    Raises:
        ValueError: `clusters` is empty, or one of them is.
    """
    if not clusters or not all(clusters):
        raise ValueError("bootstrap of no values")
    totals = [(sum(cluster), len(cluster)) for cluster in clusters]
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        drawn = rng.choices(totals, k=len(totals))
        means.append(sum(total for total, _ in drawn) / sum(count for _, count in drawn))
    return percentile(means, 100 * alpha / 2), percentile(means, 100 * (1 - alpha / 2))


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a decider did with one search under one abstention rule.

    Attributes:
        answered: It named a tool instead of abstaining.
        correct: It answered with a gold tool.
        gold_in_candidates: A gold tool was among the candidates it chose from; never true for a negative.
    """

    answered: bool
    correct: bool
    gold_in_candidates: bool


@dataclass(frozen=True, slots=True)
class SelectiveMetrics:
    """How a decider that may abstain did over a set of searches.

    An answer is wrong unless it is correct, so every answer to a negative is wrong.

    Attributes:
        coverage: The searches answered, over all searches.
        selective_accuracy: The correct answers, over the searches answered; `None` when none was.
        wrong_tool_rate: The wrong answers, over all searches.
        abstention_precision: The abstentions on a search without a gold tool among its candidates, over all
            abstentions; `None` without abstentions.
        abstention_recall: The same abstentions, over the searches without a gold tool among their candidates;
            `None` when every search had one.
    """

    coverage: float
    selective_accuracy: float | None
    wrong_tool_rate: float
    abstention_precision: float | None
    abstention_recall: float | None


def selective_metrics(outcomes: Sequence[Outcome]) -> SelectiveMetrics:
    """Coverage, selective accuracy, wrong-tool rate and abstention precision and recall of `outcomes`.

    Raises:
        ValueError: `outcomes` is empty.
    """
    if not outcomes:
        raise ValueError("metrics of no outcomes")
    answered = sum(outcome.answered for outcome in outcomes)
    correct = sum(outcome.answered and outcome.correct for outcome in outcomes)
    without_gold = sum(not outcome.gold_in_candidates for outcome in outcomes)
    abstained = len(outcomes) - answered
    rightly_abstained = sum(not outcome.answered and not outcome.gold_in_candidates for outcome in outcomes)
    return SelectiveMetrics(
        coverage=answered / len(outcomes),
        selective_accuracy=correct / answered if answered else None,
        wrong_tool_rate=(answered - correct) / len(outcomes),
        abstention_precision=rightly_abstained / abstained if abstained else None,
        abstention_recall=rightly_abstained / without_gold if without_gold else None,
    )


def utility(outcomes: Sequence[Outcome]) -> float:
    """(correct - wrong) / searches: a correct answer scores 1, a wrong one -1 and an abstention 0.

    Raises:
        ValueError: `outcomes` is empty.
    """
    if not outcomes:
        raise ValueError("utility of no outcomes")
    correct = sum(outcome.answered and outcome.correct for outcome in outcomes)
    wrong = sum(outcome.answered and not outcome.correct for outcome in outcomes)
    return (correct - wrong) / len(outcomes)


def choose_threshold(
    outcomes_at: Callable[[float], Sequence[Outcome]], *, grid: Sequence[float] = THRESHOLD_GRID
) -> float:
    """The threshold of `grid` whose outcomes have the highest `utility`; of equal scores, the lowest threshold wins.

    Args:
        outcomes_at: The outcomes of the same searches with a given threshold applied.
        grid: The thresholds to try, in any order.

    Raises:
        ValueError: `grid` is empty.
    """
    if not grid:
        raise ValueError("no threshold to choose from")
    thresholds = sorted(grid)
    scores = [utility(outcomes_at(threshold)) for threshold in thresholds]
    return thresholds[scores.index(max(scores))]  # the first best, so the lowest threshold among ties


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    """The least-squares line through the points `(xs[i], ys[i])`, as `(intercept, slope)`.

    Raises:
        ValueError: The sequences differ in length, hold fewer than two points, or `xs` is constant.
    """
    try:
        slope, intercept = statistics.linear_regression(xs, ys)
    except statistics.StatisticsError as error:
        raise ValueError(f"no line fits these points: {error}") from None
    return intercept, slope
