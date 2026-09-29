import math
import random
from typing import Any

import numpy as np
import pytest

from toolhunch_bench.metrics import (
    THRESHOLD_GRID,
    Outcome,
    bootstrap_ci,
    choose_threshold,
    cluster_bootstrap_ci,
    linear_fit,
    ndcg_at_k,
    percentile,
    precision_at_1,
    recall_at_k,
    selective_metrics,
    utility,
)


def test_ranking_metrics_by_hand() -> None:
    ranked, relevant = ["a", "x", "b"], {"a", "b"}

    assert recall_at_k(ranked, relevant, k=1) == 0.5
    assert recall_at_k(ranked, relevant, k=3) == 1.0
    assert ndcg_at_k(ranked, relevant) == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k(["b", "a"], relevant) == 1.0
    assert precision_at_1(ranked, relevant) == 1.0
    assert precision_at_1(["x", "a"], relevant) == 0.0


def test_empty_ranking_scores_zero() -> None:
    assert (recall_at_k([], {"a"}, k=5), ndcg_at_k([], {"a"}), precision_at_1([], {"a"})) == (0.0, 0.0, 0.0)


def test_percentile_interpolates_like_numpy() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 95) == pytest.approx(3.85)
    rng = random.Random(0)
    values = [rng.expovariate(1.0) for _ in range(101)]  # latency-like
    numpy: Any = np
    for q in (0, 5, 50, 90, 95, 99, 100):
        assert percentile(values, q) == pytest.approx(float(numpy.percentile(values, q)))


def test_bootstrap_ci() -> None:
    assert bootstrap_ci([0.5] * 20) == (0.5, 0.5)
    hits = [1.0] * 30 + [0.0] * 20  # e.g. P@1 over 50 tasks
    low, high = bootstrap_ci(hits, seed=1)
    assert bootstrap_ci(hits, seed=1) == (low, high)
    assert low < 0.6 < high


def test_cluster_bootstrap_resamples_whole_clusters_and_pools_their_values() -> None:
    hits = [1.0] * 30 + [0.0] * 20
    assert cluster_bootstrap_ci([[hit] for hit in hits], resamples=2000, seed=3) == bootstrap_ci(
        hits, resamples=2000, seed=3
    )
    # Three clusters drawn three times: a resample holds the nine-value cluster 0, 1, 2 or 3 times with probability
    # 8/27, 12/27, 6/27 and 1/27, so the median resample holds it once. Its mean over the pooled values is 9 / 11;
    # the mean of its cluster means would be 1/3.
    assert cluster_bootstrap_ci([[1.0] * 9, [0.0], [0.0]], resamples=2000, seed=0, alpha=1.0) == pytest.approx(
        (9 / 11, 9 / 11)
    )


def test_selective_metrics_definitions() -> None:
    outcomes = [
        Outcome(True, True, True),
        Outcome(True, False, True),
        Outcome(False, False, False),
        Outcome(False, False, True),
        Outcome(True, False, False),
    ]
    m = selective_metrics(outcomes)
    assert (m.coverage, m.selective_accuracy, m.wrong_tool_rate) == pytest.approx((3 / 5, 1 / 3, 2 / 5))
    assert (m.abstention_precision, m.abstention_recall) == pytest.approx((1 / 2, 1 / 2))
    assert utility(outcomes) == pytest.approx((1 - 2) / 5)
    # Nothing answered, and gold among every candidate list: the ratios without a denominator are None.
    silent = selective_metrics([Outcome(False, False, True)] * 2)
    assert (silent.coverage, silent.selective_accuracy, silent.wrong_tool_rate) == (0.0, None, 0.0)
    assert (silent.abstention_precision, silent.abstention_recall) == (0.0, None)


def test_threshold_is_the_best_utility_with_ties_to_the_lower() -> None:
    # Four searches: the best card's probability, and whether that card is a gold tool. A search answers when its
    # best card reaches tau. U = (correct - wrong) / 4 is 0 up to 0.20, 1/4 from 0.25 to 0.40, 0 from 0.45 to 0.60,
    # 1/4 again from 0.65 to 0.90 and 0 at 0.95.
    searches = [(0.9, True), (0.6, False), (0.4, True), (0.2, False)]
    asked: list[float] = []

    def outcomes_at(tau: float) -> list[Outcome]:
        asked.append(tau)
        return [Outcome(best >= tau, right and best >= tau, True) for best, right in searches]

    assert tuple(i / 20 for i in range(20)) == THRESHOLD_GRID
    assert choose_threshold(outcomes_at) == 0.25  # tied with 0.30 ... 0.40 and 0.65 ... 0.90
    assert sorted(asked) == list(THRESHOLD_GRID)  # every tau of the grid was scored
    assert utility(outcomes_at(0.25)) == 0.25
    # The lower of two tied thresholds wins whatever order the grid lists them in.
    assert choose_threshold(outcomes_at, grid=(0.9, 0.7, 0.5)) == 0.7
    # When no threshold does better than answering everything, nothing is added.
    assert choose_threshold(lambda tau: [Outcome(True, True, True)]) == 0.0


def test_linear_fit_recovers_overhead_and_slope() -> None:
    assert linear_fit([100, 200, 400], [350, 400, 500]) == pytest.approx((300.0, 0.5))
    with pytest.raises(ValueError, match="constant"):
        linear_fit([100, 100], [350, 400])
