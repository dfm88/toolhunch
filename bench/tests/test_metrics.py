import math
import random
from typing import Any

import numpy as np
import pytest

from toolhunch_bench.metrics import bootstrap_ci, ndcg_at_k, percentile, precision_at_1, recall_at_k


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
