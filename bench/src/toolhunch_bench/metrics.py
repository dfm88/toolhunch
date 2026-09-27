"""Retrieval metrics (binary relevance) and summary statistics, in plain Python."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence, Set

__all__ = ["bootstrap_ci", "ndcg_at_k", "percentile", "precision_at_1", "recall_at_k"]


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
