"""Compare two decision runs of the same searches: the same model on two deployments, or on two devices.

Searches are matched on task, variant, query source, K, repeat and candidate order. For each pair answered on both
sides the comparison reads the final distribution (each card's probability and the reserved option's) and the top
card. It reports what it could not match or compare instead of dropping it.
"""

from __future__ import annotations

import json
import statistics
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["compare_runs"]

type _Key = tuple[str, str, str, int, int, int | None]


def _searches(run_dir: Path, decider: str) -> dict[_Key, dict[str, Any]]:
    searches: dict[_Key, dict[str, Any]] = {}
    for line in (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record") == "search" and record.get("decider") == decider:
            key = (
                record["task"],
                record["variant"],
                record["source"],
                record["k"],
                record["repeat"],
                record.get("order_seed"),
            )
            searches[key] = record
    return searches


def _distribution(record: dict[str, Any]) -> dict[str, float]:
    return {**record["probabilities"], "none": record["none_probability"]}


def _top(record: dict[str, Any]) -> str | None:
    return record["ranked"][0] if record["ranked"] else None


def compare_runs(reference: Path, candidate: Path, *, reference_decider: str, candidate_decider: str) -> dict[str, Any]:
    """Compare `candidate_decider`'s searches in `candidate` with `reference_decider`'s in `reference`.

    Returns:
        `matched` searches present on both sides; `unmatched_reference` and `unmatched_candidate`; `errored`, the
        matched pairs where either side failed; `compared`, the rest. Over the compared pairs: `top1_agreement`, the
        share with the same top card; `median_abs_dp`, the median over searches of each search's largest absolute
        probability difference, over the options both sides share; `max_abs_dp`, the largest of those, with
        `max_abs_dp_search`; and `options_mismatched`, pairs whose option sets differ.
    """
    left = _searches(reference, reference_decider)
    right = _searches(candidate, candidate_decider)
    matched = left.keys() & right.keys()
    errored = [key for key in matched if left[key]["error"] is not None or right[key]["error"] is not None]
    compared = sorted(matched - set(errored))
    largest: list[tuple[float, _Key]] = []
    same_top = 0
    mismatched = 0
    for key in compared:
        before, after = _distribution(left[key]), _distribution(right[key])
        shared = before.keys() & after.keys()
        mismatched += before.keys() != after.keys()
        largest.append((max((abs(before[option] - after[option]) for option in shared), default=0.0), key))
        same_top += _top(left[key]) == _top(right[key])
    worst = max(largest, default=None)
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "reference_decider": reference_decider,
        "candidate_decider": candidate_decider,
        "matched": len(matched),
        "unmatched_reference": len(left.keys() - right.keys()),
        "unmatched_candidate": len(right.keys() - left.keys()),
        "errored": len(errored),
        "compared": len(compared),
        "options_mismatched": mismatched,
        "top1_agreement": same_top / len(compared) if compared else None,
        "median_abs_dp": statistics.median(value for value, _ in largest) if largest else None,
        "max_abs_dp": None if worst is None else worst[0],
        "max_abs_dp_search": None
        if worst is None
        else dict(zip(("task", "variant", "source", "k", "repeat", "order_seed"), worst[1], strict=True)),
    }
