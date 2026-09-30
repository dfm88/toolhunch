"""GPT-6 Luna as a structured-output decider on the ToolRet held-out searches: first pick, "none" and candidate order.

Three `toolhunch-bench decision --deciders luna` runs at K20 on the plain request: five candidate orders with a forced
pick (no "none" option; order 0 is the retrieval's), the reserved "none" option on positives and negatives, and three
same-order repeats with a forced pick, for run-to-run noise. The figures line up with the logprob decider's in
`2026-09-toolret-decision` and `2026-09-toolret-order`. A structured answer has no probabilities, so there is no
threshold reading and no averaging over orders.
"""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from genai_prices import Usage, calc_price

from toolhunch_bench.decision_report import published_manifest
from toolhunch_bench.metrics import Outcome, cluster_bootstrap_ci, percentile, selective_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["build_luna_report"]

ARM = "hybrid+luna@20"
_SEEDS = (0, 1, 2, 3, 4)

type Record = dict[str, Any]


def _metric(clusters: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Mean and 95% task-cluster bootstrap interval (2,000 resamples, seed 0), as in the other ToolRet reports."""
    values = [value for cluster in clusters for value in cluster]
    low, high = cluster_bootstrap_ci(clusters, resamples=2000, seed=0)
    return {"value": statistics.fmean(values), "ci95": [low, high], "tasks": len(clusters)}


def _right(record: Record) -> float:
    return float(bool(record["ranked"]) and record["ranked"][0] in record["relevant"])


def _read(run_dir: Path, **expected: Any) -> tuple[dict[str, Any], list[Record]]:
    """The manifest and the Luna arm's searches of `run_dir`, checked against the protocol it stands for."""
    manifest: dict[str, Any] = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    arm = manifest["arms"][ARM]
    shape = {
        "split": manifest["split"],
        "sources": manifest["sources"],
        "negatives": manifest["negatives"],
        "repeat": manifest["repeat"],
        "reserved_option": arm["reserved_option"],
        "orders": "orders" in manifest,
    }
    if (found := {key: shape[key] for key in expected}) != expected:
        raise ValueError(f"{run_dir} is not the expected run: {found} instead of {expected}")
    lines = (run_dir / "run.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return manifest, [r for r in records if r.get("record") == "search" and r["arm"] == ARM]


def _by_task(records: Sequence[Record], field: str) -> dict[str, list[Record]]:
    """Positives by task, sorted by `field`; a task with a failed search is left out."""
    grouped: dict[str, list[Record]] = {}
    for record in records:
        if record["variant"] == "positive":
            grouped.setdefault(record["task"], []).append(record)
    return {
        task: sorted(rows, key=lambda row: row[field])
        for task, rows in grouped.items()
        if all(row["error"] is None for row in rows)
    }


def _errors(records: Sequence[Record]) -> dict[str, Any]:
    failed = [r for r in records if r["error"] is not None]
    return {
        "searches": len(failed),
        "tasks": sorted({r["task"] for r in failed}),
        "messages": sorted({r["error"] for r in failed}),
    }


def _agreement(groups: Sequence[Sequence[Record]]) -> dict[str, Any]:
    return _metric(
        [[float(a["ranked"][0] == b["ranked"][0]) for a, b in itertools.combinations(rows, 2)] for rows in groups]
    )


def _cost(records: Sequence[Record], model: str) -> dict[str, Any]:
    input_tokens = sum(r["usage"]["input_tokens"] for r in records)
    output_tokens = sum(r["usage"]["output_tokens"] for r in records)
    # Per request: Luna's price tiers (2x above 272K input tokens) apply to one request, not to a sum.
    usd = float(
        sum(
            calc_price(
                Usage(input_tokens=r["usage"]["input_tokens"], output_tokens=r["usage"]["output_tokens"]),
                model_ref=model,
                provider_id="openai",
            ).total_price
            for r in records
        )
    )
    seconds = [r["decision_seconds"] * 1000 for r in records]
    return {
        "searches": len(records),
        "input_tokens_per_search": input_tokens / len(records),
        "output_tokens_per_search": output_tokens / len(records),
        "usd_per_1000_searches": usd / len(records) * 1000,
        "latency_ms": {"p50": percentile(seconds, 50), "p95": percentile(seconds, 95)},
    }


def build_luna_report(*, orders_run: Path, none_run: Path, repeats_run: Path, out_dir: Path) -> dict[str, Any]:
    """Write `summary.json` and `README.md` for the three Luna runs to `out_dir`; no provider is contacted.

    A failed search (Luna sometimes returns no content) is counted and left out: of every rate in the "none" reading,
    and with its whole task from the order and noise comparisons, whose tasks must answer in every order or repeat.

    Raises:
        ValueError: A run does not have the protocol of its role, or the runs do not share tasks and dataset.
    """
    orders_manifest, orders = _read(orders_run, split="heldout", reserved_option=False, orders=True, repeat=1)
    none_manifest, nones = _read(none_run, split="heldout", reserved_option=True, negatives=True, orders=False)
    repeats_manifest, repeats = _read(repeats_run, split="heldout", reserved_option=False, repeat=3, orders=False)
    manifests = (orders_manifest, none_manifest, repeats_manifest)
    if len({json.dumps([m["tasks"]["sha256"], m["dataset"]], sort_keys=True) for m in manifests}) != 1:
        raise ValueError("the three Luna runs must share tasks and dataset")
    model = orders_manifest["arms"][ARM]["model_id"].partition("@")[0]

    shuffled = _by_task(orders, "order_seed")
    if any([row["order_seed"] for row in rows] != list(_SEEDS) for rows in shuffled.values()):
        raise ValueError(f"{orders_run} lacks some of the five orders of a task")
    identity = [r for r in orders if r["order_seed"] == 0 and r["variant"] == "positive" and r["error"] is None]
    slots = {
        task: [row["candidates"].index(row["ranked"][0]) + 1 for row in rows[1:]] for task, rows in shuffled.items()
    }
    repeated = _by_task(repeats, "repeat")
    none_errors = _errors(nones)
    nones = [r for r in nones if r["error"] is None]

    outcomes = [
        Outcome(
            answered=not r["abstained"],
            correct=not r["abstained"] and r["variant"] == "positive" and bool(_right(r)),
            gold_in_candidates=r["gold_in_candidates"],
        )
        for r in nones
    ]
    positives = [o for r, o in zip(nones, outcomes, strict=True) if r["variant"] == "positive"]
    negatives = [o for r, o in zip(nones, outcomes, strict=True) if r["variant"] == "negative"]
    summary: dict[str, Any] = {
        "model": model,
        "decider": "structured output, reasoning off: one letter from a JSON-schema enum; no probabilities",
        "bootstrap": {"resamples": 2000, "seed": 0, "level": 0.95, "unit": "task"},
        "runs": {
            "orders": published_manifest(orders_manifest),
            "none": published_manifest(none_manifest),
            "repeats": published_manifest(repeats_manifest),
        },
        "first_pick": {
            "definition": "forced pick (no none option), retrieval order: a relevant tool first on positives",
            "errors": _errors([r for r in orders if r["order_seed"] == 0]),
            "p_at_1": _metric([[_right(r)] for r in identity]),
            "hybrid_p_at_1": _metric([[float(r["candidates"][0] in r["relevant"])] for r in identity]),
            "delta_vs_hybrid": _metric([[_right(r) - float(r["candidates"][0] in r["relevant"])] for r in identity]),
            "ceiling": statistics.fmean(r["gold_in_candidates"] for r in identity),
            "cost": _cost(identity, model),
        },
        "none_option": {
            "definition": 'the reserved "none" option as the last choice; positives and gold-removed negatives',
            "errors": none_errors,
            **asdict(selective_metrics(outcomes)),
            "correct": sum(o.correct for o in outcomes),
            "wrong": sum(o.answered and not o.correct for o in outcomes),
            "abstained": sum(not o.answered for o in outcomes),
            "searches": len(outcomes),
            "positives": {
                "right": sum(o.correct for o in positives) / len(positives),
                "wrong": sum(o.answered and not o.correct for o in positives) / len(positives),
                "none": sum(not o.answered for o in positives) / len(positives),
            },
            "negatives_none": sum(not o.answered for o in negatives) / len(negatives),
            "cost": _cost(nones, model),
        },
        "order": {
            "definition": "forced pick in five candidate orders (0 = retrieval order, 1-4 task-seeded shuffles)",
            "positive_tasks": len(shuffled),
            "errors": _errors(orders),
            "p_at_1": {str(seed): _metric([[_right(rows[seed])] for rows in shuffled.values()]) for seed in _SEEDS},
            "shuffle_p_at_1": _metric([[_right(row) for row in rows[1:]] for rows in shuffled.values()]),
            "pairwise_agreement": _agreement(list(shuffled.values())),
            "slot_1": _metric([[float(slot == 1) for slot in values] for values in slots.values()]),
            "slots_1_to_3": _metric([[float(slot <= 3) for slot in values] for values in slots.values()]),
            "uniform_slots_1_to_3": 0.15,
        },
        "same_order_noise": {
            "definition": "forced pick, retrieval order, three fresh asks per positive (decision cache bypassed)",
            "positive_tasks": len(repeated),
            "errors": _errors(repeats),
            "p_at_1_by_repeat": {
                str(index): _metric([[_right(rows[index])] for rows in repeated.values()]) for index in range(3)
            },
            "pairwise_agreement": _agreement(list(repeated.values())),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    def ci(metric: dict[str, Any]) -> str:
        low, high = metric["ci95"]
        return f"{metric['value']:.3f} ({low:.2f} to {high:.2f})"

    first, none, order, noise = (
        summary["first_pick"],
        summary["none_option"],
        summary["order"],
        summary["same_order_noise"],
    )
    runs = summary["runs"]
    lines = [
        f"# ToolRet held-out with {summary['model']} as a structured-output decider",
        "",
        f"200 held-out tasks, 44,453 tools, hybrid retrieval at K20, plain request. {summary['model']} answers the "
        "logprob decider's letter prompt with reasoning off, as one letter from a JSON-schema enum: it returns at "
        "most 5 top logprobs, too few to rank 20 options. The answer has no probabilities, so there is no "
        "threshold reading and no averaging over orders. 95% intervals resample tasks (2,000 resamples, seed 0).",
        "",
        "| run | role | commit |",
        "|---|---|---|",
        *(
            f"| `{manifest['run_id']}` | {role} | `{manifest['git']['commit'][:7]}"
            f"{' (dirty)' if manifest['git']['dirty'] else ''}` |"
            for role, manifest in runs.items()
        ),
        "",
        "## First pick",
        "",
        "| P@1 (95% CI) | hybrid P@1 | Δ vs hybrid (95% CI) | ceiling | $ / 1,000 searches | latency p50 / p95 ms |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {ci(first['p_at_1'])} | {first['hybrid_p_at_1']['value']:.3f} | {ci(first['delta_vs_hybrid'])} | "
        f"{first['ceiling']:.3f} | ${first['cost']['usd_per_1000_searches']:.3f} | "
        f"{first['cost']['latency_ms']['p50']:.0f} / {first['cost']['latency_ms']['p95']:.0f} |",
        "",
        '## The "none" option',
        "",
        "| searches | correct | wrong | abstained | positives right / wrong / none | negatives none |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {none['searches']} | {none['correct']} | {none['wrong']} | {none['abstained']} | "
        f"{none['positives']['right']:.1%} / {none['positives']['wrong']:.1%} / {none['positives']['none']:.1%} | "
        f"{none['negatives_none']:.1%} |",
        "",
        "## Candidate order",
        "",
        "| P@1 by order 0 / 1 / 2 / 3 / 4 | shuffle P@1 | pairwise agreement | same-order agreement | slots 1-3 |",
        "|---|---:|---:|---:|---:|",
        "| "
        + " / ".join(f"{order['p_at_1'][str(seed)]['value']:.3f}" for seed in _SEEDS)
        + f" | {ci(order['shuffle_p_at_1'])} | {ci(order['pairwise_agreement'])} | "
        f"{ci(noise['pairwise_agreement'])} | {order['slots_1_to_3']['value']:.1%} (uniform 15%) |",
        "",
        "Pairwise agreement: how often two orders of one task pick the same tool (10 pairs per task). Same-order "
        "agreement: the same over three repeats in retrieval order, the run-to-run noise to compare it with.",
        "",
        "Failed searches, left out (Luna returned no content, with an empty refusal): "
        + "; ".join(
            f"{name} {section['errors']['searches']} ({len(section['errors']['tasks'])} tasks)"
            for name, section in (("first pick", first), ('"none"', none), ("orders", order), ("repeats", noise))
        )
        + f". Order and repeat comparisons keep the {order['positive_tasks']} and {noise['positive_tasks']} "
        "tasks that answered every time.",
        "",
        "## Reproduce",
        "",
        "```shell",
        "T=bench/tasks/toolret-heldout-200.json",
        "uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain "
        "--no-reserved --no-negatives --orders",
        "uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain",
        "uv run toolhunch-bench decision --tasks $T --split heldout --deciders luna --k 20 --sources plain "
        "--no-reserved --no-negatives --repeat 3 --bypass-cache",
        "uv run toolhunch-bench luna-report --orders bench/runs/ORDERS --none bench/runs/NONE "
        "--repeats bench/runs/REPEATS",
        "```",
        "",
    ]
    return "\n".join(lines)
