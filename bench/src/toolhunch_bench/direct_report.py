"""Metrics and publication for direct-choice runs, excluding replay usage from provider measurements."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from toolhunch_bench.direct import DIRECT_ARMS
from toolhunch_bench.metrics import Outcome, cluster_bootstrap_ci, percentile, selective_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def _interval(values: Sequence[tuple[str, float]]) -> dict[str, Any] | None:
    if not values:
        return None
    clusters: dict[str, list[float]] = defaultdict(list)
    for task, value in values:
        clusters[task].append(value)
    low, high = cluster_bootstrap_ci(list(clusters.values()), resamples=2000, seed=0)
    return {"value": sum(value for _, value in values) / len(values), "ci95": [low, high]}


def _cost(records: Sequence[dict[str, Any]], *, phase: str, arm: str) -> dict[str, Any]:
    own = [r for r in records if r["phase"] == phase and (r["provider_calls"] or arm == "hybrid@20")]
    calls = [call for r in own for call in r["provider_calls"]]
    input_tokens = sum(call["input_tokens"] or 0 for call in calls)
    cached = sum(call["cache_read_tokens"] or 0 for call in calls)
    billed = sum(call["usd"] or 0 for call in calls) + sum(r["search_usd"] for r in own)
    list_price = sum(call["list_usd"] or 0 for call in calls) + sum(r["search_usd"] for r in own)
    latencies = [r["search_seconds"] + (r["decision_seconds"] or 0) for r in own if r["error"] is None]
    return {
        "observed_searches": len(own),
        "provider_calls": len(calls),
        "replays": sum(r["replayed"] for r in records if r["phase"] == phase),
        "input_tokens": input_tokens,
        "cache_read_tokens": cached,
        "cache_share": cached / input_tokens if input_tokens else None,
        "billed_usd": billed if own else None,
        "list_usd": list_price if own else None,
        "billed_usd_per_1000": billed * 1000 / len(own) if own else None,
        "list_usd_per_1000": list_price * 1000 / len(own) if own else None,
        "latency_p50_ms": percentile(latencies, 50) * 1000 if latencies else None,
        "latency_p95_ms": percentile(latencies, 95) * 1000 if latencies else None,
        "unpriced_attempts": sum(call["usd"] is None for call in calls),
    }


def _row(
    records: Sequence[dict[str, Any]],
    *,
    arm: str,
    catalog: str,
    baseline: dict[tuple[str, str], bool],
) -> dict[str, Any]:
    positives = [r for r in records if r["variant"] == "positive"]
    parsed = [r for r in records if r["error"] is None and r["abstained"] is not None]

    def correct(r: dict[str, Any]) -> bool:
        return r["variant"] == "positive" and r["pick"] in r["relevant"] and r["error"] is None

    def task_id(r: dict[str, Any]) -> str:
        return f"{r['catalog']}:{r['task']}"

    outcomes = [
        Outcome(
            answered=not r["abstained"],
            correct=correct(r),
            gold_in_candidates=r["variant"] == "positive" and bool(set(r["candidates"]) & set(r["relevant"])),
        )
        for r in parsed
    ]
    none = asdict(selective_metrics(outcomes)) if outcomes else {}
    values: dict[str, list[tuple[str, float]]] = {
        field: []
        for field in (
            "coverage",
            "selective_accuracy",
            "wrong_tool_rate",
            "abstention_precision",
            "abstention_recall",
        )
    }
    for r, o in zip(parsed, outcomes, strict=True):
        key = task_id(r)
        values["coverage"].append((key, float(o.answered)))
        values["wrong_tool_rate"].append((key, float(o.answered and not o.correct)))
        if o.answered:
            values["selective_accuracy"].append((key, float(o.correct)))
        else:
            values["abstention_precision"].append((key, float(not o.gold_in_candidates)))
        if not o.gold_in_candidates:
            values["abstention_recall"].append((key, float(not o.answered)))
    for field, observations in values.items():
        none[field] = _interval(observations)
    none |= {
        "correct": sum(correct(r) for r in parsed),
        "wrong": sum(not r["abstained"] and not correct(r) for r in parsed),
        "abstained": sum(r["abstained"] for r in parsed),
        "requests": len(records),
        "parsed_requests": len(parsed),
        "errors": len(records) - len(parsed),
        "negative_requests": sum(r["variant"] == "negative" for r in records),
        "negative_share": sum(r["variant"] == "negative" for r in records) / len(records) if records else None,
    }
    return {
        "arm": arm,
        "catalog": catalog,
        "positives": len(positives),
        "relevant_pick_rate": _interval([(task_id(r), float(correct(r))) for r in positives]),
        "delta_vs_hybrid": _interval(
            [
                (task_id(r), float(correct(r)) - baseline[(r["catalog"], r["task"])])
                for r in positives
                if (r["catalog"], r["task"]) in baseline
            ]
        ),
        "ranking_ignoring_abstention": _interval(
            [
                (task_id(r), float(r["error"] is None and bool(r["ranked"]) and r["ranked"][0] in r["relevant"]))
                for r in positives
            ]
        )
        if arm in {"hybrid@20+jev", "jev-all"}
        else None,
        "none_option": none,
        "detail_counts": {level: sum(level in r["detail"] for r in records) for level in ("FULL", "BRIEF", "NAME")},
        "multi_calls": sum(r["extra_calls"] > 0 for r in records),
        "agent_text_none": sum(r.get("text_is_none") is True for r in parsed),
        "agent_other_text_abstentions": sum(r["abstained"] and r.get("text_is_none") is False for r in parsed),
        "positive_cost": {phase: _cost(records, phase=phase, arm=arm) for phase in ("cold", "warm")},
        "negative_cost": _cost(records, phase="negative", arm=arm),
    }


def build_direct_report(run_dir: Path, *, out_dir: Path) -> dict[str, Any]:
    """Generate a summary and public table from recorded requests; this never contacts a provider."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = [json.loads(line) for line in (run_dir / "run.jsonl").read_text().splitlines() if line.strip()]
    calls = [json.loads(line) for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]
    baseline = {
        (r["catalog"], r["task"]): r["error"] is None and r["pick"] in r["relevant"]
        for r in records
        if r["arm"] == "hybrid@20" and r["variant"] == "positive"
    }
    rows: list[dict[str, Any]] = []
    validation: dict[str, Any] = {}
    for arm in DIRECT_ARMS:
        own = [r for r in records if r["arm"] == arm]
        if not own:
            continue
        rows.append(_row(own, arm=arm, catalog="pooled", baseline=baseline))
        for selected in manifest["catalogs"]:
            subset = [r for r in own if r["catalog"] == selected["source"]]
            if subset:
                rows.append(_row(subset, arm=arm, catalog=selected["source"], baseline=baseline))
        successful = [r for r in own if r["error"] is None]
        actual = [call for call in calls if call["arm"] == arm and call["model"] != "text-embedding-3-small"]
        validation[arm] = {
            "requests": len(own),
            "errors": sum(r["error"] is not None for r in own),
            "error_rate": sum(r["error"] is not None for r in own) / len(own),
            "classified_successful": sum(r["abstained"] is not None for r in successful),
            "successful_requests": len(successful),
            "classified_fraction": sum(r["abstained"] is not None for r in successful) / len(successful)
            if successful
            else None,
            "provider_calls": len(actual),
            "replayed_requests": sum(r["replayed"] for r in own),
            "cache_fields_present": all("cache_read_tokens" in call for call in actual),
            "cache_usage_received": all(
                call["cache_read_tokens"] is not None for call in actual if call["error"] is None
            ),
        }
    actual_usd = sum(call["usd"] or 0 for call in calls)
    estimated = manifest["estimate"]["usd"]
    full_estimate = manifest.get("full_estimate", manifest["estimate"])["usd"]
    remaining, projection_method = _remaining_projection(manifest, records, calls)
    summary = {
        "schema_version": 1,
        "manifest": manifest,
        "rows": rows,
        "validation": validation,
        "run_cost": {
            "provider_attempts": len(calls),
            "verified_usd": actual_usd,
            "budget_charge_usd": sum(call["budget_charge_usd"] for call in calls),
            "unpriced_attempts": sum(call["usd"] is None for call in calls),
            "estimate_usd": estimated,
            "actual_to_estimate": actual_usd / estimated if estimated else None,
            "projected_remaining_usd": remaining,
            "projected_p1_total_usd": manifest.get("prior_p1_usd", 0) + actual_usd + remaining
            if remaining is not None
            else None,
            "projection_method": projection_method,
            "full_estimate_usd": full_estimate,
        },
        "caveats": [
            "The metric is selecting a relevant tool, not completing a task; tasks can list several relevant tools.",
            "Negatives remove the task's labeled relevant tools; an unlabeled alternative may still serve it.",
            "Pooled none-option metrics state their observed positive/negative mix; errors are reported separately.",
            "Replay outcomes are scored, but their historical usage, latency and cache reads are excluded.",
            "Search overhead is attributed to each strategy that searches; physical search calls are paid once.",
            "Cold is the first scored positive; a replay there provides no new cold measurement.",
            "Caching reflects one provider's routing in one single-turn run and does not isolate a latency effect.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "README.md").write_text(_markdown(summary))
    return summary


def _remaining_projection(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    calls: Sequence[dict[str, Any]],
) -> tuple[float | None, str]:
    if not manifest["pilot"]:
        return 0.0, "completed full run"
    if any(call["usd"] is None for call in calls):
        return None, "failed attempts have unknown billed usage"
    remaining = 0.0
    full_estimate = manifest.get("full_estimate", manifest["estimate"])
    for catalog in manifest.get("full_catalogs", manifest["catalogs"]):
        for arm in DIRECT_ARMS[1:]:
            for variant, phase, field in (("positive", "warm", "positives"), ("negative", "negative", "negatives")):
                own = [
                    r
                    for r in records
                    if r["catalog"] == catalog["source"]
                    and r["arm"] == arm
                    and r["variant"] == variant
                    and r["error"] is None
                ]
                count = max(0, catalog[field] - len(own))
                real = [r for r in own if r["phase"] == phase and r["provider_calls"]]
                if not count:
                    continue
                if not real:
                    return full_estimate["usd"], "conservative uncached estimate: a source/arm has no measured phase"
                paid = sum(call["usd"] or 0 for r in real for call in r["provider_calls"])
                remaining += paid / len(real) * count
    full_embedding = sum(line["usd"] for line in full_estimate["lines"] if line["model"] == "text-embedding-3-small")
    paid_embedding = sum(call["usd"] or 0 for call in calls if call["model"] == "text-embedding-3-small")
    remaining += max(0.0, full_embedding - paid_embedding)
    return remaining, "per-source warm-positive and negative observed rates; pilot requests replayed; embedding bound"


def _markdown(summary: dict[str, Any]) -> str:
    def interval(value: dict[str, Any] | None) -> str:
        return f"{value['value']:.3f} ({value['ci95'][0]:.3f} to {value['ci95'][1]:.3f})" if value else "n/a"

    def money(value: float | None) -> str:
        return f"${value:.4f}" if value is not None else "n/a"

    lines = [
        "# ToolRet direct choice",
        "",
        "Single-turn selection of a relevant tool from real source catalogs.",
        "",
        "95% intervals resample tasks together, 2,000 resamples, seed 0. Abstentions and errors are misses in the "
        "positive-request headline. None-option metrics use parsed requests; errors are listed separately.",
        "",
        "| Arm | Catalog | Relevant picks / positives | Paired Δ vs hybrid | Correct / wrong / abstained | "
        "Negatives | Errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        none = row["none_option"]
        lines.append(
            f"| {row['arm']} | {row['catalog']} | {interval(row['relevant_pick_rate'])} | "
            f"{interval(row['delta_vs_hybrid'])} | {none['correct']} / {none['wrong']} / {none['abstained']} | "
            f"{none['negative_requests']}/{none['requests']} ({none['negative_share']:.1%}) | {none['errors']} |"
        )
    lines += [
        "",
        "## Observed provider cost",
        "",
        "Replay responses do not contribute usage or timing. Search "
        "cost and latency are included for strategies that search. Negatives are priced separately.",
        "",
        "| Arm | Catalog | Cold billed | Warm billed / 1,000 | Warm list / 1,000 | Warm cache share | "
        "Warm latency p50 / p95 ms | Negative billed / 1,000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        cold, warm = row["positive_cost"]["cold"], row["positive_cost"]["warm"]
        cache = f"{warm['cache_share']:.1%}" if warm["cache_share"] is not None else "n/a"
        latency = (
            f"{warm['latency_p50_ms']:.1f} / {warm['latency_p95_ms']:.1f}"
            if warm["latency_p50_ms"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['arm']} | {row['catalog']} | {money(cold['billed_usd'])} | "
            f"{money(warm['billed_usd_per_1000'])} | {money(warm['list_usd_per_1000'])} | {cache} | {latency} | "
            f"{money(row['negative_cost']['billed_usd_per_1000'])} |"
        )
    lines += ["", "## Caveats", ""] + [f"- {caveat}" for caveat in summary["caveats"]]
    lines += [
        "",
        "## Reproduce",
        "",
        "```shell",
        "uv run toolhunch-bench direct --dry-run",
        "uv run toolhunch-bench direct --pilot",
        "uv run toolhunch-bench direct",
        "uv run toolhunch-bench direct-report bench/runs/RUN --out bench/results/2026-09-toolret-direct/",
        "```",
        "",
        f"Verified provider usage in this run: {money(summary['run_cost']['verified_usd'])}.",
        "",
        "Not affiliated with TypeSafe, OpenAI or Pydantic.",
        "",
    ]
    return "\n".join(lines)
